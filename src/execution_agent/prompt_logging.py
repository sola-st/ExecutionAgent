from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import contextvars


# Context vars so we can attribute LLM calls to the current cycle without modifying agent internals.
_CURRENT_CYCLE: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "_CURRENT_CYCLE", default=None
)
_CURRENT_CALL_IDX: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_CURRENT_CALL_IDX", default=0
)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_messages(obj: Any) -> List[Dict[str, Any]]:
    """
    Best-effort normalization for what gets passed into the underlying model.

    - If obj is already a ChatCompletions-style messages list (list[dict]) -> copy it.
    - If obj is a string -> treat as a single user message.
    - Else -> stringify as a single user message.

    This keeps the log robust across different model call shapes, while still
    capturing "exact prompt" for the common case (messages=list[dict]).
    """
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        # Shallow copy: preserves exactly what was sent, but avoids mutations after logging.
        return [dict(x) for x in obj]

    if isinstance(obj, str):
        return [{"role": "user", "content": obj}]

    return [{"role": "user", "content": str(obj)}]


def _render_messages_txt(messages: List[Dict[str, Any]]) -> str:
    """
    Human-readable rendering that preserves the exact content values.

    Note: if your messages contain non-string content (e.g., tool/vision parts),
    we still serialize them via str(...) for a readable .txt. The .json remains
    the authoritative structured representation.
    """
    out: List[str] = []
    for m in messages:
        role = str(m.get("role", "unknown"))
        name = m.get("name")
        tag = m.get("tag")

        header_bits = [role]
        if name:
            header_bits.append(f"name={name}")
        if tag:
            header_bits.append(f"tag={tag}")

        out.append(f"--- {' '.join(header_bits)} ---")
        out.append(str(m.get("content", "")))
        out.append("")  # spacer
    return "\n".join(out).rstrip() + "\n"


@dataclass
class CycleChatLogger:
    """
    Writes per-cycle LLM prompts to:
      <run_log_dir>/cycles_chats/cycle_XXX/llm_call_YY.{json,txt}

    The logging is "best-effort": failures never break the run.
    """
    run_log_dir: Path

    def __post_init__(self) -> None:
        self.cycles_dir = self.run_log_dir / "cycles_chats"
        _safe_mkdir(self.cycles_dir)

    @contextmanager
    def cycle(self, cycle_idx: int) -> Iterable[None]:
        tok_cycle = _CURRENT_CYCLE.set(int(cycle_idx))
        tok_call = _CURRENT_CALL_IDX.set(0)
        try:
            yield
        finally:
            _CURRENT_CYCLE.reset(tok_cycle)
            _CURRENT_CALL_IDX.reset(tok_call)

    def log_llm_prompt(
        self,
        *,
        messages: Any,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        cycle_idx = _CURRENT_CYCLE.get()
        if cycle_idx is None:
            # Only log when we're inside an active cycle context.
            return

        call_idx = _CURRENT_CALL_IDX.get() + 1
        _CURRENT_CALL_IDX.set(call_idx)

        cycle_dir = self.cycles_dir / f"cycle_{cycle_idx:03d}"
        _safe_mkdir(cycle_dir)

        msg_list = _as_messages(messages)

        meta_out = dict(meta or {})
        meta_out.setdefault("ts_utc", _utc_ts())
        meta_out.setdefault("cycle", cycle_idx)
        meta_out.setdefault("call_index", call_idx)

        payload = {"meta": meta_out, "messages": msg_list}

        json_path = cycle_dir / f"llm_call_{call_idx:02d}.json"
        txt_path = cycle_dir / f"llm_call_{call_idx:02d}.txt"

        # JSON: structured, authoritative
        try:
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        # TXT: readable transcript-like view
        try:
            header = (
                f"[ts_utc={meta_out.get('ts_utc')}] "
                f"cycle={cycle_idx} call={call_idx} "
                f"model={meta_out.get('model') or ''} "
                f"phase={meta_out.get('phase') or ''}\n\n"
            )
            txt_path.write_text(
                header + _render_messages_txt(msg_list),
                encoding="utf-8",
            )
        except Exception:
            pass


class LoggedModel:
    """
    Proxy/wrapper around the underlying model that logs the *exact* message payload
    sent to the LLM.

    Supports multiple common invocation styles:
      - model(messages)
      - model(messages=...)
      - model(prompt="...")  [treated as user content]
      - model.chat(messages=...)
      - model.complete(messages=...)
      - model.generate(messages=...)
      - model.query(messages=...)

    Any other attribute/method is delegated via __getattr__.
    """

    def __init__(
        self,
        inner: Any,
        *,
        prompt_logger: CycleChatLogger,
        model_name: str = "",
    ) -> None:
        self._inner = inner
        self._prompt_logger = prompt_logger
        self._model_name = (
            model_name
            or getattr(inner, "model_name", "")
            or getattr(inner, "model", "")
            or ""
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _log(
        self,
        messages: Any,
        *,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(extra or {})
        meta.setdefault("phase", phase)
        meta.setdefault("model", self._model_name)
        self._prompt_logger.log_llm_prompt(messages=messages, meta=meta)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Common patterns:
        #   model(messages, **kwargs)
        #   model(messages=[...], **kwargs)
        #   model(prompt="...", **kwargs)
        messages = None

        if args:
            messages = args[0]

        if "messages" in kwargs:
            messages = kwargs.get("messages")

        if messages is None and "prompt" in kwargs:
            messages = kwargs.get("prompt")

        if messages is not None:
            self._log(
                messages,
                phase="model.__call__",
                extra={"kwargs_keys": list(kwargs.keys())},
            )

        return self._inner(*args, **kwargs)

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if args else None)
        if messages is not None:
            self._log(messages, phase="model.chat", extra={"kwargs_keys": list(kwargs.keys())})
        return getattr(self._inner, "chat")(*args, **kwargs)

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if args else None)
        if messages is not None:
            self._log(messages, phase="model.complete", extra={"kwargs_keys": list(kwargs.keys())})
        return getattr(self._inner, "complete")(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if args else None)
        if messages is not None:
            self._log(messages, phase="model.generate", extra={"kwargs_keys": list(kwargs.keys())})
        return getattr(self._inner, "generate")(*args, **kwargs)

    def query(self, *args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if args else None)
        if messages is not None:
            self._log(messages, phase="model.query", extra={"kwargs_keys": list(kwargs.keys())})
        return getattr(self._inner, "query")(*args, **kwargs)


def install_cycle_prompt_logging(
    *,
    agent: Any,
    base_model: Any,
    run_log_dir: Path,
    model_name: str,
    logger: Optional[Any] = None,
) -> Tuple[Any, Path]:
    """
    Installs cycle-scoped prompt logging with minimal intrusion.

    What it does:
      1) Creates CycleChatLogger under run_log_dir
      2) Wraps base_model with LoggedModel
      3) Patches agent.run_one_cycle so that each call runs inside a cycle context
         (cycle counter is maintained locally here; no dependency on agent internals)

    Returns:
      (wrapped_model, run_log_dir)
    """
    run_log_dir = Path(run_log_dir)
    _safe_mkdir(run_log_dir)

    cycle_chat_logger = CycleChatLogger(run_log_dir=run_log_dir)
    wrapped_model = LoggedModel(base_model, prompt_logger=cycle_chat_logger, model_name=model_name)

    # Attach for anyone who wants to reference it later
    try:
        agent.cycle_chat_logger = cycle_chat_logger
        agent.run_log_root = str(run_log_dir)
    except Exception:
        pass

    # Patch run_one_cycle to activate the cycle context so all LLM calls inside are logged.
    orig = getattr(agent, "run_one_cycle", None)
    if orig is None:
        raise AttributeError("agent.run_one_cycle not found; cannot install cycle prompt logging.")

    cycle_counter = {"n": 0}

    def wrapped_run_one_cycle(*args: Any, **kwargs: Any) -> Any:
        cycle_counter["n"] += 1
        cycle_idx = cycle_counter["n"]

        if logger is not None:
            try:
                logger.info("CYCLE %03d START (prompt logging enabled)", cycle_idx)
            except Exception:
                pass

        with cycle_chat_logger.cycle(cycle_idx):
            return orig(*args, **kwargs)

    try:
        agent.run_one_cycle = wrapped_run_one_cycle  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"Failed to patch agent.run_one_cycle: {e}") from e

    return wrapped_model, run_log_dir
