#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import secrets
from typing import Any, Optional, Callable, Tuple


# Global reference for cleanup on shutdown
_active_agent: Optional[Any] = None
_shutdown_in_progress = False


def _cleanup_on_shutdown(signum=None, frame=None):
    """Clean up Docker containers and resources on shutdown."""
    global _shutdown_in_progress, _active_agent

    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    if signum:
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        print(f"\nReceived {sig_name}, cleaning up...", file=sys.stderr)

    if _active_agent is not None:
        try:
            from execution_agent.docker_helpers_static import cleanup_container
            container = getattr(_active_agent, 'container', None)
            docker_tag = getattr(_active_agent, 'docker_tag', None)
            if container or docker_tag:
                print("Cleaning up Docker container...", file=sys.stderr)
                cleanup_container(container, docker_tag)
                print("Cleanup complete.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}", file=sys.stderr)

    if signum:
        sys.exit(128 + signum)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, _cleanup_on_shutdown)
signal.signal(signal.SIGTERM, _cleanup_on_shutdown)
atexit.register(_cleanup_on_shutdown)

from execution_agent.prompt_logging import install_cycle_prompt_logging
from minisweagent.models.litellm_model import (
    LitellmModel,
    LLMTimeoutError,
    LLMDeadlineExceededError,
)

from execution_agent.agent import ExecutionAgent
from execution_agent.env import ExecutionEnvironment
from execution_agent.tools import (
    ToolRegistry,
    linux_terminal,
    read_file,
    write_to_file,
    search_docker_image,
    goals_accomplished,
)
from execution_agent.context import ContextBuilder


# -------------------------
# Run logging (text + jsonl + transcript)
# -------------------------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)


class JsonlLogHandler(logging.Handler):
    """
    Structured JSONL log handler for all log records.
    """
    def __init__(self, path: Path, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": _now_iso(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # If the caller included "extra" fields, capture selected ones
            for k in ("event", "tool_name", "tool_args", "image_tag"):
                if hasattr(record, k):
                    payload[k] = getattr(record, k)

            self._fh.write(_safe_json(payload) + "\n")
            self._fh.flush()
        except Exception:
            # Never crash the run because of logging
            pass

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
        super().close()


@dataclass
class TranscriptWriters:
    text_path: Path
    jsonl_path: Path
    json_path: Path

    def __post_init__(self) -> None:
        self._text_fh = open(self.text_path, "a", encoding="utf-8")
        self._jsonl_fh = open(self.jsonl_path, "a", encoding="utf-8")

    def write_message(self, index: int, msg: dict[str, Any]) -> None:
        ts = _now_iso()
        role = str(msg.get("role") or "").upper()
        tag = str(msg.get("tag") or "").strip()
        content = str(msg.get("content") or "")

        # Text transcript
        header = f"[{ts}] #{index:04d} {role}"
        if tag:
            header += f" (tag={tag})"
        self._text_fh.write(header + "\n")
        self._text_fh.write(content.rstrip("\n") + "\n\n")
        self._text_fh.flush()

        # JSONL transcript
        self._jsonl_fh.write(
            json.dumps(
                {"ts": ts, "index": index, "role": msg.get("role"), "tag": msg.get("tag"), "content": msg.get("content")},
                ensure_ascii=False,
            )
            + "\n"
        )
        self._jsonl_fh.flush()

    def finalize_full_json(self, messages: list[dict[str, Any]]) -> None:
        try:
            self.json_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._text_fh.close()
        except Exception:
            pass
        try:
            self._jsonl_fh.close()
        except Exception:
            pass


# -------------------------
# NEW: per-cycle LLM prompt logging (exact chat prompt payload)
# -------------------------

def _format_messages_as_text(messages: Any) -> str:
    """
    Human-readable, but content-preserving: shows each message role/tag and the exact content.
    """
    if not isinstance(messages, list):
        return str(messages)

    chunks: list[str] = []
    for i, m in enumerate(messages):
        if isinstance(m, dict):
            role = str(m.get("role") or "").strip()
            tag = str(m.get("tag") or "").strip()
            header = f"----- message[{i}] role={role or 'UNKNOWN'}"
            if tag:
                header += f" tag={tag}"
            header += " -----"
            chunks.append(header)
            # Preserve exact content
            content = m.get("content")
            if content is None:
                chunks.append("")
            else:
                chunks.append(str(content))
        else:
            chunks.append(f"----- message[{i}] (non-dict) -----")
            chunks.append(str(m))
    chunks.append("")  # trailing newline
    return "\n".join(chunks)


class CyclePromptLoggerModelProxy:
    """
    Wraps the model and writes the *exact* chat prompt sent to the LLM per cycle.

    Output:
      <run_dir>/cycles_chats/cycle_<N>/prompt_<K>.json
      <run_dir>/cycles_chats/cycle_<N>/prompt_<K>.txt

    Notes:
    - Uses agent._current_cycle_idx set by the run_one_cycle wrapper.
    - Detects "messages" in kwargs or first positional arg if it looks like a chat messages list.
    - Non-invasive: if anything goes wrong, it silently falls back to the underlying model call.
    """
    def __init__(self, base_model: Any, agent: Any, run_dir: Path, log: logging.Logger) -> None:
        self._base_model = base_model
        self._agent = agent
        self._run_dir = run_dir
        self._log = log

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._base_model, name)

        if not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any):
            self._maybe_log_prompt(args, kwargs)
            return attr(*args, **kwargs)

        return _wrapped

    def _maybe_log_prompt(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            messages = None

            # Common convention: messages passed as kwarg
            if "messages" in kwargs:
                messages = kwargs.get("messages")

            # Otherwise, if first arg looks like chat messages
            if messages is None and len(args) >= 1 and isinstance(args[0], list):
                if len(args[0]) == 0 or isinstance(args[0][0], dict):
                    messages = args[0]

            if messages is None:
                return

            cycle_idx = getattr(self._agent, "_current_cycle_idx", None)
            if not isinstance(cycle_idx, int) or cycle_idx <= 0:
                # If no cycle context, still log under cycle_0
                cycle_idx = 0

            cycles_root = self._run_dir / "cycles_chats" / f"cycle_{cycle_idx}"
            cycles_root.mkdir(parents=True, exist_ok=True)

            # Sequence number for multiple LLM calls within the same cycle
            seq_key = "_cycle_prompt_seq"
            seq_map = getattr(self._agent, seq_key, None)
            if not isinstance(seq_map, dict):
                seq_map = {}
                setattr(self._agent, seq_key, seq_map)

            k = int(seq_map.get(cycle_idx, 0)) + 1
            seq_map[cycle_idx] = k

            json_path = cycles_root / f"prompt_{k}.json"
            txt_path = cycles_root / f"prompt_{k}.txt"

            # Write exact JSON payload (as provided to the model)
            try:
                json_path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                # Fall back to safe string
                json_path.write_text(_safe_json(messages), encoding="utf-8")

            # Write text view (content-preserving)
            try:
                txt_path.write_text(_format_messages_as_text(messages), encoding="utf-8")
            except Exception:
                pass

            # Also reflect in the run log (path only; payload is in files)
            try:
                self._log.info(
                    "LLM prompt logged: %s",
                    str(txt_path),
                    extra={"event": "llm_prompt_logged"},
                )
            except Exception:
                pass

        except Exception:
            # Never break the run due to logging
            return


def _attach_cycle_and_transcript_logging(agent: ExecutionAgent, run_dir: Path, log: logging.Logger) -> None:
    """
    Non-invasive:
    - Wrap run_one_cycle() to log cycle boundaries and flush new agent.messages to transcript files.
    - Works even if agent does not expose add_message().
    """
    transcript = TranscriptWriters(
        text_path=run_dir / "messages_transcript.txt",
        jsonl_path=run_dir / "messages_transcript.jsonl",
        json_path=run_dir / "messages.json",
    )

    setattr(agent, "_transcript_writers", transcript)
    setattr(agent, "_messages_logged_upto", 0)

    def _flush_new_messages() -> None:
        try:
            msgs = list(getattr(agent, "messages", []) or [])
            upto = int(getattr(agent, "_messages_logged_upto", 0) or 0)
            if upto < 0:
                upto = 0
            for i in range(upto, len(msgs)):
                m = msgs[i]
                if isinstance(m, dict):
                    transcript.write_message(i, m)
            setattr(agent, "_messages_logged_upto", len(msgs))
        except Exception:
            pass

    # Initial flush (system+initial context if any already present)
    _flush_new_messages()

    if not hasattr(agent, "run_one_cycle"):
        log.warning("ExecutionAgent has no run_one_cycle(); cycle-level logging will rely on tool logs only.")
        return

    orig = agent.run_one_cycle

    def _flush_log_handlers():
        """Flush all handlers to ensure logs are written immediately."""
        for handler in log.handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def wrapped_run_one_cycle(*args: Any, **kwargs: Any):
        cycle_idx = getattr(agent, "cycle_count", 0) + 1

        # NEW: publish current cycle index for the model proxy logger
        try:
            setattr(agent, "_current_cycle_idx", int(cycle_idx))
        except Exception:
            pass

        log.info("────────────────────────────────────────────────────────────────────────────")
        log.info("CYCLE %02d START", cycle_idx, extra={"event": "cycle_start"})
        _flush_log_handlers()

        # Flush any messages queued before the cycle starts
        _flush_new_messages()

        res = orig(*args, **kwargs)

        # Flush messages produced by the cycle
        _flush_new_messages()

        # After-cycle summary if the agent returns a dict-ish cycle record
        try:
            if isinstance(res, dict):
                tool_call = res.get("tool_call") or {}
                cmd = tool_call.get("command") if isinstance(tool_call, dict) else None
                if isinstance(cmd, dict) and cmd.get("name"):
                    log.info(
                        "CYCLE %02d TOOL: %s args=%s",
                        cycle_idx,
                        cmd.get("name"),
                        _safe_json(cmd.get("args")),
                        extra={"event": "cycle_tool", "tool_name": cmd.get("name"), "tool_args": cmd.get("args")},
                    )
                if "result" in res:
                    preview = str(res.get("result"))
                    if len(preview) > 1200:
                        preview = preview[:1199].rstrip() + "…"
                    log.info("CYCLE %02d RESULT (preview): %s", cycle_idx, preview, extra={"event": "cycle_result"})
        except Exception:
            pass

        log.info("CYCLE %02d END", cycle_idx, extra={"event": "cycle_end"})
        _flush_log_handlers()

        # NEW: clear current cycle marker (best-effort)
        try:
            setattr(agent, "_current_cycle_idx", None)
        except Exception:
            pass

        return res

    def wrapped_run_one_cycle_with_error_handling(*args: Any, **kwargs: Any):
        """Wrapper that ensures exceptions are logged before propagating."""
        try:
            return wrapped_run_one_cycle(*args, **kwargs)
        except Exception as e:
            log.error(f"Exception in cycle execution: {type(e).__name__}: {e}", exc_info=True)
            _flush_log_handlers()
            raise

    agent.run_one_cycle = wrapped_run_one_cycle_with_error_handling  # type: ignore[attr-defined]


def _configure_logging(run_dir: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("execution_agent")
    logger.setLevel(level)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)

    run_dir.mkdir(parents=True, exist_ok=True)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)

    class _Formatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            ts = self.formatTime(record, datefmt="%H:%M:%S")
            lvl = record.levelname.ljust(5)
            return f"{ts} | {lvl} | {record.name} | {record.getMessage()}"

    console.setFormatter(_Formatter())
    logger.addHandler(console)

    # Text file handler
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(_Formatter())
    logger.addHandler(file_handler)

    # JSONL handler (structured)
    jsonl_handler = JsonlLogHandler(run_dir / "run.jsonl", level=level)
    logger.addHandler(jsonl_handler)

    # Ensure sub-loggers propagate into execution_agent
    for sub in ("execution_agent.tools", "execution_agent.docker", "litellm_model"):
        l = logging.getLogger(sub)
        l.setLevel(level)
        l.propagate = True

    # Also configure litellm_model logger to output to the same handlers
    litellm_logger = logging.getLogger("litellm_model")
    litellm_logger.setLevel(level)
    for h in logger.handlers:
        litellm_logger.addHandler(h)

    return logger


# -------------------------
# CLI
# -------------------------

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run ExecutionAgent (mini-swe-agent based, Option A env).")
    ap.add_argument("--experiment-file", required=True, help="Path to project_meta_data.json")
    ap.add_argument("--task-file", default=None, help="Optional file containing the top-level task/instructions.")
    ap.add_argument("--task", default=None, help="Optional task string. Overrides --task-file if set.")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-nano"))
    ap.add_argument("--knowledge-model", default=os.getenv("KNOWLEDGE_MODEL", "gpt-5-mini"),
                    help="Model for web search analysis and unified summary (default: gpt-5-mini). "
                         "Should be an up-to-date model with good general knowledge.")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    ap.add_argument("--workspace-root", default="execution_agent_workspace")
    ap.add_argument("--prompt-files", default="src/execution_agent/prompt_files")
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")

    # NEW: optional explicit run log dir
    ap.add_argument("--run-log-dir", default=None, help="Optional directory to write run logs/transcripts into.")

    # NEW: max retries for budget exhaustion
    ap.add_argument("--max-retries", type=int, default=2,
                    help="Maximum retries after budget exhaustion (default: 2). Total attempts = 1 + max_retries.")
    return ap.parse_args()


def build_default_task(meta: dict) -> str:
    return (
        "Your objective is to set up, build, install, and run the project's test suite inside a container. "
        "You must produce a Dockerfile (written via write_to_file) that clones the repo and prepares the environment, "
        "then run installation/build/test commands via linux_terminal until tests can be executed, and write "
        "TEST_RESULTS.txt with outcomes. Only declare goals_accomplished once Dockerfile exists and results are recorded.\n\n"
        "IMPORTANT: The task is considered successful if ~80% or more of the tests pass. "
        "Having a few failing tests or errors is acceptable and expected. "
        "Once you have a substantial majority of tests passing (~80%+), declare goals accomplished. "
        "Do NOT waste cycles trying to fix the last few failing tests."
    )


def _extract_dockerfile_and_script(llm_response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract Dockerfile and bash script from LLM response.

    Expected format in response:
    ```dockerfile
    <dockerfile content>
    ```

    ```bash
    <bash script content>
    ```

    Returns:
        Tuple of (dockerfile_content, bash_script_content), either can be None if not found
    """
    import re

    dockerfile = None
    bash_script = None

    # Extract Dockerfile - look for ```dockerfile or ```Dockerfile
    dockerfile_match = re.search(
        r'```[Dd]ockerfile\s*\n(.*?)```',
        llm_response,
        re.DOTALL
    )
    if dockerfile_match:
        dockerfile = dockerfile_match.group(1).strip()

    # Extract bash script - look for ```bash or ```sh
    bash_match = re.search(
        r'```(?:bash|sh)\s*\n(.*?)```',
        llm_response,
        re.DOTALL
    )
    if bash_match:
        bash_script = bash_match.group(1).strip()

    return dockerfile, bash_script


def _run_forced_exit_cycle(
    *,
    knowledge_model,
    agent,
    project_path: str,
    project_url: str,
    workspace_root: str,
    run_dir: Path,
    log: logging.Logger,
) -> bool:
    """
    Forced exit cycle: Ask knowledge model to produce a final Dockerfile + bash script
    based on all available context, then attempt to execute them.

    Args:
        knowledge_model: The knowledge model (e.g., gpt-5-mini) to use
        agent: The agent instance with history and context
        project_path: Name/path of the project
        project_url: Git URL of the project
        workspace_root: Root workspace directory
        run_dir: Directory for logs and output files
        log: Logger instance

    Returns:
        True if the forced exit cycle succeeded (tests ran), False otherwise
    """
    log.info("=" * 80)
    log.info("🚨 FORCED EXIT CYCLE - Attempting final solution with knowledge model")
    log.info("=" * 80)

    # Gather all context for the knowledge model
    # 1. Previous attempt lessons
    previous_lessons = []
    for i, attempt in enumerate(getattr(agent, 'previous_attempts', []), 1):
        if isinstance(attempt, dict):
            previous_lessons.append(f"Attempt {i}: {json.dumps(attempt, indent=2)}")

    # 2. Command history from current/last attempt
    command_history = []
    for cmd_summary in getattr(agent, 'commands_and_summary', [])[-50:]:  # Last 50 commands
        if isinstance(cmd_summary, dict):
            cmd = cmd_summary.get('command', '')
            result = cmd_summary.get('result', '')[:500]  # Truncate long results
            command_history.append(f"$ {cmd}\n{result}")
        elif isinstance(cmd_summary, str):
            command_history.append(cmd_summary)

    # 3. Workflow/CI hints from repo context
    workflow_hints = ""
    repo_context = getattr(agent, 'repo_context', None)
    if repo_context:
        for path, content in getattr(repo_context, 'workflow_contents', [])[:3]:
            workflow_hints += f"\n--- {path} ---\n{content[:3000]}\n"

        unified_summary = getattr(repo_context, 'unified_summary', '')
        if unified_summary:
            workflow_hints += f"\n--- Unified Summary ---\n{unified_summary[:5000]}\n"

    # Build the prompt for the knowledge model
    forced_exit_prompt = f"""You are an expert at setting up software projects for testing. An automated agent has failed multiple times to install and test the project "{project_path}" from source. You need to produce a FINAL SOLUTION.

PROJECT INFORMATION:
- Project: {project_path}
- Repository URL: {project_url}
- Language: {getattr(repo_context, 'language', 'unknown') if repo_context else 'unknown'}

PREVIOUS ATTEMPT LESSONS:
{chr(10).join(previous_lessons) if previous_lessons else "No previous attempt summaries available."}

RECENT COMMAND HISTORY (what was tried):
{chr(10).join(command_history[-30:]) if command_history else "No command history available."}

CI/CD AND BUILD HINTS FROM REPOSITORY:
{workflow_hints if workflow_hints else "No CI/CD hints available."}

YOUR TASK:
Based on ALL the information above, produce:
1. A complete Dockerfile that will set up the environment for building and testing this project
2. A bash script that will run inside the container to install dependencies, build the project, and run tests

IMPORTANT REQUIREMENTS:
- The Dockerfile should be based on Ubuntu (ubuntu:22.04 or ubuntu:24.04)
- The Dockerfile must install git and clone the repository
- The bash script should be executable inside the container
- Include all necessary system dependencies
- Handle common issues that were encountered in previous attempts
- The bash script should end with running the test suite
- If tests fail, that's acceptable - we just need to run them

OUTPUT FORMAT:
You MUST provide your response in EXACTLY this format:

```dockerfile
<your complete Dockerfile here>
```

```bash
<your complete bash script here>
```

Provide ONLY these two code blocks. The Dockerfile and bash script must be complete and ready to use."""

    # Try up to 3 times to get a valid response
    max_llm_retries = 3
    dockerfile_content = None
    bash_script_content = None

    for llm_attempt in range(1, max_llm_retries + 1):
        log.info(f"🤖 Querying knowledge model (attempt {llm_attempt}/{max_llm_retries})...")
        try:
            response = knowledge_model.query([{"role": "user", "content": forced_exit_prompt}])
            llm_response = response.get("content", "")

            # Extract Dockerfile and bash script
            dockerfile_content, bash_script_content = _extract_dockerfile_and_script(llm_response)

            if dockerfile_content and bash_script_content:
                log.info("✅ Successfully extracted Dockerfile and bash script from LLM response")
                break
            else:
                missing = []
                if not dockerfile_content:
                    missing.append("Dockerfile")
                if not bash_script_content:
                    missing.append("bash script")
                log.warning(f"⚠️ LLM response missing: {', '.join(missing)}")
                if llm_attempt < max_llm_retries:
                    log.info("Retrying with clarification...")
                    forced_exit_prompt += "\n\nIMPORTANT: Your previous response was missing required components. Please provide BOTH a ```dockerfile block AND a ```bash block."

        except Exception as e:
            log.error(f"❌ LLM query failed: {e}")
            if llm_attempt >= max_llm_retries:
                log.error("All LLM query attempts failed")
                return False

    if not dockerfile_content or not bash_script_content:
        log.error("❌ Failed to extract valid Dockerfile and bash script after all retries")
        return False

    # Save the extracted files
    forced_exit_dir = run_dir / "forced_exit_cycle"
    forced_exit_dir.mkdir(parents=True, exist_ok=True)

    dockerfile_path = forced_exit_dir / "Dockerfile"
    script_path = forced_exit_dir / "run_tests.sh"

    dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
    script_path.write_text(bash_script_content, encoding="utf-8")
    log.info(f"📄 Saved Dockerfile to: {dockerfile_path}")
    log.info(f"📄 Saved bash script to: {script_path}")

    # Log the contents for debugging
    log.info("--- Dockerfile content ---")
    for line in dockerfile_content.split('\n')[:30]:
        log.info(f"  {line}")
    if dockerfile_content.count('\n') > 30:
        log.info(f"  ... ({dockerfile_content.count(chr(10)) - 30} more lines)")

    log.info("--- Bash script content ---")
    for line in bash_script_content.split('\n')[:30]:
        log.info(f"  {line}")
    if bash_script_content.count('\n') > 30:
        log.info(f"  ... ({bash_script_content.count(chr(10)) - 30} more lines)")

    # Now attempt to build and run
    log.info("🔨 Building Docker image from forced exit Dockerfile...")

    try:
        from execution_agent.docker_helpers_static import cleanup_container
        import subprocess
        import secrets

        # Generate a unique tag for this forced exit attempt
        docker_tag = f"forced_exit_{project_path}_{secrets.token_hex(4)}"

        # Build the Docker image
        build_result = subprocess.run(
            ["docker", "build", "-t", docker_tag, "-f", str(dockerfile_path), str(forced_exit_dir)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for build
            cwd=str(forced_exit_dir),
        )

        if build_result.returncode != 0:
            log.error(f"❌ Docker build failed:\n{build_result.stderr}")
            # Save build output for debugging
            (forced_exit_dir / "docker_build.log").write_text(
                f"STDOUT:\n{build_result.stdout}\n\nSTDERR:\n{build_result.stderr}",
                encoding="utf-8"
            )
            return False

        log.info("✅ Docker image built successfully")

        # Run the container with the bash script
        log.info("🚀 Running test script inside container...")

        # Copy the script into the container and execute it
        run_result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{script_path}:/run_tests.sh:ro",
                docker_tag,
                "bash", "/run_tests.sh"
            ],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout for tests
        )

        # Save the output
        test_output = f"STDOUT:\n{run_result.stdout}\n\nSTDERR:\n{run_result.stderr}\n\nReturn code: {run_result.returncode}"
        (forced_exit_dir / "test_output.log").write_text(test_output, encoding="utf-8")

        log.info(f"📋 Test execution completed with return code: {run_result.returncode}")
        log.info("--- Test output (last 50 lines) ---")
        output_lines = (run_result.stdout + run_result.stderr).split('\n')
        for line in output_lines[-50:]:
            log.info(f"  {line}")

        # Cleanup the Docker image
        try:
            subprocess.run(["docker", "rmi", docker_tag], capture_output=True, timeout=60)
        except Exception:
            pass

        # Consider it a success if the script ran (even if tests failed)
        # The goal is to at least execute the test suite
        if run_result.returncode == 0:
            log.info("✅ Forced exit cycle: Tests completed successfully!")
            return True
        else:
            log.warning(f"⚠️ Forced exit cycle: Tests ran but exited with code {run_result.returncode}")
            # Still consider it partially successful - tests ran
            return True

    except subprocess.TimeoutExpired as e:
        log.error(f"❌ Timeout during forced exit cycle: {e}")
        return False
    except Exception as e:
        log.error(f"❌ Error during forced exit cycle: {e}", exc_info=True)
        return False


def main() -> int:
    args = parse_args()

    # Load meta early so we can compute run_dir before configuring logging
    meta = json.loads(Path(args.experiment_file).read_text(encoding="utf-8"))
    project_path = meta["project_path"]
    project_url = meta["project_url"]
    language = meta.get("language", "unknown")

    # Run directory for logs/transcripts
    if args.run_log_dir:
        run_dir = Path(args.run_log_dir)
    else:
        safe_proj = str(project_path).replace(os.sep, "__").replace("/", "__")
        run_dir = Path(args.workspace_root) / "_run_logs" / safe_proj / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Configure logging (console + file + jsonl)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    LOG = _configure_logging(run_dir, level=level)

    # Task selection
    if args.task:
        task = args.task
    elif args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8", errors="ignore")
    else:
        task = build_default_task(meta)

    # API key
    if not args.api_key:
        raise SystemExit("Missing OPENAI_API_KEY (or pass --api-key).")
    os.environ["OPENAI_API_KEY"] = args.api_key

    # =========================================================================
    # PREPARATION PHASE - Collecting context and building main prompt
    # =========================================================================
    LOG.info("=" * 80)
    LOG.info("PREPARATION PHASE - Collecting context and building main prompt")
    LOG.info("=" * 80)

    LOG.info("Project: %s", project_path)
    LOG.info("Repo:    %s", project_url)
    LOG.info("Model:   %s", args.model)
    LOG.info("Knowledge Model: %s", args.knowledge_model)
    LOG.info("Run dir: %s", str(run_dir))

    # Load prompt snippets
    LOG.info("Loading prompt templates...")
    pf = Path(args.prompt_files)
    cycle_instruction = load_text(str(pf / "cycle_instruction"))
    summarize_cycle = load_text(str(pf / "summarize_cycle"))
    search_workflows_summary = load_text(str(pf / "search_workflows_summary"))
    remove_progress_bars_prompt = load_text(str(pf / "remove_progress_bars"))

    LOG.info("Initializing models...")
    model = LitellmModel(model_name=args.model, model_kwargs={})

    # Create a separate knowledge model for web search analysis and unified summary
    # This model should be up-to-date and knowledgeable about current technologies
    knowledge_model = LitellmModel(model_name=args.knowledge_model, model_kwargs={})

    LOG.info("Registering tools...")
    commands_schema = {
        "linux_terminal": ["command"],
        "read_file": ["file_path"],
        "write_to_file": ["filename", "text"],
        "search_docker_image": ["search_term"],
        "goals_accomplished": ["reason"],
    }

    tool_registry = ToolRegistry(commands_schema)
    tool_registry.register("linux_terminal", linux_terminal)
    tool_registry.register("read_file", read_file)
    tool_registry.register("write_to_file", write_to_file)
    tool_registry.register("search_docker_image", search_docker_image)
    tool_registry.register("goals_accomplished", goals_accomplished)

    def local_shell_interact(cmd: str):
        import subprocess

        cwd = Path(args.workspace_root) / project_path
        p = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return out, str(cwd)

    env = ExecutionEnvironment(
        workspace_path=args.workspace_root,
        project_path=project_path,
        shell_interact_fn=local_shell_interact,
    )

    LOG.info("Building repository context (cloning repo, finding workflows, requirements, README)...")
    ctx_builder = ContextBuilder(workspace_root=args.workspace_root)
    repo_context = ctx_builder.build_repo_context(
        model=model,
        knowledge_model=knowledge_model,
        project_path=project_path,
        project_url=project_url,
        language=language,
        search_workflows_summary_prompt=search_workflows_summary,
    )
    LOG.info("Repository context built successfully")

    tools_doc_path = pf / "tools_list"
    tools_doc_string = load_text(str(tools_doc_path)) if tools_doc_path.exists() else ""

    # Load language-specific guidelines if available
    language_guidelines = ""
    if language:
        lang_lower = language.lower().strip()
        # Map common language names to guideline file names
        lang_map = {
            "python": "python_guidelines",
            "py": "python_guidelines",
            "java": "java_guidelines",
            "javascript": "javascript_guidelines",
            "js": "javascript_guidelines",
            "typescript": "javascript_guidelines",
            "ts": "javascript_guidelines",
            "c": "c_guidelines",
            "c++": "cpp_guidelines",
            "cpp": "cpp_guidelines",
            "rust": "rust_guidelines",
            "rs": "rust_guidelines",
        }
        guideline_name = lang_map.get(lang_lower, f"{lang_lower}_guidelines")
        guideline_path = pf / guideline_name
        if guideline_path.exists():
            language_guidelines = load_text(str(guideline_path))
            LOG.info(f"Loaded language guidelines for: {language} from {guideline_name}")
        else:
            LOG.info(f"No language guidelines found for: {language} (tried {guideline_name})")

    agent = ExecutionAgent(
        model=model,
        env=env,
        tool_registry=tool_registry,
        cycle_instruction=cycle_instruction,
        summarize_cycle=summarize_cycle,
        remove_progress_bars_prompt=remove_progress_bars_prompt,
        search_workflows_summary_prompt=search_workflows_summary,
        step_limit=int(meta.get("budget", 40)) if isinstance(meta, dict) else 40,
    )

    # Attach runtime metadata/state
    agent.workspace_path = args.workspace_root
    agent.project_path = project_path
    agent.project_url = project_url
    agent.hyperparams = meta
    agent.repo_context = repo_context
    agent.tools_doc_string = tools_doc_string
    agent.language_guidelines = language_guidelines
    agent.written_files = []
    agent.commands_and_summary = []

    # State used by the upgraded tools
    agent.command_stuck = False
    agent.current_logfile = None
    agent.stuck_commands = []
    agent.docker_tag = ""

    # Register agent for graceful shutdown cleanup
    global _active_agent
    _active_agent = agent

    # Set up state persistence for recovery
    from execution_agent.state_persistence import create_state_persistence
    state_persistence = create_state_persistence(run_dir)
    agent._state_persistence = state_persistence

    # Check for existing state to resume from
    if state_persistence.has_saved_state():
        saved_state = state_persistence.load_state()
        if saved_state and saved_state.cycle_count > 0:
            LOG.info(f"Found saved state from cycle {saved_state.cycle_count}")
            # For now, just log - actual restoration would need more work
            # to handle container reconnection etc.
            LOG.info("To implement: automatic state restoration")

    # NEW: cycle + message transcript logging
    _attach_cycle_and_transcript_logging(agent, run_dir, LOG)

    # NEW: wrap the model so we persist the exact chat prompt sent to the LLM per cycle
    try:
        agent.model = CyclePromptLoggerModelProxy(agent.model, agent, run_dir, LOG)
    except Exception:
        pass

    # Optional: also attempt to enable any built-in prompt logging helper (best-effort, non-fatal)
    try:
        # Some repos expose install_cycle_prompt_logging; if present and compatible, enable it as well.
        install_cycle_prompt_logging(agent=agent, run_dir=run_dir, logger=LOG)  # type: ignore[call-arg]
    except TypeError:
        try:
            install_cycle_prompt_logging(agent, run_dir)  # type: ignore[misc]
        except Exception:
            pass
    except Exception:
        pass

    LOG.info("=" * 80)
    LOG.info("PREPARATION PHASE COMPLETE")
    LOG.info("=" * 80)

    # Import BudgetExhausted exception and cleanup function
    from execution_agent.exceptions import BudgetExhausted
    from execution_agent.docker_helpers_static import cleanup_container

    # =========================================================================
    # MAIN PHASE - Agent execution cycles
    # =========================================================================
    LOG.info("=" * 80)
    LOG.info("MAIN PHASE - Starting agent execution cycles")
    LOG.info("=" * 80)

    LOG.info("Starting agent run with retry support...")
    max_attempts = 1 + args.max_retries
    LOG.info(f"Configuration: max_attempts={max_attempts} (1 initial + {args.max_retries} retries)")

    final_success = False

    for attempt in range(1, max_attempts + 1):
        LOG.info("=" * 80)
        LOG.info(f"ATTEMPT {attempt} of {max_attempts}")
        LOG.info("=" * 80)

        try:
            agent.run(task=task)
            # Success!
            LOG.info(f"✅ Goals accomplished on attempt {attempt}")
            final_success = True

            # Generate exit artifacts for successful (non-forced) exit
            try:
                from execution_agent.exit_artifacts import generate_exit_artifacts
                LOG.info("📦 Generating exit artifacts for successful run...")
                artifacts_generated = generate_exit_artifacts(agent, run_dir, LOG)
                if artifacts_generated:
                    LOG.info("✅ Exit artifacts generated successfully")
                else:
                    LOG.warning("⚠️ Could not generate exit artifacts (no Dockerfile found)")
            except Exception as artifact_error:
                LOG.warning(f"⚠️ Failed to generate exit artifacts: {artifact_error}")

            break

        except BudgetExhausted as e:
            LOG.warning(f"⚠️  Attempt {attempt} exhausted budget: {e}")

            # Generate attempt summary for this attempt (needed for forced exit cycle too)
            LOG.info("📊 Generating summary of failed attempt...")
            try:
                summary = agent.generate_attempt_summary()
                LOG.info(f"Attempt {attempt} summary:")
                LOG.info(json.dumps(summary, indent=2))
                agent.previous_attempts.append(summary)
            except Exception as summary_error:
                LOG.error(f"Failed to generate attempt summary: {summary_error}")
                agent.previous_attempts.append({
                    "problems": "Summary generation failed",
                    "actions": f"Executed {len(agent.commands_and_summary)} commands",
                    "lessons": "Unable to extract detailed lessons",
                    "suggestions": "Try a different approach; review logs manually"
                })

            # If this was the last attempt, try forced exit cycle
            if attempt >= max_attempts:
                LOG.error(f"❌ All {max_attempts} retry attempts exhausted")

                # Try forced exit cycle with knowledge model
                LOG.info("🚨 Attempting forced exit cycle with knowledge model...")
                try:
                    forced_exit_success = _run_forced_exit_cycle(
                        knowledge_model=knowledge_model,
                        agent=agent,
                        project_path=project_path,
                        project_url=project_url,
                        workspace_root=args.workspace_root,
                        run_dir=run_dir,
                        log=LOG,
                    )
                    if forced_exit_success:
                        LOG.info("✅ Forced exit cycle succeeded!")
                        final_success = True
                except Exception as forced_exit_error:
                    LOG.error(f"❌ Forced exit cycle failed: {forced_exit_error}", exc_info=True)

                break

            # Cleanup Docker resources (summary was already generated above)
            LOG.info("🧹 Cleaning up Docker resources...")
            cleanup_container(agent.container, agent.docker_tag)

            # Reset agent state for next attempt
            LOG.info("🔄 Resetting agent state for next attempt...")

            # Save state that must be preserved
            saved_attempts = list(agent.previous_attempts)
            saved_model = agent.model._base_model if hasattr(agent.model, '_base_model') else agent.model
            saved_env = agent.env
            saved_tool_registry = agent.tool_registry
            saved_workspace_path = agent.workspace_path
            saved_project_path = agent.project_path
            saved_project_url = agent.project_url
            saved_hyperparams = agent.hyperparams
            saved_repo_context = agent.repo_context
            saved_tools_doc_string = agent.tools_doc_string
            saved_language_guidelines = agent.language_guidelines
            saved_cycle_instruction = agent.cycle_instruction
            saved_summarize_cycle = agent.summarize_cycle
            saved_remove_progress_bars_prompt = agent.remove_progress_bars_prompt
            saved_search_workflows_summary_prompt = agent.search_workflows_summary_prompt
            saved_step_limit = agent.step_limit

            # Reset volatile state
            agent.commands_and_summary = []
            agent.written_files = []
            agent.messages = []
            agent.cycle_count = 0
            agent.last_action = None
            agent.last_result = None
            agent.last_thoughts = None
            agent.last_format_error = None
            agent._last_failed_response = None
            agent.command_stuck = False
            agent.current_logfile = None
            agent.stuck_commands = []
            agent.container = None
            agent.docker_tag = ""

            # Restore preserved state
            agent.previous_attempts = saved_attempts
            agent.model = saved_model
            agent.env = saved_env
            agent.tool_registry = saved_tool_registry
            agent.workspace_path = saved_workspace_path
            agent.project_path = saved_project_path
            agent.project_url = saved_project_url
            agent.hyperparams = saved_hyperparams
            agent.repo_context = saved_repo_context
            agent.tools_doc_string = saved_tools_doc_string
            agent.language_guidelines = saved_language_guidelines
            agent.cycle_instruction = saved_cycle_instruction
            agent.summarize_cycle = saved_summarize_cycle
            agent.remove_progress_bars_prompt = saved_remove_progress_bars_prompt
            agent.search_workflows_summary_prompt = saved_search_workflows_summary_prompt
            agent.step_limit = saved_step_limit

            # Reset environment container reference if it exists
            if hasattr(agent.env, 'container'):
                agent.env.container = None

            # Reinitialize logging wrappers for new attempt
            LOG.info("📝 Reinitializing logging wrappers...")
            _attach_cycle_and_transcript_logging(agent, run_dir, LOG)
            try:
                agent.model = CyclePromptLoggerModelProxy(agent.model, agent, run_dir, LOG)
            except Exception as wrap_err:
                LOG.warning(f"Failed to reinitialize model wrapper: {wrap_err}")

            LOG.info(f"✓ State reset complete. Preserved {len(saved_attempts)} previous attempt summaries.")
            LOG.info(f"Starting attempt {attempt + 1}...")

        except Exception as e:
            LOG.error(f"❌ Unexpected error during attempt {attempt}: {e}", exc_info=True)
            break

        finally:
            # Persist messages for this attempt
            try:
                tw = getattr(agent, "_transcript_writers", None)
                if tw is not None:
                    msgs = list(getattr(agent, "messages", []) or [])
                    tw.finalize_full_json(msgs)
            except Exception:
                pass

    # Final cleanup
    try:
        tw = getattr(agent, "_transcript_writers", None)
        if tw is not None:
            tw.close()
    except Exception:
        pass

    # Generate bash script from trace
    try:
        from execution_agent.trace_to_bash import save_bash_script_from_agent

        bash_script_path = run_dir / "replay_trace.sh"
        LOG.info("Generating bash script from execution trace...")
        save_bash_script_from_agent(agent, bash_script_path)
        LOG.info(f"✓ Bash script saved to: {bash_script_path}")
    except Exception as e:
        LOG.warning(f"Failed to generate bash script from trace: {e}")

    # Save tool execution metrics
    try:
        from execution_agent.shared_utils import get_metrics_collector
        metrics = get_metrics_collector().get_all_metrics()
        if metrics:
            metrics_path = run_dir / "tool_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            LOG.info(f"✓ Tool metrics saved to: {metrics_path}")

            # Log summary of tool metrics
            for tool_name, tool_metrics in metrics.items():
                LOG.info(
                    f"  {tool_name}: {tool_metrics['total_calls']} calls, "
                    f"{tool_metrics['success_rate_percent']:.1f}% success, "
                    f"avg {tool_metrics['avg_duration_seconds']:.2f}s"
                )
    except Exception as e:
        LOG.warning(f"Failed to save tool metrics: {e}")

    if final_success:
        LOG.info("=" * 80)
        LOG.info("🎉 Agent run completed successfully!")
        LOG.info("=" * 80)
        return 0
    else:
        LOG.error("=" * 80)
        LOG.error("💥 Agent run failed to accomplish goals")
        LOG.error("=" * 80)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
