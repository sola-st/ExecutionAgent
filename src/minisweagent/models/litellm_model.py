import json
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import litellm
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

logger = logging.getLogger("litellm_model")


class LLMTimeoutError(Exception):
    """Raised when LLM call times out (backup timeout mechanism)."""
    pass


class LLMDeadlineExceededError(Exception):
    """Raised when the overall deadline for LLM query (including all retries) is exceeded."""
    pass


class LitellmModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    timeout: int = int(os.getenv("EXECUTION_AGENT_LLM_TIMEOUT", "300"))
    """Timeout in seconds for LLM API calls (default: 300s / 5 minutes)"""
    query_deadline: int = int(os.getenv("EXECUTION_AGENT_LLM_QUERY_DEADLINE", "600"))
    """Maximum total time in seconds for a query including all retries (default: 600s / 10 minutes)"""


class LitellmModel:
    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _call_with_thread_timeout(
        self,
        messages: list[dict[str, str]],
        timeout: int,
        **kwargs
    ) -> Any:
        """
        Call litellm.completion with a threading-based timeout as backup.

        This provides a more reliable timeout than litellm's built-in timeout,
        which can be unreliable in some cases (see litellm issues #7996, #14635).
        """
        start_time = time.time()
        effective_timeout = timeout + 30  # Grace period for litellm's internal timeout

        logger.info(
            f"[THREAD-TIMEOUT] Starting LLM call with thread timeout. "
            f"litellm_timeout={timeout}s, thread_timeout={effective_timeout}s, "
            f"model={self.config.model_name}, messages={len(messages)}"
        )

        result_holder: Dict[str, Any] = {"result": None, "error": None, "completed": False}

        def run_completion():
            thread_start = time.time()
            try:
                # Minimal logging inside thread to avoid blocking on log handlers
                result_holder["result"] = litellm.completion(
                    model=self.config.model_name,
                    messages=messages,
                    timeout=timeout,  # Still pass to litellm as first line of defense
                    **(self.config.model_kwargs | kwargs)
                )
                result_holder["completed"] = True
                result_holder["elapsed"] = time.time() - thread_start
            except Exception as e:
                result_holder["error"] = e
                result_holder["completed"] = True
                result_holder["elapsed"] = time.time() - thread_start

        thread = threading.Thread(target=run_completion, daemon=True)
        logger.info(f"[THREAD-TIMEOUT] Starting worker thread...")
        thread.start()

        logger.info(f"[THREAD-TIMEOUT] Calling thread.join(timeout={effective_timeout}s)...")

        # Use polling instead of blocking join to ensure we can detect hangs
        poll_interval = 5.0  # Check every 5 seconds
        waited = 0.0
        while waited < effective_timeout:
            thread.join(timeout=poll_interval)
            waited += poll_interval
            if not thread.is_alive():
                logger.info(f"[THREAD-TIMEOUT] Thread finished after {waited:.1f}s of waiting")
                break
            if waited % 30 < poll_interval:  # Log every ~30 seconds
                logger.info(f"[THREAD-TIMEOUT] Still waiting for thread... ({waited:.0f}s elapsed)")

        # Final check
        if thread.is_alive():
            logger.info(f"[THREAD-TIMEOUT] Polling complete, thread still alive after {waited:.1f}s")

        elapsed = time.time() - start_time
        thread_elapsed = result_holder.get("elapsed", 0)
        logger.info(
            f"[THREAD-TIMEOUT] thread.join() returned after {elapsed:.2f}s. "
            f"thread.is_alive()={thread.is_alive()}, completed={result_holder['completed']}, "
            f"thread_internal_elapsed={thread_elapsed:.2f}s"
        )

        # Log response content OUTSIDE the thread (on main thread)
        if result_holder["completed"] and result_holder["result"] is not None:
            resp = result_holder["result"]
            content = ""
            if resp and hasattr(resp, "choices") and resp.choices:
                content = resp.choices[0].message.content or ""
            logger.info(f"[THREAD-TIMEOUT] Response received, length={len(content)} chars")
            if content:
                # Truncate to first 5000 chars if very long
                log_content = content[:5000] + "..." if len(content) > 5000 else content
                logger.info(f"[THREAD-TIMEOUT] === FULL LLM RESPONSE ===\n{log_content}\n=== END RESPONSE ===")

        if thread.is_alive():
            logger.error(
                f"[THREAD-TIMEOUT] TIMEOUT! LLM call still running after {elapsed:.2f}s "
                f"(expected timeout: {effective_timeout}s). Thread will be abandoned. "
                f"This indicates litellm's timeout failed to trigger."
            )
            raise LLMTimeoutError(
                f"LLM call timed out after {elapsed:.2f} seconds (backup timeout). "
                f"Thread abandoned but may still be running in background."
            )

        if result_holder["error"] is not None:
            logger.info(f"[THREAD-TIMEOUT] Raising captured error: {type(result_holder['error']).__name__}")
            raise result_holder["error"]

        logger.info(f"[THREAD-TIMEOUT] LLM call completed successfully in {elapsed:.2f}s, returning result...")
        return result_holder["result"]

    def _log_query_progress(self, stage: str, details: str = "") -> None:
        """Helper to log query progress stages."""
        logger.info(f"[QUERY-PROGRESS] {stage}" + (f": {details}" if details else ""))

    def _check_deadline(self) -> None:
        """
        Check if the overall query deadline has been exceeded.
        Raises LLMDeadlineExceededError if so.
        """
        deadline = getattr(self, "_query_deadline", None)
        if deadline is not None and time.time() > deadline:
            elapsed = time.time() - (deadline - self.config.query_deadline)
            logger.error(
                f"[DEADLINE] Query deadline exceeded! "
                f"Elapsed: {elapsed:.2f}s, Deadline: {self.config.query_deadline}s"
            )
            raise LLMDeadlineExceededError(
                f"LLM query deadline of {self.config.query_deadline}s exceeded "
                f"(total elapsed: {elapsed:.2f}s including retries). "
                f"This prevents the agent from getting stuck in infinite retry loops."
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        # NOTE: Timeout errors (both litellm.Timeout and LLMTimeoutError) are NOT in this list,
        # so they WILL be retried. RateLimitError is also not listed, so rate limits will be retried.
        # LLMDeadlineExceededError IS in the list so it won't be retried (deadline exceeded is final).
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                LLMDeadlineExceededError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        # Check deadline before each attempt (including retries)
        self._check_deadline()

        # Extract timeout from kwargs or use config default
        timeout = kwargs.pop("timeout", self.config.timeout)

        # If we're close to the deadline, reduce the timeout to not exceed it
        deadline = getattr(self, "_query_deadline", None)
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining < timeout:
                old_timeout = timeout
                timeout = max(30, int(remaining - 10))  # Leave 10s buffer, minimum 30s
                logger.info(
                    f"[DEADLINE] Reducing timeout from {old_timeout}s to {timeout}s "
                    f"due to approaching deadline (remaining: {remaining:.1f}s)"
                )

        logger.info(f"Querying LLM (model={self.config.model_name}, timeout={timeout}s, messages={len(messages)})")

        try:
            return self._call_with_thread_timeout(messages, timeout, **kwargs)
        except LLMTimeoutError:
            logger.warning(f"LLM API call timed out after {timeout}s (backup timeout triggered)")
            # Check deadline after timeout - might have exceeded it
            self._check_deadline()
            raise
        except litellm.exceptions.Timeout as e:
            logger.warning(f"LLM API call timed out after {timeout}s: {e}")
            # Check deadline after timeout - might have exceeded it
            self._check_deadline()
            raise
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """
        Query the LLM with an overall deadline that caps total time including retries.
        """
        query_start = time.time()
        deadline = self.config.query_deadline

        logger.info(
            f"[QUERY] Starting query with deadline={deadline}s, "
            f"model={self.config.model_name}, messages={len(messages)}"
        )

        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
            logger.info("[QUERY] Cache control set")

        # Store the deadline in a context that _query can check
        self._query_deadline = query_start + deadline

        try:
            logger.info("[QUERY] Calling _query()...")
            response = self._query([{"role": msg["role"], "content": msg["content"]} for msg in messages], **kwargs)
            logger.info("[QUERY] _query() returned successfully")
        finally:
            # Clear the deadline after query completes
            self._query_deadline = None

        elapsed = time.time() - query_start
        logger.info(f"[QUERY] LLM call completed in {elapsed:.2f}s, now calculating cost...")

        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
            logger.info(f"[QUERY] Cost calculated: ${cost:.6f}")
        except Exception as e:
            cost = 0.0
            logger.info(f"[QUERY] Cost calculation failed (cost=0): {e}")
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)

        logger.info(f"[QUERY] Extracting response content...")
        content = response.choices[0].message.content or ""
        logger.info(f"[QUERY] Response content extracted ({len(content)} chars), building return dict...")

        result = {
            "content": content,
            "extra": {
                "response": response.model_dump(),
            },
        }
        logger.info(f"[QUERY] Query method complete, returning result")
        return result

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}
