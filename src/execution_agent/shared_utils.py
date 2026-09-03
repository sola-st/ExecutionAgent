# execution_agent/shared_utils.py
"""
Shared utilities for the execution agent.

This module contains:
- Docker container execution primitives
- XML/YAML conversion utilities
- Common constants
- Metrics and timing decorators
"""
from __future__ import annotations

import base64
import functools
import logging
import os
import re
import shlex
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    import docker  # type: ignore
    from docker.models.containers import Container as DockerContainer  # type: ignore
except Exception:  # pragma: no cover
    docker = None
    DockerContainer = Any  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


_LOG = logging.getLogger("execution_agent.shared")

# ---------------------------------------------------------------------
# Configurable constants (can be overridden via environment variables)
# ---------------------------------------------------------------------

# Screen session name
SCREEN_SESSION = os.environ.get("EXECUTION_AGENT_SCREEN_SESSION", "exec_agent_screen")

# Directories for temporary files
RUN_DIR = os.environ.get("EXECUTION_AGENT_RUN_DIR", "/tmp/screen_runs")
LOG_DIR = os.environ.get("EXECUTION_AGENT_LOG_DIR", "/tmp")

# Timeout for command execution before marking as "stuck" (seconds)
# Reduced from 600s to 300s (5 minutes) - most legitimate commands complete faster
STUCK_TIMEOUT_SECONDS = int(os.environ.get("EXECUTION_AGENT_STUCK_TIMEOUT", "300"))

# Polling interval when waiting for command output (seconds)
POLL_INTERVAL_SECONDS = float(os.environ.get("EXECUTION_AGENT_POLL_INTERVAL", "0.5"))

# Maximum bytes to read from log files
MAX_TAIL_BYTES = int(os.environ.get("EXECUTION_AGENT_MAX_TAIL_BYTES", "2000000"))

# Default timeout for _exec() calls (seconds)
DEFAULT_EXEC_TIMEOUT = int(os.environ.get("EXECUTION_AGENT_EXEC_TIMEOUT", "60"))

# ANSI escape sequence pattern
ANSI_ESCAPE_RE = re.compile(r"(?:\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_])")

# Return code marker pattern
RC_MARKER_RE = re.compile(r"<<RC:[^:]+:(-?\d+)>>")


# ---------------------------------------------------------------------
# Metrics and Observability
# ---------------------------------------------------------------------

@dataclass
class ToolMetrics:
    """Tracks execution metrics for tools."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_duration_seconds: float = 0.0
    durations: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Return success rate as a percentage."""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100

    @property
    def avg_duration_seconds(self) -> float:
        """Return average duration in seconds."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_seconds / self.total_calls

    def record_call(self, duration: float, success: bool, error: Optional[str] = None) -> None:
        """Record a tool call."""
        self.total_calls += 1
        self.total_duration_seconds += duration
        self.durations.append(duration)
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error:
                self.errors.append({"time": time.time(), "error": error, "duration": duration})

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate_percent": round(self.success_rate, 2),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "avg_duration_seconds": round(self.avg_duration_seconds, 3),
            "recent_errors": self.errors[-10:] if self.errors else [],
        }


class MetricsCollector:
    """
    Singleton metrics collector for all tool executions.
    Thread-safe for concurrent tool calls.
    """
    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._metrics: Dict[str, ToolMetrics] = {}
        self._metrics_lock = threading.Lock()
        self._initialized = True

    def get_metrics(self, tool_name: str) -> ToolMetrics:
        """Get or create metrics for a tool."""
        with self._metrics_lock:
            if tool_name not in self._metrics:
                self._metrics[tool_name] = ToolMetrics()
            return self._metrics[tool_name]

    def record_call(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Record a tool call with metrics."""
        metrics = self.get_metrics(tool_name)
        metrics.record_call(duration, success, error)

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as a dictionary."""
        with self._metrics_lock:
            return {name: m.to_dict() for name, m in self._metrics.items()}

    def reset(self) -> None:
        """Reset all metrics."""
        with self._metrics_lock:
            self._metrics.clear()


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


F = TypeVar('F', bound=Callable[..., Any])


def timed_tool(tool_name: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator to track execution time and success/failure of tools.

    Usage:
        @timed_tool("linux_terminal")
        def linux_terminal(command: str, agent, ...) -> dict:
            ...

        @timed_tool()  # Will use function name
        def my_tool(...):
            ...
    """
    def decorator(func: F) -> F:
        name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            success = True
            error_msg = None

            try:
                result = func(*args, **kwargs)

                # Check if result indicates failure
                if isinstance(result, dict):
                    returncode = result.get("returncode", 0)
                    if returncode != 0:
                        success = False
                        error_msg = result.get("output", "")[:200]

                return result

            except Exception as e:
                success = False
                error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                raise

            finally:
                duration = time.time() - start_time
                _metrics_collector.record_call(name, duration, success, error_msg)
                _LOG.debug(
                    f"Tool '{name}' completed in {duration:.3f}s "
                    f"(success={success})"
                )

        return wrapper  # type: ignore

    return decorator


# ---------------------------------------------------------------------
# Docker Client Helper
# ---------------------------------------------------------------------

def get_docker_client():
    """Get a Docker client instance."""
    if docker is None:
        raise RuntimeError(
            "Docker SDK not available. Install 'docker' (pip) and ensure Docker daemon is reachable."
        )
    return docker.from_env()


# ---------------------------------------------------------------------
# Container Execution Primitives
# ---------------------------------------------------------------------

def exec_in_container(
    container: DockerContainer,
    cmd: str,
    tty: bool = False,
    timeout: int = DEFAULT_EXEC_TIMEOUT
) -> tuple[int, str]:
    """
    Run a command inside the container using explicit shell invocation.

    This is the canonical implementation used throughout the codebase.
    Uses threading to implement timeout handling.

    Args:
        container: Docker container to execute in
        cmd: Command to run
        tty: Whether to allocate a TTY
        timeout: Maximum seconds to wait for command (default from env or 60)

    Returns:
        (exit_code, output) tuple. Returns (124, timeout message) if command times out.
    """
    result_holder: Dict[str, Any] = {"exit_code": None, "output": None, "error": None}

    def run_command() -> None:
        try:
            res = container.exec_run(["/bin/sh", "-lc", cmd], tty=tty)
            output = res.output
            if isinstance(output, (bytes, bytearray)):
                output = output.decode("utf-8", errors="replace")
            else:
                output = str(output)
            result_holder["exit_code"] = int(res.exit_code)
            result_holder["output"] = output
        except Exception as e:
            result_holder["error"] = str(e)
            _LOG.warning(f"Error executing command in container: {e}", exc_info=True)

    thread = threading.Thread(target=run_command, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        cmd_preview = cmd[:100] + "..." if len(cmd) > 100 else cmd
        _LOG.warning(f"Command timed out after {timeout}s: {cmd_preview}")
        return 124, f"Command timed out after {timeout} seconds: {cmd_preview}"

    if result_holder["error"]:
        return 1, f"Error executing command: {result_holder['error']}"

    return result_holder["exit_code"], result_holder["output"]


def read_file_tail(container: DockerContainer, path: str, max_bytes: int = MAX_TAIL_BYTES) -> str:
    """Read the tail of a file from a container."""
    quoted_path = shlex.quote(path)
    code, out = exec_in_container(
        container,
        f"if [ -f {quoted_path} ]; then tail -c {max_bytes} {quoted_path}; fi"
    )
    return out if code == 0 else ""


def write_file_to_container(
    container: DockerContainer,
    file_path: str,
    content: str
) -> Optional[str]:
    """
    Write content to a file inside the container.

    Uses base64 encoding to safely transfer any content without shell escaping issues.
    Creates parent directories if needed.
    Verifies the file was actually written.

    Args:
        container: Docker container
        file_path: Absolute path inside the container
        content: File content to write

    Returns:
        None on success, error message on failure
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(file_path)
    if parent_dir and parent_dir != "/":
        mkdir_code, mkdir_out = exec_in_container(
            container,
            f"mkdir -p {shlex.quote(parent_dir)}"
        )
        if mkdir_code != 0:
            return f"Failed to create parent directory {parent_dir}: {mkdir_out}"

    # Write using base64 encoding for safety
    encoded_content = base64.b64encode(content.encode('utf-8')).decode('ascii')
    payload = f"echo '{encoded_content}' | base64 -d > {shlex.quote(file_path)}"
    code, out = exec_in_container(container, payload)
    if code != 0:
        return f"Write command failed: {out}"

    # Verify the file was written
    verify_code, verify_out = exec_in_container(
        container,
        f"test -f {shlex.quote(file_path)} && wc -c < {shlex.quote(file_path)}"
    )
    if verify_code != 0:
        return f"File verification failed - file may not exist at {file_path}: {verify_out}"

    # Check that file has content if we wrote content
    try:
        written_bytes = int(verify_out.strip())
        if content and written_bytes == 0:
            return "File exists but appears empty (0 bytes) - write may have failed silently"
    except ValueError:
        pass

    return None


def read_file_from_container(container: DockerContainer, file_path: str) -> str:
    """
    Read a file from inside a container.

    Handles XML files by converting to YAML for easier reading.

    Args:
        container: Docker container
        file_path: Path to file inside container

    Returns:
        File contents as string, or error message if read fails
    """
    code, out = exec_in_container(container, f"cat {shlex.quote(file_path)}", tty=True)
    if code != 0:
        return f"Failed to read {file_path} in the container.\nOutput:\n{out}"

    # Convert XML to YAML for readability
    if file_path.lower().endswith(".xml"):
        try:
            return convert_xml_to_yaml(out)
        except Exception as e:
            _LOG.debug(f"Failed to convert XML to YAML: {e}")
            return out

    return out


# ---------------------------------------------------------------------
# XML/YAML Conversion Utilities
# ---------------------------------------------------------------------

def xml_element_to_dict(element: ET.Element) -> Any:
    """
    Convert an XML element to a Python dictionary recursively.

    Args:
        element: XML Element to convert

    Returns:
        Dictionary representation of the element
    """
    if len(element) == 0:
        return element.text
    return {element.tag: {child.tag: xml_element_to_dict(child) for child in element}}


def convert_xml_to_yaml(xml_content: str) -> str:
    """
    Convert XML content to YAML format.

    Args:
        xml_content: XML string to convert

    Returns:
        YAML string representation

    Raises:
        ValueError: If PyYAML is not installed
        xml.etree.ElementTree.ParseError: If XML is malformed
    """
    if yaml is None:
        raise ValueError("PyYAML is not installed; cannot convert XML -> YAML.")
    root = ET.fromstring(xml_content)
    xml_dict = xml_element_to_dict(root)
    return yaml.dump(xml_dict, default_flow_style=False)


def convert_xml_file_to_yaml(xml_file_path: str) -> str:
    """
    Read an XML file and convert it to YAML format.

    Args:
        xml_file_path: Path to XML file

    Returns:
        YAML string representation
    """
    with open(xml_file_path, "r") as f:
        xml_content = f.read()
    return convert_xml_to_yaml(xml_content)


# ---------------------------------------------------------------------
# ANSI and Text Processing Utilities
# ---------------------------------------------------------------------

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return ANSI_ESCAPE_RE.sub("", text)


def has_return_code_marker(text: str) -> bool:
    """Check if text contains a return code marker."""
    return bool(RC_MARKER_RE.search(text or ""))


def extract_return_code(text: str) -> Optional[int]:
    """Extract return code from marker in text, if present."""
    match = RC_MARKER_RE.search(text or "")
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------
# Logging Helpers
# ---------------------------------------------------------------------

def log_exception(
    logger: logging.Logger,
    message: str,
    level: int = logging.ERROR
) -> Callable[[F], F]:
    """
    Decorator to log exceptions with full traceback.

    Usage:
        @log_exception(_LOG, "Failed to process file")
        def process_file(...):
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log(level, f"{message}: {e}", exc_info=True)
                raise
        return wrapper  # type: ignore
    return decorator


# ---------------------------------------------------------------------
# Screen executor support
# ---------------------------------------------------------------------

def screen_staging_script(run_id: str, logfile: str, script: str, cmd: str, *, source: bool) -> tuple[str, str]:
    """
    Build the ONE shell script that stages a command for the screen executor, and the
    path of the per-command wrapper it installs. Returns (wrapper_path, staging_text).

    The staging text (run via docker exec, never typed) does, under `set -e` so any
    failure - a truncated heredoc on a full disk included - surfaces as a non-zero
    status: write the command script, truncate the logfile, write the wrapper.

    The wrapper is what gets typed into screen, as the bare text `. <wrapper>`:
      - GNU screen's `stuff` expands `$NAME` in typed text (verified by capturing the
        keystrokes), so the exit-status capture cannot be typed; it lives in this
        file, which is only sourced.
      - All paths are literals and the wrapper is sourced WITHOUT arguments, so the
        user's command runs with the shell's own (empty) positional parameters, as it
        did with the original inline payload; a `source activate`-style script that
        forwards "$@", or a `set --`/`shift` in the command, cannot disturb the
        markers. It also works in a non-bash shell, which ignores `.` arguments.
      - The user script is sourced inside `if`, so a `set -e` in it cannot kill the
        interactive shell (bash ignores errexit for an `if` condition).
      - The wrapper removes the script and itself when done: no per-command litter.
    `source=True` keeps `cd`/`export` in the shell; `exec` runs a child bash (probe).
    """
    wrapper = f"/tmp/screen_wrap_{run_id}.sh"
    run = f". {script}" if source else f"/usr/bin/env bash {script}"
    body = (
        "#!/bin/bash\n"
        f'printf "%s\\n" "<<BEGIN:{run_id}>>" >> {logfile}\n'
        f"if {run} >> {logfile} 2>&1; then __rc=0; else __rc=$?; fi\n"
        f'printf "%s\\n" "<<END:{run_id}>>" >> {logfile}\n'
        f'printf "<<RC:{run_id}:%d>>\\n" "$__rc" >> {logfile}\n'
        "unset __rc\n"
        f"rm -f {script} {wrapper}\n"
    )
    staging = (
        "set -e\n"
        f"cat > {script} <<'__SRC_{run_id}__'\n{cmd}\n__SRC_{run_id}__\n"
        f"chmod +x {script}\n"
        f": > {logfile}\n"
        f"cat > {wrapper} <<'__WRAP_{run_id}__'\n{body}__WRAP_{run_id}__\n"
    )
    return wrapper, staging


def stuff_escape(text: str) -> str:
    """
    Escape free text for `screen -S <s> -X stuff '<text>'`: backslashes, `$` and `^`
    for screen's own escape/variable/caret processing (`^X` would become Ctrl-X),
    and `'` for the single-quoted shell argument. Screen then types the text literally.
    """
    return (text.replace("\\", "\\\\").replace("$", "\\$").replace("^", "\\^")
                .replace("'", "'\\''"))


# ---------------------------------------------------------------------
# File-name conventions shared by the tools and the exit artifacts
# ---------------------------------------------------------------------

def resolve_write_target(args: Dict[str, Any]) -> str:
    """The path a write_to_file call targets, resolving the tool's aliases in the
    same order write_to_file itself does."""
    if not isinstance(args, dict):
        return ""
    return str(args.get("file_path") or args.get("path") or args.get("filename") or "")


def is_dockerfile_name(path: str) -> bool:
    """True if `path` names a Dockerfile (`Dockerfile` or `*.dockerfile`), matching
    write_to_file's detection (normpath + basename, case-insensitive)."""
    if not path:
        return False
    base = os.path.basename(os.path.normpath(str(path))).lower()
    return base == "dockerfile" or base.endswith(".dockerfile")


# ---------------------------------------------------------------------
# Commit pinning
# ---------------------------------------------------------------------

def pinned_clone_snippet(commit: str) -> str:
    """
    The one Dockerfile fragment for building at a pinned commit, shared by the agent
    prompt, the summary generator and the tool error message so they cannot drift.
    """
    return (
        "ARG REPO_URL\n"
        "ARG PROJECT_DIR=project\n"
        f"ARG COMMIT_SHA={commit}\n"
        'RUN test -n "$REPO_URL" && git clone "$REPO_URL" "/app/$PROJECT_DIR" \\\n'
        '    && git -C "/app/$PROJECT_DIR" checkout --detach "$COMMIT_SHA"\n'
        "WORKDIR /app/${PROJECT_DIR}\n"
    )


_TEMPLATE_SHALLOW_CLONE = 'RUN test -n "$REPO_URL" && git clone --depth 1 "$REPO_URL" "$PROJECT_DIR"'


def render_pinned_template(query: str, commit: str) -> str:
    """
    Rewrite the summary prompt's Dockerfile template for a pinned commit: the shallow
    clone line becomes a full clone plus checkout at the commit, so the template the
    knowledge model is told to reproduce is right by construction instead of being
    contradicted by an appended instruction.
    """
    if not commit or _TEMPLATE_SHALLOW_CLONE not in query:
        return query
    replacement = (
        f"ARG COMMIT_SHA={commit}\n"
        'RUN test -n "$REPO_URL" && git clone "$REPO_URL" "/app/$PROJECT_DIR" \\\n'
        '    && git -C "/app/$PROJECT_DIR" checkout --detach "$COMMIT_SHA"'
    )
    return query.replace(_TEMPLATE_SHALLOW_CLONE, replacement)


def pinned_commit_of(agent: Any) -> str:
    """The commit a run is pinned to ('' if none), via the agent's accessor when it
    has one. The single reader used by the tools and the exit artifacts."""
    fn = getattr(agent, "pinned_commit", None)
    if callable(fn):
        try:
            return str(fn() or "").strip()
        except Exception:
            pass
    return str(getattr(agent, "commit", "") or "").strip()
