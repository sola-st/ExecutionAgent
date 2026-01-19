# execution_agent/tools.py
from __future__ import annotations

import logging
import os
import re
import shlex
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

try:
    import docker  # type: ignore
    from docker.models.containers import Container as DockerContainer  # type: ignore
except Exception:  # pragma: no cover
    docker = None
    DockerContainer = Any  # type: ignore

# Import shared utilities
from execution_agent.shared_utils import (
    exec_in_container,
    read_file_tail,
    write_file_to_container,
    read_file_from_container as _read_file_from_container_shared,
    convert_xml_to_yaml,
    convert_xml_file_to_yaml,
    strip_ansi_codes,
    get_docker_client,
    get_metrics_collector,
    timed_tool,
    SCREEN_SESSION,
    RUN_DIR,
    LOG_DIR,
    STUCK_TIMEOUT_SECONDS,
    POLL_INTERVAL_SECONDS,
    MAX_TAIL_BYTES,
    DEFAULT_EXEC_TIMEOUT,
    ANSI_ESCAPE_RE,
)

_LOG = logging.getLogger("execution_agent.tools")


# ---------------------------------------------------------------------
# Tool registry (used by ExecutionAgent)
# ---------------------------------------------------------------------
class ToolRegistry:
    """
    Simple tool registry for ExecutionAgent.

    - commands_schema: {tool_name: [required_arg1, required_arg2, ...]}
    - register(name, fn): binds the callable.

    The callable is expected to accept (agent=...) either as a kwarg or positional
    depending on your agent implementation; this registry simply stores functions.
    """

    def __init__(self, commands_schema: dict[str, list[str]] | None = None) -> None:
        self.commands_schema: dict[str, list[str]] = commands_schema or {}
        self._tools: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def get(self, name: str) -> Callable[..., Any]:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def normalize_and_validate(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Validate that required arguments are present and return a normalized copy.

        Raises ValueError if the tool is unknown or required args are missing.
        """
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")

        required = self.commands_schema.get(name, [])
        missing = [r for r in required if r not in args]
        if missing:
            raise ValueError(f"Tool '{name}' missing required arguments: {missing}")

        return dict(args)

    def call(self, name: str, args: dict[str, Any], agent: Any = None) -> Any:
        """
        Call a registered tool by name with the given arguments.

        Raises KeyError if the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered")

        fn = self._tools[name]
        return fn(**args, agent=agent)

    @property
    def tools(self) -> dict[str, Callable[..., Any]]:
        return dict(self._tools)


# ---------------------------------------------------------------------
# Constants - imported from shared_utils, with local aliases for compatibility
# ---------------------------------------------------------------------
# These are re-exported from shared_utils:
# SCREEN_SESSION, RUN_DIR, LOG_DIR, STUCK_TIMEOUT_SECONDS, POLL_INTERVAL_SECONDS,
# MAX_TAIL_BYTES, DEFAULT_EXEC_TIMEOUT, ANSI_ESCAPE_RE

# Local aliases for legacy compatibility
THRESH = STUCK_TIMEOUT_SECONDS
WAIT = POLL_INTERVAL_SECONDS
_ANSI_RE = ANSI_ESCAPE_RE


def _result(
    ok: bool,
    title: str,
    details: str | None = None,
    why: list[str] | None = None,
    try_next: list[str] | None = None,
) -> str:
    """
    Legacy-style message formatting. Keep as-is for feedback fidelity.
    """
    lines = [("✅ " if ok else "❌ ") + title.strip()]
    if details:
        lines += ["", "Details:", details.strip()]
    if why:
        lines += ["", "Why this might happen:"] + [f"- {w.strip()}" for w in why if w.strip()]
    if try_next:
        lines += ["", "Try this next:"] + [f"• {t.strip()}" for t in try_next if t.strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# XML helpers - now delegating to shared_utils
# ---------------------------------------------------------------------
# Legacy aliases for compatibility
_convert_xml_to_yaml_file = convert_xml_file_to_yaml
_convert_xml_to_yaml_content = convert_xml_to_yaml


# ---------------------------------------------------------------------
# Docker helpers - using shared implementations with local wrappers
# ---------------------------------------------------------------------
def _docker_client():
    """Get Docker client. Wrapper around shared_utils.get_docker_client()."""
    return get_docker_client()


def _get_container(agent) -> Optional[DockerContainer]:
    """Get container from agent, checking multiple possible locations."""
    if getattr(agent, "container", None) is not None:
        return agent.container
    cid = getattr(getattr(agent, "env", None), "container_id", None) or getattr(agent, "container_id", None)
    if cid and docker is not None:
        try:
            return get_docker_client().containers.get(cid)
        except Exception as e:
            _LOG.warning(f"Failed to get container by ID {cid}: {e}", exc_info=True)
            return None
    return None


def _exec(container: DockerContainer, cmd: str, tty: bool = False, timeout: int = DEFAULT_EXEC_TIMEOUT) -> tuple[int, str]:
    """
    Run a command inside the container.
    Wrapper around shared_utils.exec_in_container() for compatibility.
    """
    return exec_in_container(container, cmd, tty=tty, timeout=timeout)


def _read_tail(container: DockerContainer, path: str) -> str:
    """Read tail of file from container. Wrapper around shared_utils."""
    return read_file_tail(container, path, max_bytes=MAX_TAIL_BYTES)


def read_file_from_container(container: DockerContainer, file_path: str) -> str:
    """Read file from container. Wrapper around shared_utils."""
    return _read_file_from_container_shared(container, file_path)


def write_string_to_file(container: DockerContainer, file_content: str, file_path: str) -> Optional[str]:
    """
    Write content to a file inside the container.
    Wrapper around shared_utils.write_file_to_container() for compatibility.
    """
    return write_file_to_container(container, file_path, file_content)


def _command_exists(container: DockerContainer, cmd: str) -> bool:
    code, _ = _exec(container, f"command -v {shlex.quote(cmd)} >/dev/null 2>&1")
    return code == 0


def _whoami_root(container: DockerContainer) -> bool:
    code, out = _exec(container, "id -u")
    return code == 0 and out.strip() == "0"


def _has_passwordless_sudo(container: DockerContainer) -> bool:
    if not _command_exists(container, "sudo"):
        return False
    code, _ = _exec(container, "sudo -n true")
    return code == 0


def _escalation_prefix(container: DockerContainer) -> str | None:
    """
    Return prefix for privileged commands:
      - "" if already root
      - "sudo -n " if passwordless sudo exists
      - None if neither possible
    """
    if _whoami_root(container):
        return ""
    if _has_passwordless_sudo(container):
        return "sudo -n "
    return None


def _detect_pkg_manager(container: DockerContainer) -> str | None:
    checks = [
        ("microdnf", "command -v microdnf >/dev/null 2>&1"),
        ("dnf",      "command -v dnf >/dev/null 2>&1"),
        ("yum",      "command -v yum >/dev/null 2>&1"),
        ("apt-get",  "command -v apt-get >/dev/null 2>&1"),
        ("apk",      "command -v apk >/dev/null 2>&1"),
    ]
    for pm, probe in checks:
        code, _ = _exec(container, probe)
        if code == 0:
            return pm
    return None


def _install_cmd(pm: str, pkgs: list[str]) -> list[str]:
    pkgs_joined = " ".join(shlex.quote(p) for p in pkgs)
    if pm in ("apt", "apt-get"):
        return [
            "apt-get update",
            f"DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {pkgs_joined}",
        ]
    if pm in ("dnf", "microdnf"):
        tool = "microdnf" if pm == "microdnf" else "dnf"
        return [f"{tool} -y install {pkgs_joined}"]
    if pm == "yum":
        return [f"yum -y install {pkgs_joined}"]
    if pm == "apk":
        return [f"apk add --no-cache {pkgs_joined}"]
    return []


def _ensure_run_dir(container: DockerContainer) -> None:
    _exec(container, f"mkdir -p {shlex.quote(RUN_DIR)} && chmod 1777 {shlex.quote(RUN_DIR)}")


def _best_effort_timezone_setup(container: DockerContainer) -> None:
    # Best-effort; never hard fail. (Europe/Berlin expectation.)
    prefix = _escalation_prefix(container)
    tz_cmd = "TZ=Europe/Berlin && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone"
    if prefix is None:
        _exec(container, "echo Europe/Berlin > /tmp/timezone.info")
    else:
        _exec(container, f"{prefix}{tz_cmd}")


def _ensure_packages(container: DockerContainer, pkgs: list[str]) -> str | None:
    pm = _detect_pkg_manager(container)
    prefix = _escalation_prefix(container)

    if pm is None:
        return _result(
            False,
            "Cannot install packages (no supported package manager detected).",
            details="Tried: microdnf/dnf/yum/apt-get/apk",
            try_next=["Use a base image with a known package manager, or bake dependencies into the Dockerfile."],
        )

    if prefix is None:
        return _result(
            False,
            "Cannot install packages (need root privileges).",
            details=f"Package manager: {pm}. Current user is not root and passwordless sudo is unavailable.",
            try_next=[
                "Run the container as root, or bake dependencies into the image.",
                "If the image supports sudo, configure passwordless sudo.",
            ],
        )

    cmds = _install_cmd(pm, pkgs)
    if not cmds:
        return _result(
            False,
            f"Package manager '{pm}' detected but installer does not support it.",
            try_next=["Bake dependencies into the image in the Dockerfile."],
        )

    for c in cmds:
        code, out = _exec(container, f"{prefix}{c}")
        if code != 0:
            return _result(
                False,
                "Package installation failed.",
                details=f"Command: {prefix}{c}\nExit code: {code}\nOutput:\n{out}",
                try_next=[
                    "Check network/DNS in the container.",
                    "Verify the packages exist for this distro.",
                    "Bake dependencies into the Dockerfile to avoid runtime installs.",
                ],
            )

    return None


# ---------------------------------------------------------------------
# Stateful execution in GNU screen (legacy semantics)
# ---------------------------------------------------------------------
def exec_in_screen_and_get_log(container: DockerContainer, cmd: str) -> Tuple[int, str, str, bool]:
    """
    Legacy-faithful, stateful execution inside GNU screen session.

    Returns: (rc, cleaned_output, logfile_path, stuck_flag)
    """
    def _stuff_single_quoted(session: str, text: str) -> None:
        safe = text.replace("'", r"'\''")
        _exec(container, f"screen -S {shlex.quote(session)} -X stuff '{safe}\\r'")

    run_id = uuid.uuid4().hex
    BEGIN = f"<<BEGIN:{run_id}>>"
    END = f"<<END:{run_id}>>"
    rc_rx = re.compile(rf"<<RC:{re.escape(run_id)}:(-?\d+)>>")
    delim = f"__PAYLOAD_{run_id}__"

    logfile = f"/tmp/screen_exec_stateful_{run_id}.log"
    script = f"/tmp/screen_src_{run_id}.sh"

    if cmd.strip() in {
        'exec "$SHELL" -l',
        "exec '$SHELL' -l",
        'exec "$SHELL" -l ',
        "exec '$SHELL' -l ",
    }:
        _stuff_single_quoted(SCREEN_SESSION, "exec /bin/bash -l")
        time.sleep(0.3)
        return 0, "The shell has been renewed (exec /bin/bash -l).", logfile, False

    _exec(container, f"cat > {shlex.quote(script)} <<'{delim}'\n{cmd}\n{delim}\nchmod +x {shlex.quote(script)}")
    _exec(container, f": > {shlex.quote(logfile)}")

    _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X logfile flush 1 >/dev/null 2>&1 || true")

    payload = (
        f'printf "%s\\n" "{BEGIN}" >> {logfile}; '
        f'if . {script} >> {logfile} 2>&1; then __rc=0; else __rc=$?; fi; '
        f'printf "%s\\n" "{END}" >> {logfile}; '
        f'printf "<<RC:{run_id}:%d>>\\n" "$__rc" >> {logfile}'
    )
    _stuff_single_quoted(SCREEN_SESSION, payload)

    last_buf = ""
    last_change = time.time()
    rc: Optional[int] = None

    while True:
        buf = _read_tail(container, logfile)
        # Compare content, not just length. This fixes a bug where long-running commands
        # that produce lots of output (e.g., Maven builds) were incorrectly flagged as stuck.
        # When the log file exceeds MAX_TAIL_BYTES, tail -c returns a fixed-size buffer,
        # so the length stays constant even as new content is added. Comparing the actual
        # content (or a hash/sample of it) correctly detects changes.
        if buf != last_buf:
            last_buf = buf
            last_change = time.time()

        m = rc_rx.search(buf.replace("\r", ""))
        if m:
            rc = int(m.group(1))
            break

        if time.time() - last_change >= THRESH:
            clean = _ANSI_RE.sub("", buf)
            return 1, (
                "The command appears stuck/idle (no output change within the timeout).\n\n"
                f"Partial output:\n{clean}\n\n"
                "You can WAIT, TERMINATE, or WRITE:<input> to continue."
            ), logfile, True

        time.sleep(WAIT)

    final = _read_tail(container, logfile)
    final_noansi = _ANSI_RE.sub("", final)

    bpos = final_noansi.rfind(BEGIN)
    epos = final_noansi.rfind(END)

    if bpos != -1 and epos != -1 and epos > bpos:
        region = final_noansi[bpos + len(BEGIN): epos]
    else:
        region_lines = []
        for ln in final_noansi.splitlines():
            if ln.startswith("<<BEGIN:") or ln.startswith("<<END:") or ln.startswith("<<RC:"):
                continue
            region_lines.append(ln)
        region = "\n".join(region_lines)

    region = "\n".join(ln for ln in region.splitlines() if not ln.startswith("<<RC:")).strip()
    _exec(container, f"rm -f {shlex.quote(script)}")

    return (rc if rc is not None else 0), region, logfile, False


# ---------------------------------------------------------------------
# Screen health probe + restore (legacy semantics)
# ---------------------------------------------------------------------
SCREEN_HEALTH_TIMEOUT = 10.0
SCREEN_HEALTH_CMD = "pwd"


def parse_screen_session_id(screen_ls: str) -> str:
    for line in screen_ls.splitlines():
        if f".{SCREEN_SESSION}" in line:
            wanted_line = line
            break
    else:
        raise ValueError(f"{SCREEN_SESSION} was not found in `screen -ls` output.")

    for part in wanted_line.split():
        if f".{SCREEN_SESSION}" in part:
            return part.split(".")[0]

    raise ValueError("Could not extract session token from `screen -ls` output.")


def create_screen_session(container: DockerContainer) -> str:
    """
    Legacy-faithful creation:
      - Ensure bash + screen + psmisc exist (+ coreutils on apt)
      - Start/replace a detached screen session running bash -l
      - Ensure RUN_DIR exists
      - Best-effort timezone setup (Europe/Berlin)
    """
    pm = _detect_pkg_manager(container)
    needed = ["bash", "screen", "psmisc"]
    if pm in ("apt", "apt-get"):
        needed.append("coreutils")

    msg = _ensure_packages(container, needed)
    if msg is not None:
        return msg

    _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X quit >/dev/null 2>&1 || true")

    code, out = _exec(container, f"screen -dmS {shlex.quote(SCREEN_SESSION)} bash -l")
    if code != 0:
        return _result(False, f"Failed to start screen session '{SCREEN_SESSION}'.", details=out)

    code, ls_out = _exec(container, "screen -ls")
    if code != 0:
        return _result(False, "Failed to list screen sessions.", details=ls_out)

    try:
        sess_id = parse_screen_session_id(ls_out)
        ACTIVE_SCREEN["id"] = sess_id
        ACTIVE_SCREEN["prep_end"] = True
    except Exception:
        sess_id = ""

    _ensure_run_dir(container)
    _best_effort_timezone_setup(container)

    details = f"Session id: {sess_id}" if sess_id else None
    return _result(True, f"Screen session '{SCREEN_SESSION}' created.", details=details)


def _reset_screen_state(agent) -> None:
    agent.command_stuck = False
    agent.current_logfile = None
    agent.stuck_commands = []


def _probe_screen_via_log(container: DockerContainer, session_name: str, timeout: float = SCREEN_HEALTH_TIMEOUT) -> bool:
    run_id = uuid.uuid4().hex
    logfile = f"/tmp/screen_health_{run_id}.log"
    script = f"/tmp/screen_health_{run_id}.sh"
    RC_MARK = f"<<RC:{run_id}:"

    create_script = (
        f"cat > {script} <<'EOF'\n{SCREEN_HEALTH_CMD}\nEOF\n"
        f"chmod +x {script}\n"
        f": > {logfile}\n"
    )
    rc, _ = _exec(container, create_script)
    if rc != 0:
        return False

    payload = (
        f'printf "%s\\n" "<<BEGIN:{run_id}>>" >> {logfile}; '
        f'/usr/bin/env bash {script} >> {logfile} 2>&1; __rc=$?; '
        f'printf "%s\\n" "<<END:{run_id}>>" >> {logfile}; '
        f'printf "<<RC:{run_id}:%d>>\\n" "$__rc" >> {logfile}'
    )
    safe_payload = payload.replace("'", r"'\''")
    _exec(container, f"screen -S {session_name} -X stuff '{safe_payload}\\r'")

    deadline = time.time() + timeout
    while time.time() < deadline:
        buf = _read_tail(container, logfile)
        if RC_MARK in buf:
            return True
        time.sleep(0.5)
    return False


def _ensure_container_screen_alive(agent) -> bool:
    container = _get_container(agent)
    if not container:
        return True

    session_name = SCREEN_SESSION
    rc, _ = _exec(container, f"screen -S {session_name} -Q windows")
    if rc != 0:
        msg = create_screen_session(container)
        _reset_screen_state(agent)
        if msg.startswith("❌"):
            return False

    if _probe_screen_via_log(container, session_name):
        return True

    _exec(container, f"screen -S {session_name} -X quit || true")
    msg = create_screen_session(container)
    _reset_screen_state(agent)
    if msg.startswith("❌"):
        return False

    return _probe_screen_via_log(container, session_name)


# ---------------------------------------------------------------------
# Stuck handling: WAIT / TERMINATE / WRITE: (legacy semantics)
# ---------------------------------------------------------------------
def _handle_stuck(command: str, agent) -> Optional[str]:
    if not getattr(agent, "command_stuck", False):
        return None

    container = _get_container(agent)
    if not container:
        agent.command_stuck = False
        agent.current_logfile = None
        return None

    NO_CHANGE_TIMEOUT = 300
    POLL_INTERVAL_SECONDS = 5
    WRITE_GRACE_SECONDS = 3
    RC_ANY_RX = re.compile(r"<<RC:[^:]+:(-?\d+)>>")

    def _read_clean_log() -> str:
        try:
            if not getattr(agent, "current_logfile", None):
                return ""
            raw = read_file_from_container(container, agent.current_logfile)
            return _ANSI_RE.sub("", raw)
        except Exception:
            return ""

    def _has_rc_marker(s: str) -> bool:
        return bool(RC_ANY_RX.search(s))

    def _extract_final_region(clean_log: str) -> str:
        bpos = clean_log.rfind("<<BEGIN:")
        epos = clean_log.rfind("<<END:")
        if bpos != -1 and epos != -1 and epos > bpos:
            arrow = clean_log.find(">>", bpos)
            if arrow != -1 and arrow + 2 <= epos:
                return clean_log[arrow + 2 : epos].strip()
        lines = []
        for ln in clean_log.splitlines():
            if ln.startswith("<<BEGIN:") or ln.startswith("<<END:") or ln.startswith("<<RC:"):
                continue
            lines.append(ln)
        return "\n".join(lines).strip()

    def _progress_aware_wait(after_write: bool = False) -> tuple[bool, str]:
        if after_write:
            time.sleep(WRITE_GRACE_SECONDS)

        remaining = NO_CHANGE_TIMEOUT
        last = _read_clean_log()

        if last and _has_rc_marker(last):
            return True, _extract_final_region(last)

        while remaining > 0:
            time.sleep(POLL_INTERVAL_SECONDS)
            cur = _read_clean_log()

            if cur and _has_rc_marker(cur):
                return True, _extract_final_region(cur)

            if cur != last and cur != "":
                remaining = NO_CHANGE_TIMEOUT
                last = cur
            else:
                remaining -= POLL_INTERVAL_SECONDS

        return False, last or ""

    def _nuke_running_and_reset_session():
        try:
            _exec(container, f"screen -S {SCREEN_SESSION} -p 0 -X stuff $'\\003'")
            time.sleep(0.2)
            _exec(container, f"screen -S {SCREEN_SESSION} -p 0 -X stuff $'\\003'")
            time.sleep(0.2)
            _exec(container, f"screen -S {SCREEN_SESSION} -p 0 -X stuff 'exit\\r'")
            time.sleep(0.2)
            _exec(container, f"screen -S {SCREEN_SESSION} -p 0 -X kill")
        except Exception:
            pass
        try:
            _exec(container, f"screen -S {SCREEN_SESSION} -X quit")
        except Exception:
            pass

        create_screen_session(container)
        agent.command_stuck = False
        agent.current_logfile = None

    if command == "WAIT":
        finished, output = _progress_aware_wait(after_write=False)
        if finished:
            agent.command_stuck = False
            return f"Command finished. Output:\n{output}"
        return "command waited for more time and there  was no change you can WAIT more, TERMINATE, or WRITE input to command."

    if command == "TERMINATE":
        _nuke_running_and_reset_session()
        return "Previous command terminated; fresh screen session is ready."

    if command.startswith("WRITE:"):
        user_input = command.split("WRITE:", 1)[1]
        _exec(container, f"screen -S {SCREEN_SESSION} -X stuff '{user_input}\\n'")
        finished, output = _progress_aware_wait(after_write=True)
        if finished:
            agent.command_stuck = False
            return f"Command finished after input. Output:\n{output}"
        return "command waited for more time and there  was no change you can WAIT more, TERMINATE, or WRITE input to command."

    _nuke_running_and_reset_session()
    return None


# ---------------------------------------------------------------------
# Legacy command preprocess + validation
# ---------------------------------------------------------------------
ALLOWLIST_PRE_DOCKER = {"tree", "ls", "cat", "head", "tail", "more", "less", "grep", "find"}


def _preprocess_command(command: str) -> str:
    """
    Preprocess command to fix common issues and add non-interactive flags.
    """
    command = command.replace(" || exit 0", "")
    if command.startswith("bash "):
        return command[len("bash ") :]
    if command == "bash":
        return ""

    # Add -y flag to add-apt-repository if not present (prevents interactive prompts)
    if "add-apt-repository" in command and " -y" not in command and " --yes" not in command:
        # Insert -y after add-apt-repository
        command = command.replace("add-apt-repository ", "add-apt-repository -y ")

    # Ensure apt/apt-get commands have -y flag and DEBIAN_FRONTEND=noninteractive
    apt_patterns = [
        (r'^(apt(?:-get)?)\s+install\b', 'DEBIAN_FRONTEND=noninteractive'),
        (r'^(apt(?:-get)?)\s+upgrade\b', 'DEBIAN_FRONTEND=noninteractive'),
        (r'^(apt(?:-get)?)\s+dist-upgrade\b', 'DEBIAN_FRONTEND=noninteractive'),
        (r'^(apt(?:-get)?)\s+remove\b', 'DEBIAN_FRONTEND=noninteractive'),
        (r'^(apt(?:-get)?)\s+purge\b', 'DEBIAN_FRONTEND=noninteractive'),
        (r'^(apt(?:-get)?)\s+autoremove\b', 'DEBIAN_FRONTEND=noninteractive'),
    ]

    for pattern, prefix in apt_patterns:
        if re.match(pattern, command) and prefix not in command:
            # Add -y if not present
            if " -y" not in command and " --yes" not in command:
                command = re.sub(r'^(apt(?:-get)?)\s+(\w+)', r'\1 \2 -y', command)
            # Add DEBIAN_FRONTEND prefix
            if not command.startswith(prefix):
                command = f"{prefix} {command}"

    # Handle dpkg-reconfigure with noninteractive frontend
    if "dpkg-reconfigure" in command and "DEBIAN_FRONTEND" not in command:
        command = f"DEBIAN_FRONTEND=noninteractive {command}"

    return command


# Interactive commands that should be blocked (will hang waiting for input)
INTERACTIVE_COMMANDS = {
    "nano", "vim", "vi", "emacs", "pico", "joe",  # Editors
    "less", "more",  # Pagers (when used interactively)
    "top", "htop", "iotop",  # System monitors
    "ssh", "telnet", "ftp", "sftp",  # Network tools
    "mysql", "psql", "mongo", "redis-cli",  # Database CLIs (interactive mode)
    "python", "python3", "ipython", "node", "irb", "ghci",  # REPLs (without args)
    "bash", "sh", "zsh", "fish",  # Shells (without args)
}


def _validate_and_block_interactive(command: str) -> Optional[str]:
    """
    Block commands that are known to hang waiting for interactive input.
    """
    # Block known interactive editors/tools
    if "nano " in command or command == "nano":
        return "Error: interactive commands like nano are not allowed. Use write_to_file tool instead."
    if "vim " in command or command in ("vim", "vi"):
        return "Error: interactive commands like vim/vi are not allowed. Use write_to_file tool instead."

    # Block ls -R (too verbose)
    if command == "ls -R":
        return "Error: ls -R is too verbose and is disallowed."

    # Check for standalone interactive commands (REPL mode)
    try:
        parts = shlex.split(command)
        if parts:
            cmd_name = os.path.basename(parts[0])
            # Only block if it's the command alone (REPL mode) or with minimal args
            if cmd_name in INTERACTIVE_COMMANDS and len(parts) == 1:
                return f"Error: '{cmd_name}' without arguments starts an interactive session which would hang. Provide a script or command to execute."
    except ValueError:
        pass

    return None


def _run_local_pre_container(command: str, agent) -> Tuple[int, str]:
    if command.startswith("bash "):
        return 1, (
            "Running a bash script is disallowed at this stage. "
            "Write a Dockerfile first. Once the Dockerfile is built and the container is running, "
            "you can issue commands one at a time to debug and isolate issues."
        )
    if command.startswith("sudo "):
        return 1, "‘sudo’ is not needed. You already have the required permissions—please omit ‘sudo.’"
    if command.startswith("rm "):
        return 1, (
            "Removing files (using ‘rm’) is not permitted right now. "
            "First, create a Dockerfile to manage file modifications inside the container."
        )
    if "SETUP_AND_INSTALL.sh" in command:
        return 1, (
            "Executing setup/install scripts is not allowed at this point. "
            "Please create a Dockerfile first. When you build that Dockerfile, "
            "it will run any installation steps in a controlled way."
        )

    if re.search(r"[|&;`$><]", command):
        return 1, (
            "Piping, redirection, or chaining multiple commands is not allowed. "
            "Submit one simple command at a time (e.g., ‘ls’, ‘cat file.txt’, ‘grep pattern file’)."
            f"Allowed commands at this point are: {', '.join(sorted(ALLOWLIST_PRE_DOCKER))}. "
            "You would have access to more commands once you have written a Dockerfile which would automatically "
            "instantiate a docker container in which you can run more commands."
        )

    try:
        parts = shlex.split(command)
    except ValueError:
        return 1, "Invalid shell syntax—please check your quotes and try again."

    if not parts:
        return 1, "No command provided. Please enter a valid command."

    cmd = parts[0]
    if cmd not in ALLOWLIST_PRE_DOCKER:
        return 1, (
            f"‘{cmd}’ is not permitted. "
            f"Allowed commands at this point are: {', '.join(sorted(ALLOWLIST_PRE_DOCKER))}. "
            "You would have access to more commands once you have written a Dockerfile which would automatically "
            "instantiate a docker container in which you can run more commands."
        )

    if cmd == "find" and ("-exec" in parts or "-ok" in parts):
        return 1, "Using ‘-exec’ or ‘-ok’ with ‘find’ is disallowed. Stick to simple file searches."

    try:
        res = agent.env.execute(command)
        if isinstance(res, dict):
            return int(res.get("returncode", 0) or 0), str(res.get("output", "") or "")
        return 0, str(res)
    except Exception as e:
        return 1, f"Error: {e}"


# ---------------------------------------------------------------------
# Env/pkg snapshot diffing (legacy behavior)
# ---------------------------------------------------------------------
def _diff_dicts(old: dict[str, str], new: dict[str, str]):
    added = {k: new[k] for k in new.keys() - old.keys()}
    removed = {k: old[k] for k in old.keys() - new.keys()}
    changed = {k: (old[k], new[k]) for k in new.keys() & old.keys() if old[k] != new[k]}
    return added, removed, changed


def _safe_env_snapshot(container: DockerContainer) -> Optional[dict[str, str]]:
    rc, out, _, stuck = exec_in_screen_and_get_log(container, "env -0")
    if stuck or rc != 0:
        rc2, out2, _, stuck2 = exec_in_screen_and_get_log(container, "env")
        if stuck2 or rc2 != 0:
            return None
        lines = out2.splitlines()
    else:
        lines = out.split("\x00")
    snap: dict[str, str] = {}
    for line in lines:
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        snap[k] = v
    return snap


def _pkg_snapshot(container: DockerContainer) -> Optional[dict[str, str]]:
    probes = [
        r"command -v dpkg-query >/dev/null 2>&1 && dpkg-query -W -f='${Package}\t${Version}\n'",
        r"command -v rpm >/dev/null 2>&1 && rpm -qa --qf '%{NAME}\t%{VERSION}-%{RELEASE}\n'",
        r"command -v pacman >/dev/null 2>&1 && pacman -Q",
        r"command -v apk >/dev/null 2>&1 && apk info -v",
    ]
    for p in probes:
        rc, out, _, stuck = exec_in_screen_and_get_log(container, p)
        if stuck or rc != 0 or not out.strip():
            continue
        snap: dict[str, str] = {}
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if "\t" in ln:
                name, ver = ln.split("\t", 1)
            elif " " in ln:
                name, ver = ln.split(" ", 1)
            elif "-" in ln:
                name, ver = ln.rsplit("-", 1)
            else:
                name, ver = ln, ""
            snap[name.strip()] = ver.strip()
        if snap:
            return snap
    return None


def _sanitize_cwd(raw: str) -> str:
    text = raw.replace("\n", " ").replace("#", " ").strip()
    last = text.splitlines()[-1] if text.splitlines() else text
    last = re.sub(r"^.*[#\$]\s*", "", last)
    return last


# ---------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------
@timed_tool("linux_terminal")
def linux_terminal(command: str, agent, cwd: str = "", timeout: int | None = None) -> dict[str, Any]:
    """
    Execute a command in the Linux terminal.

    Pre-container: Restricted to safe read-only commands.
    In-container: Full shell access with screen-based stateful execution.

    Args:
        command: The command to execute
        agent: Agent instance
        cwd: Working directory (unused, kept for compatibility)
        timeout: Command timeout (unused, kept for compatibility)

    Returns:
        dict with 'output' and 'returncode' keys
    """
    cmd = _preprocess_command((command or "").strip())

    if err := _validate_and_block_interactive(cmd):
        return {"output": err, "returncode": 1}

    container = _get_container(agent)
    if container and getattr(agent, "command_stuck", False):
        stuck_result = _handle_stuck(cmd, agent)
        if stuck_result is not None:
            return {"output": stuck_result, "returncode": 0}

    if container and not _ensure_container_screen_alive(agent):
        return {
            "output": (
                "Error: could not restore a healthy screen session inside the container. "
                "Please try your command again."
            ),
            "returncode": 1,
        }

    if not container:
        rc, out = _run_local_pre_container(cmd, agent)
        return {"output": f"Output in terminal after executing the command:\n{out}", "returncode": rc}

    # In-container safety checks
    if cmd in getattr(agent, "stuck_commands", []):
        return {
            "output": "Error: This command was previously attempted and got stuck for over 5 minutes. "
                     "It has been added to the blocked list. Please try a different approach or command.",
            "returncode": 1,
        }
    # Block 'exit' command - agent should use goals_accomplished tool instead
    if cmd.strip() == "exit" or cmd.strip().startswith("exit "):
        return {
            "output": "Error: The 'exit' command is not allowed. If you have successfully completed the task "
                     "(tests are passing), please use the 'goals_accomplished' tool instead to properly "
                     "signal completion and provide a summary of what was achieved.",
            "returncode": 1,
        }
    if "~" in cmd:
        return {"output": "Error: Tilde (~) expansion is not supported in this environment. Please use absolute paths instead.", "returncode": 1}
    if cmd.startswith("su -") or cmd == "su -":
        return {"output": "```su - ``` cannot be used, you can either try directly without su or sudo, if it fails, try with sudo but definitely not with `su -`", "returncode": 1}
    if "sed -i" in cmd:
        return {"output": "'sed' based commands are not allowed, if you want to interract with files, have the tools, read_file and write_to_file", "returncode": 1}
    if "SETUP_AND_INSTALL.sh" in cmd:
        return {"output": "Running SETUP_AND_INSTALL.sh right now is not recommended. Please execute each step manually in the terminal.", "returncode": 1}
    if cmd.startswith("docker ") or cmd == "docker":
        return {"output": "Docker commands are not allowed inside the container.", "returncode": 1}

    install_detected = bool(re.match(r"^(apt|apt-get)\s+install\b", cmd))

    # Log command execution for debugging
    cmd_preview = cmd[:200] + "..." if len(cmd) > 200 else cmd
    _LOG.info(f"Executing command in container: {cmd_preview}")

    import time as _time
    start_time = _time.time()
    exit_code, output, logfile, stuck = exec_in_screen_and_get_log(container, cmd)
    duration = _time.time() - start_time

    agent.current_logfile = logfile
    agent.command_stuck = bool(stuck)

    if stuck:
        _LOG.warning(f"Command stuck after {duration:.1f}s: {cmd_preview}")
        agent.stuck_commands.append(cmd)
        return {"output": output, "returncode": 124}

    _LOG.info(f"Command completed in {duration:.1f}s with exit code {exit_code}")

    _, pwd_out, _, pwd_stuck = exec_in_screen_and_get_log(container, "pwd")
    cwd_str = "\n" if pwd_stuck else pwd_out
    cwd_str = _sanitize_cwd(cwd_str)

    if getattr(agent, "envs", None) is None:
        agent.envs = _safe_env_snapshot(container) or {}
    if getattr(agent, "pckgs", None) is None:
        agent.pckgs = _pkg_snapshot(container) or {}

    new_env = _safe_env_snapshot(container) or {}
    env_added, env_removed, env_changed = _diff_dicts(agent.envs or {}, new_env)
    agent.envs = new_env

    new_pkgs = _pkg_snapshot(container) or {}
    pkg_added, pkg_removed, pkg_updated = _diff_dicts(agent.pckgs or {}, new_pkgs)
    agent.pckgs = new_pkgs

    mental_bits: list[str] = []

    if install_detected:
        mental_bits.append(
            "\n\nNOTE: It looks like you just installed a new package. If it provides an executable "
            "that should be set as the default, don’t forget to update alternatives (non‐interactively) "
            "and verify the change. For example:\n\n"
            "  1) If you installed OpenJDK 17 (e.g. `apt install openjdk-17-jdk`), set it as default:\n"
            "       update-alternatives --set java /usr/lib/jvm/java-17-openjdk-amd64/bin/java\n"
            "     Then confirm with:\n"
            "       java -version\n\n"
            "  2) If you installed Python 3.9 (e.g. `apt install python3.9`), switch the “python3” link:\n"
            "       update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1\n"
            "       update-alternatives --set python3 /usr/bin/python3.9\n"
            "     Then verify:\n"
            "       python3 --version\n"
        )

    if env_added or env_removed or env_changed:
        lines = []
        for k, v in sorted(env_added.items()):
            lines.append(f"  + {k}={v}")
        for k, v in sorted(env_removed.items()):
            lines.append(f"  - {k} (was {v})")
        for k, (ov, nv) in sorted(env_changed.items()):
            lines.append(f"  ~ {k}: '{ov}' -> '{nv}'")
        mental_bits.append("\nEnvironment changes since last command:\n" + "\n".join(lines))

    if pkg_added or pkg_removed or pkg_updated:
        lines = []
        for k, v in sorted(pkg_added.items()):
            lines.append(f"  + {k} {v}")
        for k, v in sorted(pkg_removed.items()):
            lines.append(f"  - {k} {v}")
        for k, (ov, nv) in sorted(pkg_updated.items()):
            lines.append(f"  ~ {k}: {ov} -> {nv}")
        mental_bits.append("\nPackage changes since last command:\n" + "\n".join(lines))

    mental_bits.append(f"\nThe current working directory after executing the last command is: {cwd_str}")

    agent.command_stuck = False
    return {"output": "Output in terminal after executing the command:\n" + output + "".join(mental_bits), "returncode": exit_code}


def _workspace_root_for_agent(agent) -> Path:
    # Your main.py sets agent.workspace_path; some older code uses agent.workspace_root.
    wr = getattr(agent, "workspace_root", None) or getattr(agent, "workspace_path", None) or "execution_agent_workspace"
    return Path(str(wr)).resolve()


@timed_tool("read_file")
def read_file(file_path: str, agent) -> dict[str, Any]:
    """
    Read a file from the local filesystem or container.

    Args:
        file_path: Path to the file to read
        agent: Agent instance

    Returns:
        dict with 'output' (file contents) and 'returncode' keys
    """
    container = _get_container(agent)
    if not container:
        try:
            workspace = str(_workspace_root_for_agent(agent))
            project_path = str(getattr(agent, "project_path", "") or "")
            full = os.path.join(workspace, project_path, file_path)

            if file_path.lower().endswith("xml"):
                yaml_content = _convert_xml_to_yaml_file(full)
                return {
                    "output": _result(
                        True,
                        f"Read & converted XML → YAML: {file_path}",
                        details="The XML file was converted to YAML for readability:\n" + yaml_content,
                    ),
                    "returncode": 0,
                }

            with open(full, "r", encoding="utf-8") as f:
                return {"output": f.read(), "returncode": 0}
        except Exception as e:
            return {
                "output": _result(
                    False,
                    f"Failed to read file: {file_path}",
                    details=f"{type(e).__name__}: {e}",
                    why=["The path is wrong.", "The file does not exist or cannot be read (permissions)."],
                ),
                "returncode": 1,
            }

    assumed_path = os.path.join("/app", str(getattr(agent, "project_path", "") or ""), file_path)
    try:
        content = read_file_from_container(container, assumed_path)
        header = (
            "The read_file tool assumes you are in directory "
            f"{os.path.join('/app', str(getattr(agent, 'project_path', '') or ''))}\n"
            f"It tried to read: {assumed_path}\n\n"
            "If this is not what you intended, pass an absolute path instead. "
            " The content of the file you read is bellow:\n\n"
        )
        return {"output": header + content, "returncode": 0}
    except Exception as e:
        return {
            "output": _result(
                False,
                f"Failed to read file in container: {file_path}",
                details=f"Attempted: {assumed_path}\n{type(e).__name__}: {e}",
                why=[
                    "The path is relative but your current working directory is different.",
                    "The file does not exist inside the container at that path.",
                ],
            ),
            "returncode": 1,
        }


# ---------------------------------------------------------------------
# Docker build / start tooling (preserve build failure feedback)
# ---------------------------------------------------------------------
from datetime import datetime as _dt


def _docker_build_image(dockerfile_dir: str, tag: str) -> tuple[bool, str]:
    """
    Build an image and return (ok, full_build_log_text).

    Keeps rich legacy logging, including docker daemon error messages, so failures
    contain actionable feedback.
    """
    client = _docker_client()

    def _ts() -> str:
        return _dt.now().strftime("%H:%M:%S")

    log_lines: list[str] = []

    def _log(msg: str) -> None:
        for ln in (msg.rstrip("\n").splitlines() or [""]):
            log_lines.append(f"[{_ts()}] {ln}")

    _log(f"Starting build: context='{dockerfile_dir}', tag='{tag}'")

    try:
        # Build the image - Docker SDK returns (image, logs_generator) tuple
        # The logs are already decoded by default (no need for decode=True parameter)
        image, logs = client.images.build(
            path=dockerfile_dir,
            tag=tag,
            rm=True,
            pull=True,
            nocache=False
        )

        for chunk in logs:
            # Maintain all the useful detail from legacy
            if isinstance(chunk, dict):
                if "stream" in chunk and chunk["stream"]:
                    _log(chunk["stream"])
                if "status" in chunk:
                    prog = chunk.get("progress")
                    _log(f"{chunk['status']} {prog or ''}".rstrip())
                if "errorDetail" in chunk or "error" in chunk:
                    detail = chunk.get("errorDetail", {}).get("message") or chunk.get("error") or ""
                    _log(f"ERROR: {detail}")
                # Also capture aux messages which might contain useful info
                if "aux" in chunk:
                    _log(f"AUX: {chunk['aux']}")
            else:
                _log(str(chunk))

        img_id = getattr(image, "short_id", None) or getattr(image, "id", None) or "unknown"
        _log(f"Build completed successfully. Image: {img_id}")
        return True, "\n".join(log_lines)

    except Exception as e:
        # Try to extract more detailed error information
        error_msg = f"{type(e).__name__}: {e}"

        # Docker SDK exceptions might have additional attributes
        if hasattr(e, 'explanation'):
            _log(f"ERROR EXPLANATION: {e.explanation}")

        # Some exceptions include build_log attribute with complete build output
        if hasattr(e, 'build_log'):
            for entry in e.build_log:
                if isinstance(entry, dict):
                    if "stream" in entry:
                        _log(entry["stream"])
                    if "errorDetail" in entry:
                        detail = entry["errorDetail"].get('message', str(entry['errorDetail']))
                        _log(f"ERROR: {detail}")
                    if "error" in entry:
                        _log(f"ERROR: {entry['error']}")
                    if "status" in entry:
                        prog = entry.get("progress") or ""
                        _log(f"{entry['status']} {prog}".rstrip())
                    if "aux" in entry:
                        _log(f"AUX: {entry['aux']}")
                else:
                    _log(str(entry))

        # Log the exception details
        _log(f"BUILD FAILED: {error_msg}")

        # Also log the exception type and any additional context
        if hasattr(e, '__cause__') and e.__cause__:
            _log(f"CAUSED BY: {type(e.__cause__).__name__}: {e.__cause__}")

        return False, "\n".join(log_lines)


def _docker_start_container(tag: str) -> Optional[DockerContainer]:
    client = _docker_client()
    try:
        container = client.containers.run(tag, command=["tail", "-f", "/dev/null"], detach=True, tty=True)

        msg = create_screen_session(container)
        if msg.startswith("❌"):
            try:
                container.remove(force=True)
            except Exception:
                pass
            return None

        rc, _ = _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -Q windows")
        if rc != 0:
            try:
                container.remove(force=True)
            except Exception:
                pass
            return None

        return container
    except Exception:
        return None


@timed_tool("write_to_file")
def write_to_file(
    *,
    # Primary parameters (preferred)
    file_path: str | None = None,
    content: str | None = None,
    # Legacy aliases for backward compatibility
    filename: str | None = None,
    path: str | None = None,
    text: str | None = None,
    agent=None,
) -> dict[str, Any]:
    """
    Write content to a file, either locally or inside a Docker container.

    Args:
        file_path: Path to the file to write (preferred parameter name)
        content: Content to write to the file (preferred parameter name)
        filename: Legacy alias for file_path
        path: Legacy alias for file_path
        text: Legacy alias for content
        agent: Agent instance (passed automatically)

    The function handles:
      - Local file writes (before container is created)
      - Dockerfile writes with automatic image build and container start
      - Container file writes (after container is running)

    Note: For new code, use `file_path` and `content` parameters.
    The aliases (filename, path, text) are kept for backward compatibility.
    """
    import secrets

    # Normalize parameter names - prefer file_path/content, fall back to aliases
    target = file_path or path or filename
    data = content if content is not None else text

    if not target:
        return {
            "output": _result(
                False,
                "Missing required parameter: file_path",
                details="You must provide a file path using the 'file_path' parameter.",
                try_next=["Provide a valid file path, e.g., file_path='config.txt'"],
            ),
            "returncode": 1,
        }
    if data is None:
        return {
            "output": _result(
                False,
                "Missing required parameter: content",
                details="You must provide file content using the 'content' parameter.",
                try_next=["Provide the content to write, e.g., content='file contents here'"],
            ),
            "returncode": 1,
        }

    normalized = os.path.normpath(target)
    base = os.path.basename(normalized)
    is_dockerfile = base.lower() == "dockerfile" or base.lower().endswith(".dockerfile")

    if is_dockerfile and "COPY " in data:
        return {
            "output": _result(
                False,
                "Prohibited Dockerfile instruction: COPY",
                why=["This tool avoids relying on local filesystem context."],
                try_next=["Replace COPY with a `git clone` or download via curl/wget and rebuild."],
            ),
            "returncode": 1,
        }

    container = _get_container(agent)

    if container and (is_dockerfile or "Dockerfile" in target):
        return {
            "output": _result(
                False,
                "Cannot write a Dockerfile after the container is running. You cannot stop the running container. Try to fix the issue inside the container via terminal. You cannot run docker commands.",
            ),
            "returncode": 1,
        }

    # Local mode
    if not container:
        if os.path.isabs(normalized):
            return {"output": _result(False, f"Absolute paths are not allowed for local writes: {normalized}"), "returncode": 1}

        workspace_root = _workspace_root_for_agent(agent)

        # Keep legacy-ish layout but add robust fallbacks (main.py may not set these fields)
        tool_dir_name = str(getattr(agent, "tool_repo_rel", None) or getattr(agent, "tool_name", None) or "execution_agent")
        tool_dir = Path(tool_dir_name)

        target_name = getattr(agent, "target_name", None)
        if not target_name:
            proj = str(getattr(agent, "project_path", "") or "")
            target_name = Path(proj).name if proj else "target"

        dockerfile_dir = workspace_root / tool_dir / str(target_name) / "Dockerfile_path"
        dockerfile_dir.mkdir(parents=True, exist_ok=True)

        try:
            if is_dockerfile:
                out_file = dockerfile_dir / "Dockerfile"
            else:
                out_file = workspace_root / normalized
                out_file.parent.mkdir(parents=True, exist_ok=True)

            out_file.write_text(data, encoding="utf-8")
            if hasattr(agent, "written_files"):
                agent.written_files.append((target, "local", str(out_file), data))

            if is_dockerfile:
                lines = data.splitlines()
                run_cmds = sum(1 for ln in lines if ln.strip().upper().startswith("RUN "))
                if len(lines) > 100 or run_cmds > 100:
                    return {
                        "output": _result(
                            False,
                            "Dockerfile rejected: too large/complex",
                            why=["Huge Dockerfiles slow iteration and are more likely to fail."],
                        ),
                        "returncode": 1,
                    }

                now = int(time.time())
                rand = secrets.token_hex(4)
                unique = f"{now}-{rand}"

                tool_part = str(getattr(agent, "tool_name", "tool"))
                base_tag = f"{tool_part}{target_name}_image:executionagent-{unique}".lower()
                tag = "".join(ch if (ch.isalnum() or ch in "._-:") else "-" for ch in base_tag)

                # Store the docker tag for trace generation
                agent.docker_tag = tag

                ok, build_log = _docker_build_image(str(dockerfile_dir), tag)
                if not ok:
                    # Return the FULL build log to the LLM for complete context
                    # The LLM can analyze the entire log to understand what went wrong

                    return {
                        "output": _result(
                            False,
                            f"Build failed for image: {tag}",
                            details=f"Complete Docker build log:\n\n{build_log}",
                            why=[
                                "A RUN command failed (missing package, network issue, wrong command).",
                                "Package name might be wrong or not available in the distribution.",
                                "Network connectivity issues during package download.",
                                "Syntax error in the Dockerfile.",
                            ],
                            try_next=[
                                "Read the complete build log above carefully to identify the exact error.",
                                "Look for lines containing 'ERROR', 'failed', 'cannot', or 'E:' to find the root cause.",
                                "If a package is missing, check the correct package name for this distribution.",
                               ],
                        ),
                        "returncode": 1,
                    }

                c = _docker_start_container(tag)
                if not c:
                    return {
                        "output": _result(
                            False,
                            f"Failed to start container for image: {tag}",
                            why=["The image built but the container failed immediately."],
                            try_next=["Add a minimal CMD/ENTRYPOINT to keep it alive (e.g. tail -f /dev/null)."],
                        ),
                        "returncode": 1,
                    }

                agent.container = c
                agent.base_dockerfile = data
                agent.container_script = data

                try:
                    setattr(agent.env, "container_id", c.id)
                except Exception:
                    pass

                try:
                    if getattr(agent, "stage", None) == "docker_setup":
                        if hasattr(agent, "set_stage"):
                            agent.set_stage("tool_setup", "Container successfully created; moving to tool setup.")
                        else:
                            agent.stage = "tool_setup"
                except Exception:
                    pass

                rc, cwd_out = _exec(c, "pwd")
                cwd_inside = _sanitize_cwd(cwd_out) if rc == 0 else "/app"

                extra_docs = ""
                try:
                    if hasattr(agent, "_load_extra_docs"):
                        extra_docs = agent._load_extra_docs() or ""
                except Exception:
                    extra_docs = ""

                expected = ""
                try:
                    if hasattr(agent, "_load_expected_output_example"):
                        expected = agent._load_expected_output_example() or ""
                except Exception:
                    expected = ""

                msg = (
                    "Docker image built and container started. This container will be our environment from now on. "
                    "We cannot change it. We cannot create another container. We cannot log out of it. These are the rules "
                    "and should not be changed, otherwise I would fail the task."
                )
                if extra_docs.strip():
                    msg += "\n\n" + extra_docs
                if expected.strip():
                    msg += "\n\n" + expected

                return {
                    "output": _result(True, msg, details=f"Image tag: {tag}\nWorking directory inside container: {cwd_inside}"),
                    "returncode": 0,
                }

            # Check if this looks like a test results file
            result_msg = _result(True, "File written successfully", details=f"Local path: {out_file}")
            base_name = os.path.basename(str(out_file)).lower()
            test_result_patterns = [
                "test_results", "testresults", "test-results",
                "test_output", "testoutput", "test-output",
                "results.txt", "results.json", "results.xml",
                "junit", "pytest", "test_report", "testreport"
            ]
            if any(pattern in base_name for pattern in test_result_patterns):
                result_msg += (
                    "\n\n" + "=" * 60 +
                    "\nNOTE: It appears you have saved test results to a file."
                    "\nIf the test results satisfy the task requirements (tests are passing), "
                    "you should call the 'goals_accomplished' tool to properly signal completion."
                    "\n" + "=" * 60
                )
            return {"output": result_msg, "returncode": 0}

        except Exception as e:
            return {"output": _result(False, f"Failed to write locally: {target}", details=f"{type(e).__name__}: {e}"), "returncode": 1}

    # Container mode
    try:
        _, pwd_out, _, stuck = exec_in_screen_and_get_log(container, "pwd")
        cwd_inside = _sanitize_cwd(pwd_out) if not stuck else f"/app/{getattr(agent, 'project_path', '')}"

        if os.path.isabs(target):
            dest = target
        else:
            dest = os.path.normpath(os.path.join(cwd_inside, target))

        if not str(dest).startswith("/"):
            dest = os.path.join(f"/app/{getattr(agent, 'project_path', '')}", target)

        err = write_string_to_file(container, data, dest)
        if err is None:
            if hasattr(agent, "written_files"):
                agent.written_files.append((target, "container", dest, data))

            # Check if this looks like a test results file
            result_msg = _result(True, "File written to container", details=f"Path: {dest}")
            base_name = os.path.basename(dest).lower()
            test_result_patterns = [
                "test_results", "testresults", "test-results",
                "test_output", "testoutput", "test-output",
                "results.txt", "results.json", "results.xml",
                "junit", "pytest", "test_report", "testreport"
            ]
            if any(pattern in base_name for pattern in test_result_patterns):
                result_msg += (
                    "\n\n" + "=" * 60 +
                    "\nNOTE: It appears you have saved test results to a file."
                    "\nIf the test results satisfy the task requirements (tests are passing), "
                    "you should call the 'goals_accomplished' tool to properly signal completion."
                    "\n" + "=" * 60
                )

            return {"output": result_msg, "returncode": 0}

        return {
            "output": _result(
                False,
                f"Failed to write inside container: {target}",
                details=f"Attempted: {dest}\nContainer error: {err}",
                why=[
                    "The destination directory does not exist inside the container.",
                    "Permissions prevent writing to that location.",
                    "The path is relative to a different working directory than you expected.",
                ],
                try_next=[f"Create the directory first: `mkdir -p {os.path.dirname(dest)}`."],
            ),
            "returncode": 1,
        }
    except Exception as e:
        return {"output": _result(False, f"Failed to write file: {target}", details=f"{type(e).__name__}: {e}"), "returncode": 1}


@timed_tool("search_docker_image")
def search_docker_image(search_term: str, agent=None) -> dict[str, Any]:
    """
    Search Docker Hub for images matching the given term.

    Args:
        search_term: Search query for Docker Hub
        agent: Agent instance (optional)

    Returns:
        dict with 'output' (search results) and 'returncode' keys
    """
    term = (search_term or "").strip()
    if not term:
        return {"output": _result(False, "search_docker_image requires a non-empty search_term."), "returncode": 1}

    try:
        client = _docker_client()
    except Exception as e:
        return {
            "output": _result(
                False,
                "Docker SDK not available; cannot search Docker Hub.",
                details=f"{type(e).__name__}: {e}",
                try_next=["Install docker python package and ensure the Docker daemon is reachable."],
            ),
            "returncode": 1,
        }

    try:
        results = client.images.search(term)
        if not results:
            return {
                "output": _result(
                    False,
                    f"No Docker images found for '{term}'.",
                    try_next=["Try a shorter or more general term.", "Search for the upstream project name."],
                ),
                "returncode": 1,
            }

        lines: list[str] = []
        for r in results[:20]:
            name = r.get("name") or ""
            desc = (r.get("description") or "").strip()
            stars = r.get("star_count", 0)
            official = r.get("is_official", False)
            automated = r.get("is_automated", False)
            flags = []
            if official:
                flags.append("official")
            if automated:
                flags.append("automated")
            flag_str = f" ({', '.join(flags)})" if flags else ""
            if len(desc) > 160:
                desc = desc[:159].rstrip() + "…"
            lines.append(f"- {name}{flag_str} — ⭐ {stars} — {desc}")

        return {
            "output": _result(True, f"Top Docker Hub matches for '{term}':", details="\n".join(lines)),
            "returncode": 0,
        }
    except Exception as e:
        return {
            "output": _result(
                False,
                f"Failed to search Docker images for '{term}'.",
                details=f"{type(e).__name__}: {e}",
                why=["Network/registry access is blocked.", "Docker daemon is not reachable."],
            ),
            "returncode": 1,
        }


@timed_tool("goals_accomplished")
def goals_accomplished(reason: str, agent) -> dict[str, Any]:
    """
    Signal that all goals have been accomplished.

    This function sets the success flag and raises GoalsAccomplished exception
    to cleanly exit the agent run loop.

    Args:
        reason: Explanation of what was accomplished
        agent: The ExecutionAgent instance

    Raises:
        GoalsAccomplished: Always raised to signal completion
    """
    from execution_agent.exceptions import GoalsAccomplished

    agent.analysis_succeeded = True
    # Raise exception to cleanly exit the run loop
    raise GoalsAccomplished(reason or "goals accomplished")
