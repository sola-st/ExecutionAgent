# execution_agent/docker_helpers_static.py
"""
Docker helper functions for the execution agent.

This module uses shared_utils for common functionality like exec_in_container,
XML conversion, etc. to avoid code duplication.
"""
from __future__ import annotations

import logging
import shlex
import time
import uuid
from typing import Any, Optional, Tuple

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
    read_file_from_container,
    convert_xml_to_yaml,
    strip_ansi_codes,
    get_docker_client,
    SCREEN_SESSION,
    RUN_DIR,
    LOG_DIR,
    STUCK_TIMEOUT_SECONDS as NO_OUTPUT_CHANGE_SECONDS,
    POLL_INTERVAL_SECONDS as WAIT_POLL_SECONDS,
    MAX_TAIL_BYTES,
    ANSI_ESCAPE_RE as _ANSI_RE,
    RC_MARKER_RE as _RC_ANY_RX,
)


_LOG = logging.getLogger("execution_agent.docker")


# ----------------------------
# Docker client helpers - using shared implementations
# ----------------------------

def _docker_client():
    """Get Docker client. Wrapper around shared_utils."""
    return get_docker_client()


def check_image_exists(tag: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        get_docker_client().images.get(tag)
        return True
    except Exception as e:
        _LOG.debug(f"Image {tag} not found: {e}")
        return False


def build_image(dockerfile_dir: str, tag: str) -> str:
    """
    Build docker image and return a human-readable build log.
    """
    client = get_docker_client()
    log_lines: list[str] = []
    log_lines.append(f"Starting build: context='{dockerfile_dir}', tag='{tag}'")

    try:
        image, logs = client.images.build(path=dockerfile_dir, tag=tag, rm=True, pull=True, nocache=False)
        for chunk in logs:
            if isinstance(chunk, dict):
                if chunk.get("stream"):
                    for ln in str(chunk["stream"]).rstrip("\n").splitlines():
                        log_lines.append(ln)
                if chunk.get("status"):
                    prog = chunk.get("progress") or ""
                    log_lines.append(f"{chunk['status']} {prog}".rstrip())
                if chunk.get("errorDetail") or chunk.get("error"):
                    detail = (chunk.get("errorDetail") or {}).get("message") or chunk.get("error") or ""
                    log_lines.append(f"ERROR: {detail}")
                if chunk.get("aux"):
                    log_lines.append(f"AUX: {chunk['aux']}")
            else:
                log_lines.append(str(chunk))

        img_id = getattr(image, "short_id", None) or getattr(image, "id", None) or "unknown"
        log_lines.append(f"Build completed successfully. Image: {img_id}")
        return "\n".join(log_lines)

    except Exception as e:
        _LOG.error(f"Docker build failed: {e}", exc_info=True)
        error_msg = f"{type(e).__name__}: {e}"

        if hasattr(e, 'explanation'):
            log_lines.append(f"ERROR EXPLANATION: {e.explanation}")

        if hasattr(e, 'build_log'):
            for entry in e.build_log:
                if isinstance(entry, dict):
                    if entry.get("stream"):
                        for ln in str(entry["stream"]).rstrip("\n").splitlines():
                            log_lines.append(ln)
                    if entry.get("errorDetail"):
                        detail = entry["errorDetail"].get("message", str(entry["errorDetail"]))
                        log_lines.append(f"ERROR: {detail}")
                    if entry.get("error"):
                        log_lines.append(f"ERROR: {entry['error']}")
                    if entry.get("status"):
                        prog = entry.get("progress") or ""
                        log_lines.append(f"{entry['status']} {prog}".rstrip())
                else:
                    log_lines.append(str(entry))

        log_lines.append(f"BUILD FAILED: {error_msg}")
        return "\n".join(log_lines)


# ----------------------------
# Container exec primitives - using shared implementations
# ----------------------------

def _exec(container: DockerContainer, cmd: str, tty: bool = False) -> tuple[int, str]:
    """Run a command inside container. Wrapper around shared_utils.exec_in_container()."""
    return exec_in_container(container, cmd, tty=tty)


def _read_tail(container: DockerContainer, path: str) -> str:
    """Read tail of file from container. Wrapper around shared_utils."""
    return read_file_tail(container, path, max_bytes=MAX_TAIL_BYTES)


# ----------------------------
# XML helpers - using shared implementations
# ----------------------------
_convert_xml_to_yaml_content = convert_xml_to_yaml


# ----------------------------
# File read/write in container - now using shared implementations
# Note: read_file_from_container is imported from shared_utils
# We keep a local write_string_to_file that delegates to shared_utils

from execution_agent.shared_utils import write_file_to_container


def write_string_to_file(container: DockerContainer, file_content: str, file_path: str) -> Optional[str]:
    """
    Write content to a file inside the container.
    Wrapper around shared_utils.write_file_to_container() for compatibility.
    """
    return write_file_to_container(container, file_path, file_content)


# ----------------------------
# Package/screen setup (legacy-inspired)
# ----------------------------

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


def _escalation_prefix(container: DockerContainer) -> Optional[str]:
    if _whoami_root(container):
        return ""
    if _has_passwordless_sudo(container):
        return "sudo -n "
    return None


def _detect_pkg_manager(container: DockerContainer) -> Optional[str]:
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


def _install_cmds(pm: str, pkgs: list[str]) -> list[str]:
    pkgs_joined = " ".join(shlex.quote(p) for p in pkgs)
    if pm == "apt-get":
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
    # Best-effort Europe/Berlin, never hard-fail
    prefix = _escalation_prefix(container)
    tz_cmd = "TZ=Europe/Berlin && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone"
    if prefix is None:
        _exec(container, "echo Europe/Berlin > /tmp/timezone.info || true")
    else:
        _exec(container, f"{prefix}sh -lc {shlex.quote(tz_cmd)} || true")


def _ensure_packages(container: DockerContainer, pkgs: list[str]) -> Optional[str]:
    pm = _detect_pkg_manager(container)
    prefix = _escalation_prefix(container)

    if pm is None:
        return "Cannot install packages: no supported package manager detected (microdnf/dnf/yum/apt-get/apk)."
    if prefix is None:
        return "Cannot install packages: need root privileges (not root; passwordless sudo unavailable)."

    cmds = _install_cmds(pm, pkgs)
    if not cmds:
        return f"Package manager '{pm}' detected but installer does not support it."

    for c in cmds:
        code, out = _exec(container, f"{prefix}{c}")
        if code != 0:
            return f"Package installation failed.\nCommand: {prefix}{c}\nExit code: {code}\nOutput:\n{out}"

    return None


def create_screen_session(container: DockerContainer) -> tuple[bool, str]:
    """
    Ensure bash+screen+psmisc exist, then start a detached screen session with bash -l.
    """
    needed = ["bash", "screen", "psmisc"]
    pm = _detect_pkg_manager(container)
    if pm == "apt-get":
        needed.append("coreutils")

    msg = _ensure_packages(container, needed)
    if msg:
        return False, msg

    # Replace session if exists
    _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X quit >/dev/null 2>&1 || true")
    code, out = _exec(container, f"screen -dmS {shlex.quote(SCREEN_SESSION)} bash -l")
    if code != 0:
        return False, f"Failed to start screen session '{SCREEN_SESSION}'. Output:\n{out}"

    # Ensure it exists
    code, ls_out = _exec(container, "screen -ls")
    if code != 0 or f".{SCREEN_SESSION}" not in ls_out:
        return False, f"Screen session '{SCREEN_SESSION}' not found after creation. screen -ls:\n{ls_out}"

    _ensure_run_dir(container)
    _best_effort_timezone_setup(container)
    return True, f"Screen session '{SCREEN_SESSION}' created."


def _ensure_screen_alive(container: DockerContainer) -> tuple[bool, str]:
    code, _ = _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -Q windows")
    if code == 0:
        return True, "ok"
    ok, msg = create_screen_session(container)
    return ok, msg


# ----------------------------
# Stateful screen execution (legacy-faithful)
# ----------------------------

def exec_in_screen_and_get_log(container: DockerContainer, cmd: str) -> Tuple[int, str, str, bool]:
    """
    Stateful execution inside a GNU screen session.

    Returns: (rc, cleaned_output, logfile_path, stuck_flag)
    Stuck is defined as "no output growth/change for NO_OUTPUT_CHANGE_SECONDS".
    """
    ok, msg = _ensure_screen_alive(container)
    if not ok:
        return 1, f"Error: could not ensure screen session. {msg}", "", False

    run_id = uuid.uuid4().hex
    BEGIN = f"<<BEGIN:{run_id}>>"
    END = f"<<END:{run_id}>>"
    rc_rx = re.compile(rf"<<RC:{re.escape(run_id)}:(-?\d+)>>")
    delim = f"__PAYLOAD_{run_id}__"

    logfile = f"{LOG_DIR}/screen_exec_stateful_{run_id}.log"
    script = f"{LOG_DIR}/screen_src_{run_id}.sh"

    def _stuff_single_quoted(text: str) -> None:
        safe = text.replace("'", r"'\''")
        _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X stuff '{safe}\\r'")

    # Legacy special-case: renew shell
    if cmd.strip() in {'exec "$SHELL" -l', "exec '$SHELL' -l"}:
        _stuff_single_quoted("exec /bin/bash -l")
        time.sleep(0.3)
        return 0, "The shell has been renewed (exec /bin/bash -l).", logfile, False

    # Write script
    _exec(
        container,
        f"cat > {shlex.quote(script)} <<'{delim}'\n{cmd}\n{delim}\nchmod +x {shlex.quote(script)}"
    )
    _exec(container, f": > {shlex.quote(logfile)}")

    payload = (
        f'printf "%s\\n" "{BEGIN}" >> {logfile}; '
        f'if . {script} >> {logfile} 2>&1; then __rc=0; else __rc=$?; fi; '
        f'printf "%s\\n" "{END}" >> {logfile}; '
        f'printf "<<RC:{run_id}:%d>>\\n" "$__rc" >> {logfile}'
    )
    _stuff_single_quoted(payload)

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

        if time.time() - last_change >= NO_OUTPUT_CHANGE_SECONDS:
            clean = _ANSI_RE.sub("", buf)
            return 124, (
                "The command appears stuck/idle (no output change within the timeout).\n\n"
                f"Partial output:\n{clean}\n\n"
                "You can WAIT, TERMINATE, or WRITE:<input> to continue."
            ), logfile, True

        time.sleep(WAIT_POLL_SECONDS)

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


# ----------------------------
# Container lifecycle
# ----------------------------

def start_container(tag: str) -> Optional[DockerContainer]:
    """
    Start a container from the given image tag and ensure screen session exists.
    """
    client = _docker_client()
    try:
        c = client.containers.run(tag, command=["tail", "-f", "/dev/null"], detach=True, tty=True)
    except Exception as e:
        _LOG.error("Failed to start container: %s", e)
        return None

    ok, msg = create_screen_session(c)
    if not ok:
        try:
            c.remove(force=True)
        except Exception:
            pass
        _LOG.error("Failed to initialize screen session: %s", msg)
        return None

    return c


# ----------------------------
# Stuck handling (WAIT/TERMINATE/WRITE:)
# ----------------------------

def handle_stuck_action(agent: Any, command: str) -> Optional[str]:
    """
    If agent.command_stuck is True, interpret command as one of:
      WAIT | TERMINATE | WRITE:<text>
    Returns a user-facing terminal string if it handled the request.
    Returns None if not in stuck mode or if no action was taken.
    """
    if not getattr(agent, "command_stuck", False):
        return None

    container = getattr(agent, "container", None)
    logfile = getattr(agent, "current_logfile", None)

    if not container or not logfile:
        agent.command_stuck = False
        agent.current_logfile = None
        return None

    NO_CHANGE_TIMEOUT = 300
    POLL_INTERVAL_SECONDS = 5
    WRITE_GRACE_SECONDS = 2

    def _read_clean_log() -> str:
        try:
            raw = read_file_from_container(container, logfile)
            return _ANSI_RE.sub("", raw)
        except Exception:
            return ""

    def _has_rc_marker(s: str) -> bool:
        return bool(_RC_ANY_RX.search(s or ""))

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

        return False, (last or "")

    def _reset_screen_session() -> None:
        # Best-effort terminate running program and recreate screen
        try:
            _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -p 0 -X stuff $'\\003'")
            time.sleep(0.2)
            _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -p 0 -X stuff $'\\003'")
            time.sleep(0.2)
        except Exception:
            pass
        try:
            _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X quit || true")
        except Exception:
            pass

        create_screen_session(container)
        agent.command_stuck = False
        agent.current_logfile = None

    cmd = (command or "").strip()

    if cmd == "WAIT":
        finished, output = _progress_aware_wait(after_write=False)
        if finished:
            agent.command_stuck = False
            return (
                "Output in terminal after executing the command:\n"
                f"Command finished. Output:\n{output}\n\nReturn code: 0\n"
            )
        return (
            "Output in terminal after executing the command:\n"
            "command waited for more time and there was no change; you can WAIT more, TERMINATE, or WRITE input to command.\n\n"
            "Return code: 124\n"
        )

    if cmd == "TERMINATE":
        _reset_screen_session()
        return (
            "Output in terminal after executing the command:\n"
            "Previous command terminated; fresh screen session is ready.\n\nReturn code: 0\n"
        )

    if cmd.startswith("WRITE:"):
        user_input = cmd.split("WRITE:", 1)[1]
        safe = user_input.replace("'", r"'\''")
        _exec(container, f"screen -S {shlex.quote(SCREEN_SESSION)} -X stuff '{safe}\\r'")
        finished, output = _progress_aware_wait(after_write=True)
        if finished:
            agent.command_stuck = False
            return (
                "Output in terminal after executing the command:\n"
                f"Command finished after input. Output:\n{output}\n\nReturn code: 0\n"
            )
        return (
            "Output in terminal after executing the command:\n"
            "command waited for more time and there was no change; you can WAIT more, TERMINATE, or WRITE input to command.\n\n"
            "Return code: 124\n"
        )

    # Unknown action => reset to keep system usable
    _reset_screen_session()
    return (
        "Output in terminal after executing the command:\n"
        "Unknown stuck action. Previous command terminated and screen session reset.\n\nReturn code: 0\n"
    )


# ----------------------------
# Container cleanup (for retry loop)
# ----------------------------

def cleanup_container(container: Optional[DockerContainer], docker_tag: Optional[str] = None) -> None:
    """
    Stop and remove Docker container, and optionally remove the image.
    Handles errors gracefully with warnings.

    Args:
        container: Docker container object to clean up
        docker_tag: Optional image tag to remove after container cleanup
    """
    # Stop and remove container
    if container is not None:
        try:
            container_id = container.id if hasattr(container, 'id') else str(container)
            _LOG.info(f"Stopping container: {container_id}")
            container.stop(timeout=10)
            _LOG.info(f"Removing container: {container_id}")
            container.remove(force=True)
            _LOG.info(f"Container {container_id} cleaned up successfully")
        except Exception as e:
            _LOG.warning(f"Failed to cleanup container: {e}")

    # Remove Docker image if tag provided
    if docker_tag and docker_tag.strip():
        try:
            client = _docker_client()
            _LOG.info(f"Removing Docker image: {docker_tag}")
            client.images.remove(image=docker_tag, force=True)
            _LOG.info(f"Docker image {docker_tag} removed successfully")
        except Exception as e:
            _LOG.warning(f"Failed to remove Docker image '{docker_tag}': {e}")
