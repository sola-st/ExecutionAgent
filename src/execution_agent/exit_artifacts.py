#!/usr/bin/env python3
"""
Generate exit artifacts for successful agent runs.

When the agent successfully accomplishes its goals (non-forced exit), this module
generates artifacts that allow reproducing the successful setup:
  1. Dockerfile - The successful Dockerfile that was used
  2. commands.sh - The bash commands executed inside the successful container
  3. launch.sh - A script that builds the container and executes the commands inside it
"""

from __future__ import annotations

import json
import os
import logging
import stat
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from execution_agent.shared_utils import is_dockerfile_name, resolve_write_target, pinned_commit_of

_LOG = logging.getLogger("execution_agent.exit_artifacts")


def _escape_bash_string(s: str) -> str:
    """Escape a string for safe use in bash single quotes."""
    return s.replace("'", "'\\''")


def _extract_dockerfile_content(agent: Any) -> Optional[str]:
    """
    Extract the Dockerfile content from the agent.

    Prefers agent.base_dockerfile, which is only set once a Dockerfile has
    successfully built an image and started a container. Falls back to the most
    recently written Dockerfile, since earlier ones may have failed to build.

    Returns:
        The Dockerfile content if found, None otherwise.
    """
    # Prefer the Dockerfile that actually built and started a container
    if getattr(agent, "base_dockerfile", None):
        return agent.base_dockerfile

    written_files = getattr(agent, "written_files", [])

    for target_name, location, actual_path, content in reversed(written_files):
        # Check if this is a Dockerfile
        if is_dockerfile_name(target_name):
            return content

    return None


def _is_dockerfile_write(tool: str, args: Dict[str, Any]) -> bool:
    return tool == "write_to_file" and is_dockerfile_name(resolve_write_target(args))


def _extract_container_steps(agent: Any) -> List[Dict[str, Any]]:
    """
    Everything that changed the container after it started, in execution order:
      {"kind": "command", "command": str, "returncode": int|None}
      {"kind": "write", "path": str, "content": str, "returncode": int|None}

    Commands come from agent.command_history (recorded per cycle with the tool's
    return code). "The container started" means the Dockerfile write that returned 0;
    commands after a Dockerfile write that failed to build ran in pre-container mode
    and are skipped. Files the model wrote INTO the container with write_to_file are
    replayed too: live, a run patched tests/conftest.py and a pytest internal before
    its final test command, and a replay of the commands alone could not reproduce it.
    Their content comes from agent.written_files (command_history keeps only a
    length marker for bulk payloads).

    Falls back to commands_and_summary (no exit codes, no writes) so artifacts can be
    regenerated from agent_state.json files written before command_history existed.
    """
    history = getattr(agent, "command_history", None) or []
    container_writes = [
        (target, dest, content)
        for target, location, dest, content in getattr(agent, "written_files", []) or []
        if location == "container"
    ]
    steps: List[Dict[str, Any]] = []

    if history:
        container_started = False
        for entry in history:
            tool = str(entry.get("tool", ""))
            args = entry.get("args") or {}
            rc = entry.get("returncode")
            rc = rc if isinstance(rc, int) else None
            if _is_dockerfile_write(tool, args):
                if rc == 0:
                    # A successful write means a (new) container: earlier steps ran in
                    # a container that no longer exists.
                    container_started = True
                    steps = []
                # rc != 0: either the build failed (no container yet) or the write was
                # rejected because a container is already running. Neither changes
                # which container the following steps run in.
                continue
            if not container_started:
                continue
            if tool == "linux_terminal":
                # Prefer what actually ran (after the tool's preprocessing) over the
                # model's raw text; skip the stuck-handler keywords, which are not
                # shell commands.
                command = str(entry.get("executed_command") or args.get("command", "") or "")
                stripped = command.strip()
                if not stripped or stripped.upper() in ("WAIT", "TERMINATE") or stripped.startswith("WRITE:"):
                    continue
                steps.append({"kind": "command", "command": command, "returncode": rc})
            elif tool == "write_to_file" and rc == 0:
                wanted = os.path.basename(os.path.normpath(resolve_write_target(args) or ""))
                # Match the next container write with the same file name (writes are
                # recorded in the same order as the cycles that made them).
                for i, (target, dest, content) in enumerate(container_writes):
                    if os.path.basename(os.path.normpath(target)) == wanted:
                        container_writes.pop(i)
                        steps.append({"kind": "write", "path": dest, "content": content, "returncode": rc})
                        break
        return steps

    container_started = False
    for cmd_str, result in getattr(agent, "commands_and_summary", []):
        if "write_to_file" in cmd_str.lower() and "dockerfile" in cmd_str.lower():
            container_started = True
            continue
        if container_started and cmd_str.startswith("Call to tool linux_terminal with arguments "):
            args_json = cmd_str[len("Call to tool linux_terminal with arguments "):]
            try:
                args = json.loads(args_json)
                command = args.get("command", "")
                if command and command.strip():
                    steps.append({"kind": "command", "command": command, "returncode": None})
            except json.JSONDecodeError:
                pass
    return steps


def _extract_container_commands(agent: Any) -> List[Tuple[str, Optional[int]]]:
    """The shell commands from _extract_container_steps, as (command, returncode)."""
    return [(st["command"], st["returncode"]) for st in _extract_container_steps(agent) if st["kind"] == "command"]


def _heredoc_delimiter(content: str, index: int) -> str:
    delim = f"__EA_FILE_{index}__"
    while delim in content:
        delim += "_"
    return delim


def _generate_commands_script(steps: List[Any], project_path: str) -> str:
    """
    Generate a bash script that replays, in order, what happened inside the container.

    The script keeps going after a failing command. During the run the agent's shell
    did the same: a command could fail (e.g. a test run under the wrong JDK) and a
    later command fixed and repeated it, and the failing command often left state the
    later one needs (packages installed before the step that failed). Aborting on the
    first non-zero exit, as the previous `set -e` did, therefore never reached the
    command that actually succeeded. Each command's exit code is printed next to the
    code recorded during the run, and the script exits with the last command's status.
    Files written into the container are replayed as heredocs at their place in the
    sequence.

    Args:
        steps: List of step dicts from _extract_container_steps; plain
            (command, returncode) tuples are accepted too.
        project_path: The project path for context
    """
    entries: List[Dict[str, Any]] = []
    for item in steps:
        if isinstance(item, dict):
            entries.append(item)
        else:
            entries.append({"kind": "command", "command": str(item[0]), "returncode": item[1]})

    n_failed = sum(1 for e in entries if e.get("returncode") not in (None, 0))
    n_writes = sum(1 for e in entries if e["kind"] == "write")
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("#")
    lines.append("# Commands executed inside the container during the agent run, in order,")
    lines.append("# including the files the agent wrote into the container.")
    lines.append(f"# Project: {project_path}")
    lines.append("#")
    lines.append("# This script does NOT stop on errors. Commands that failed during the run are")
    lines.append("# kept because later commands may depend on what they left behind (packages")
    lines.append("# installed before the failing step, a virtualenv, ...). Each command's exit code")
    lines.append("# is printed next to the code recorded during the run; the script's own exit")
    lines.append("# status is that of the LAST command.")
    if entries:
        lines.append(f"# Recorded: {len(entries)} step(s), {n_writes} file write(s), {n_failed} with a non-zero exit during the run.")
    else:
        lines.append("# No commands were executed after the container was created.")
    lines.append("#")
    lines.append("")
    lines.append("set +e")
    lines.append("_rc=0")
    lines.append("")
    for i, e in enumerate(entries, 1):
        rc = e.get("returncode")
        recorded = "unknown" if rc is None else ("124, timed out / stuck" if rc == 124 else str(rc))
        if e["kind"] == "write":
            path = str(e["path"])
            content = str(e.get("content") or "")
            if not content.endswith("\n"):
                content += "\n"
            delim = _heredoc_delimiter(content, i)
            lines.append(f"# Command {i}: write file {path} (exit code during the run: {recorded})")
            lines.append(f"mkdir -p {_shell_quote(os.path.dirname(path) or '.')} && cat > {_shell_quote(path)} <<'{delim}'")
            lines.append(content.rstrip("\n"))
            lines.append(delim)
            lines.append(f"_rc=$?; echo \"[commands.sh] command {i} (write {path}) exited with $_rc (during the run: {recorded})\" >&2")
        else:
            lines.append(f"# Command {i} (exit code during the run: {recorded})")
            lines.append(str(e["command"]))
            lines.append(f"_rc=$?; echo \"[commands.sh] command {i} exited with $_rc (during the run: {recorded})\" >&2")
        lines.append("")
    lines.append("exit $_rc")
    return "\n".join(lines) + "\n"


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _generate_launch_script(
    dockerfile_path: str,
    commands_script_path: str,
    project_path: str,
    docker_tag: str = "",
    project_url: str = "",
    commit: str = "",
) -> str:
    """
    Generate a launch.sh script that builds the Docker image and runs the commands.

    Args:
        dockerfile_path: Path to the Dockerfile (relative to launch.sh)
        commands_script_path: Path to the commands script (relative to launch.sh)
        project_path: The project path for context
        docker_tag: Optional custom docker tag
        project_url: Repository URL, passed as the REPO_URL build arg to mirror the
            agent's own build (generated Dockerfiles often declare `ARG REPO_URL`)

    Returns:
        Launch script content
    """
    # Generate a default tag if not provided
    safe_project = project_path.replace("/", "-").replace("\\", "-").lower()
    tag = docker_tag or f"execution-agent-{safe_project}"

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("#")
    lines.append("# Launch script for successful execution agent run")
    lines.append(f"# Project: {project_path}")
    lines.append("#")
    lines.append("# This script:")
    lines.append("#   1. Builds the Docker image from the Dockerfile")
    lines.append("#   2. Starts a container from the image")
    lines.append("#   3. Executes the commands inside the container")
    lines.append("#   4. Cleans up the container on exit")
    lines.append("#")
    lines.append("# Usage: ./launch.sh [--keep-container]")
    lines.append("#   --keep-container: Don't remove the container after execution")
    lines.append("#")
    lines.append("")
    # No `set -e`: build/exec statuses are checked explicitly below, and commands.sh
    # deliberately exits with its LAST command's status, which may be non-zero.
    lines.append("set +e")
    lines.append("set -u  # Exit on undefined variable")
    lines.append("")
    lines.append("# Parse arguments")
    lines.append("KEEP_CONTAINER=false")
    lines.append('for arg in "$@"; do')
    lines.append('  case $arg in')
    lines.append("    --keep-container)")
    lines.append("      KEEP_CONTAINER=true")
    lines.append("      shift")
    lines.append("      ;;")
    lines.append("  esac")
    lines.append("done")
    lines.append("")
    lines.append("# Get the directory where this script is located")
    lines.append('SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"')
    lines.append("")
    lines.append("# Configuration")
    lines.append(f"DOCKER_TAG='{_escape_bash_string(tag)}'")
    if project_url:
        lines.append(f"REPO_URL='{_escape_bash_string(project_url)}'")
    if commit:
        lines.append(f"COMMIT_SHA='{_escape_bash_string(commit)}'")
    lines.append(f'DOCKERFILE_PATH="$SCRIPT_DIR/{dockerfile_path}"')
    lines.append(f'COMMANDS_SCRIPT="$SCRIPT_DIR/{commands_script_path}"')
    lines.append('CONTAINER_ID=""')
    lines.append("")
    lines.append("# Cleanup function")
    lines.append("cleanup() {")
    lines.append('  if [ -n "$CONTAINER_ID" ] && [ "$KEEP_CONTAINER" = false ]; then')
    lines.append("    echo 'Cleaning up container...'")
    lines.append("    docker stop $CONTAINER_ID >/dev/null 2>&1 || true")
    lines.append("    docker rm $CONTAINER_ID >/dev/null 2>&1 || true")
    lines.append("    echo 'Container removed.'")
    lines.append("  fi")
    lines.append("}")
    lines.append("")
    lines.append("# Register cleanup on exit")
    lines.append("trap cleanup EXIT")
    lines.append("")
    lines.append("echo '========================================='")
    lines.append(f"echo 'Building and running: {project_path}'")
    lines.append("echo '========================================='")
    lines.append("")
    lines.append("# Step 1: Build the Docker image")
    lines.append("echo ''")
    lines.append("echo '[Step 1/3] Building Docker image...'")
    lines.append("echo ''")
    lines.append("")
    lines.append('DOCKERFILE_DIR="$(dirname "$DOCKERFILE_PATH")"')
    # Build args mirror the agent's own build. Harmless if the Dockerfile does not
    # declare them: docker only warns about unused build args.
    build_args = ""
    if project_url:
        build_args += ' --build-arg REPO_URL="$REPO_URL"'
    if commit:
        build_args += ' --build-arg COMMIT_SHA="$COMMIT_SHA"'
    lines.append(f'docker build{build_args} -t "$DOCKER_TAG" "$DOCKERFILE_DIR"')
    lines.append("")
    lines.append("BUILD_STATUS=$?")
    lines.append("if [ $BUILD_STATUS -ne 0 ]; then")
    lines.append('  echo "ERROR: Docker build failed with exit code $BUILD_STATUS"')
    lines.append("  exit $BUILD_STATUS")
    lines.append("fi")
    lines.append("echo 'Docker image built successfully.'")
    lines.append("")
    lines.append("# Step 2: Start the container")
    lines.append("echo ''")
    lines.append("echo '[Step 2/3] Starting container...'")
    lines.append("echo ''")
    lines.append("")
    lines.append('CONTAINER_ID=$(docker run -d -t "$DOCKER_TAG" tail -f /dev/null)')
    lines.append("START_STATUS=$?")
    lines.append("if [ $START_STATUS -ne 0 ]; then")
    lines.append("  echo 'ERROR: Failed to start container'")
    lines.append("  exit $START_STATUS")
    lines.append("fi")
    lines.append("echo \"Container started with ID: $CONTAINER_ID\"")
    lines.append("")
    lines.append("# Step 3: Execute commands inside the container")
    lines.append("echo ''")
    lines.append("echo '[Step 3/3] Executing commands inside container...'")
    lines.append("echo ''")
    lines.append("")
    lines.append("# Copy the commands script into the container")
    lines.append('docker cp "$COMMANDS_SCRIPT" "$CONTAINER_ID:/tmp/commands.sh"')
    lines.append('docker exec "$CONTAINER_ID" chmod +x /tmp/commands.sh')
    lines.append("")
    lines.append("# Execute the commands script")
    lines.append('docker exec "$CONTAINER_ID" bash -l /tmp/commands.sh')
    lines.append("EXEC_STATUS=$?")
    lines.append("")
    lines.append("echo ''")
    lines.append("echo '========================================='")
    lines.append("if [ $EXEC_STATUS -eq 0 ]; then")
    lines.append("  echo 'Execution completed successfully!'")
    lines.append("else")
    lines.append("  echo \"Execution completed with exit code: $EXEC_STATUS\"")
    lines.append("fi")
    lines.append("echo '========================================='")
    lines.append("")
    lines.append('if [ "$KEEP_CONTAINER" = true ]; then')
    lines.append("  echo ''")
    lines.append("  echo \"Container is still running: $CONTAINER_ID\"")
    lines.append("  echo 'To access it: docker exec -it $CONTAINER_ID bash'")
    lines.append("  echo 'To stop it: docker stop $CONTAINER_ID && docker rm $CONTAINER_ID'")
    lines.append("fi")
    lines.append("")
    lines.append("exit $EXEC_STATUS")

    return "\n".join(lines)


def generate_exit_artifacts(
    agent: Any,
    output_dir: Path,
    log: Optional[logging.Logger] = None,
) -> bool:
    """
    Generate exit artifacts for a successful agent run.

    Creates:
      - Dockerfile: The successful Dockerfile
      - commands.sh: The commands executed inside the container
      - launch.sh: A script to build and run everything

    Args:
        agent: The ExecutionAgent instance
        output_dir: Directory where to save the artifacts
        log: Optional logger instance

    Returns:
        True if artifacts were generated successfully, False otherwise
    """
    if log is None:
        log = _LOG

    project_path = getattr(agent, "project_path", "unknown")

    log.info("=" * 60)
    log.info("Generating exit artifacts for successful run")
    log.info("=" * 60)

    # Create output directory
    artifacts_dir = output_dir / "success_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract and save Dockerfile
    dockerfile_content = _extract_dockerfile_content(agent)
    if not dockerfile_content:
        log.warning("No Dockerfile found in agent state - skipping artifact generation")
        return False

    dockerfile_path = artifacts_dir / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content, encoding="utf-8")
    log.info(f"Saved Dockerfile to: {dockerfile_path}")

    # 2. Extract and save container commands
    commands = _extract_container_steps(agent)
    # An empty list still yields a valid script (header, `set +e`, `exit 0`).
    commands_script = _generate_commands_script(commands, project_path)
    commands_path = artifacts_dir / "commands.sh"
    commands_path.write_text(commands_script, encoding="utf-8")
    commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log.info(f"Saved commands script to: {commands_path} ({len(commands)} commands)")

    # 3. Generate and save launch script
    docker_tag = getattr(agent, "docker_tag", "")
    launch_script = _generate_launch_script(
        dockerfile_path="Dockerfile",
        commands_script_path="commands.sh",
        project_path=project_path,
        docker_tag=docker_tag,
        project_url=str(getattr(agent, "project_url", "") or ""),
        commit=pinned_commit_of(agent),
    )
    launch_path = artifacts_dir / "launch.sh"
    launch_path.write_text(launch_script, encoding="utf-8")
    launch_path.chmod(launch_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log.info(f"Saved launch script to: {launch_path}")

    # 4. Save a summary/manifest
    manifest = {
        "project_path": project_path,
        "project_url": getattr(agent, "project_url", ""),
        "docker_tag": docker_tag,
        "commit": pinned_commit_of(agent),
        "num_commands": len(commands),
        "files": [
            "Dockerfile",
            "commands.sh",
            "launch.sh",
        ],
        "usage": "Run './launch.sh' to build the container and execute the commands. Use '--keep-container' to keep the container running after execution.",
    }
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Saved manifest to: {manifest_path}")

    log.info("=" * 60)
    log.info(f"Exit artifacts saved to: {artifacts_dir}")
    log.info("To reproduce the successful run, execute: ./launch.sh")
    log.info("=" * 60)

    return True
