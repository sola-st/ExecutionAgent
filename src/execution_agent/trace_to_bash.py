#!/usr/bin/env python3
"""
Generate a bash script from agent execution trace.

This module takes the agent's command history and generates a standalone bash script
that reproduces the actions taken by the agent, including:
- Writing the Dockerfile
- Building the Docker image
- Starting the container
- Executing commands inside the container
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, List, Tuple


def _escape_bash_string(s: str) -> str:
    """
    Escape a string for use in bash heredoc or double-quoted context.
    Uses single quotes for maximum safety.
    """
    # Replace single quotes with '\'' (end quote, escaped quote, start quote)
    return s.replace("'", "'\\''")


def _generate_file_write(filename: str, content: str, location: str) -> str:
    """
    Generate bash command to write a file using heredoc.

    Args:
        filename: The target file path
        content: The file content
        location: 'local' or 'container'

    Returns:
        Bash script fragment
    """
    lines = []
    lines.append(f"# Writing file: {filename} ({location})")

    if location == "local":
        # Local file write
        lines.append(f"cat > '{_escape_bash_string(filename)}' <<'EOF_FILE'")
        lines.append(content)
        lines.append("EOF_FILE")
    else:
        # Container file write (exec into container)
        lines.append(f"docker exec $CONTAINER_ID bash -c 'cat > {shlex.quote(filename)} <<'\"'\"'EOF_FILE'\"'\"''")
        lines.append(content)
        lines.append("EOF_FILE")
        lines.append("'")

    lines.append("")
    return "\n".join(lines)


def _generate_docker_build(dockerfile_dir: str, tag: str) -> str:
    """Generate bash command to build Docker image."""
    lines = []
    lines.append(f"# Building Docker image: {tag}")
    lines.append(f"echo 'Building Docker image: {tag}'")
    lines.append(f"docker build -t '{_escape_bash_string(tag)}' '{_escape_bash_string(dockerfile_dir)}'")
    lines.append("BUILD_STATUS=$?")
    lines.append("if [ $BUILD_STATUS -ne 0 ]; then")
    lines.append("  echo 'ERROR: Docker build failed with exit code $BUILD_STATUS'")
    lines.append("  exit $BUILD_STATUS")
    lines.append("fi")
    lines.append(f"DOCKER_TAG='{_escape_bash_string(tag)}'")
    lines.append("")
    return "\n".join(lines)


def _generate_docker_start(tag: str) -> str:
    """Generate bash command to start Docker container."""
    lines = []
    lines.append(f"# Starting Docker container from image: {tag}")
    lines.append(f"echo 'Starting Docker container from image: {tag}'")
    lines.append(f"CONTAINER_ID=$(docker run -d -t '{_escape_bash_string(tag)}' tail -f /dev/null)")
    lines.append("START_STATUS=$?")
    lines.append("if [ $START_STATUS -ne 0 ]; then")
    lines.append("  echo 'ERROR: Failed to start container'")
    lines.append("  exit $START_STATUS")
    lines.append("fi")
    lines.append("echo 'Container started with ID: $CONTAINER_ID'")
    lines.append("")
    return "\n".join(lines)


def _generate_terminal_command(command: str, in_container: bool) -> str:
    """
    Generate bash command for terminal execution.

    Args:
        command: The command to execute
        in_container: Whether to execute inside container

    Returns:
        Bash script fragment
    """
    lines = []

    if in_container:
        lines.append(f"# Executing in container: {command}")
        lines.append(f"echo 'Executing: {_escape_bash_string(command)}'")
        # Use bash -lc to ensure login shell environment
        lines.append(f"docker exec $CONTAINER_ID bash -lc {shlex.quote(command)}")
        lines.append("CMD_STATUS=$?")
        lines.append("if [ $CMD_STATUS -ne 0 ]; then")
        lines.append(f"  echo 'WARNING: Command failed with exit code $CMD_STATUS'")
        lines.append("  # Continuing despite error (agent may have handled this)")
        lines.append("fi")
    else:
        lines.append(f"# Executing locally: {command}")
        lines.append(f"echo 'Executing: {_escape_bash_string(command)}'")
        lines.append(command)
        lines.append("CMD_STATUS=$?")
        lines.append("if [ $CMD_STATUS -ne 0 ]; then")
        lines.append(f"  echo 'WARNING: Command failed with exit code $CMD_STATUS'")
        lines.append("fi")

    lines.append("")
    return "\n".join(lines)


def generate_bash_script_from_trace(
    commands_and_summary: List[Tuple[str, Any]],
    written_files: List[Tuple[str, str, str, str]],
    dockerfile_tag: str = "",
    project_path: str = "",
) -> str:
    """
    Generate a standalone bash script from agent execution trace.

    Args:
        commands_and_summary: List of (command_string, result_dict) tuples
        written_files: List of (target_name, location, actual_path, content) tuples
        dockerfile_tag: The Docker image tag if a container was built
        project_path: The project path for context

    Returns:
        Complete bash script as string
    """
    lines = []

    # Header
    lines.append("#!/usr/bin/env bash")
    lines.append("#")
    lines.append("# Auto-generated bash script from execution agent trace")
    lines.append(f"# Project: {project_path}")
    lines.append("#")
    lines.append("# This script reproduces the actions taken by the execution agent")
    lines.append("#")
    lines.append("")
    lines.append("set -e  # Exit on error")
    lines.append("set -u  # Exit on undefined variable")
    lines.append("")
    lines.append("# Configuration")
    if dockerfile_tag:
        lines.append(f"DOCKER_TAG='{_escape_bash_string(dockerfile_tag)}'")
    lines.append("CONTAINER_ID=''")
    lines.append("")
    lines.append("# Cleanup function")
    lines.append("cleanup() {")
    lines.append("  if [ -n \"$CONTAINER_ID\" ]; then")
    lines.append("    echo 'Cleaning up container: $CONTAINER_ID'")
    lines.append("    docker stop $CONTAINER_ID >/dev/null 2>&1 || true")
    lines.append("    docker rm $CONTAINER_ID >/dev/null 2>&1 || true")
    lines.append("  fi")
    lines.append("}")
    lines.append("")
    lines.append("# Register cleanup on exit")
    lines.append("trap cleanup EXIT")
    lines.append("")
    lines.append("echo '========================================='")
    lines.append("echo 'Starting execution agent trace replay'")
    lines.append("echo '========================================='")
    lines.append("")

    # Track state
    container_started = False
    dockerfile_dir = None

    # Process written files first (to create Dockerfile)
    if written_files:
        lines.append("# ============================================")
        lines.append("# File writes")
        lines.append("# ============================================")
        lines.append("")

        for target_name, location, actual_path, content in written_files:
            # Check if this is a Dockerfile
            is_dockerfile = target_name.lower() == "dockerfile" or target_name.lower().endswith(".dockerfile")

            if is_dockerfile and location == "local":
                # Extract directory for build
                dockerfile_dir = str(Path(actual_path).parent)
                lines.append(f"# Creating Dockerfile directory")
                lines.append(f"mkdir -p '{_escape_bash_string(dockerfile_dir)}'")
                lines.append("")

            lines.append(_generate_file_write(actual_path, content, location))

    # If Dockerfile was written, build and start container
    if dockerfile_dir and dockerfile_tag:
        lines.append("# ============================================")
        lines.append("# Docker image build and container start")
        lines.append("# ============================================")
        lines.append("")
        lines.append(_generate_docker_build(dockerfile_dir, dockerfile_tag))
        lines.append(_generate_docker_start(dockerfile_tag))
        container_started = True

    # Process commands
    if commands_and_summary:
        lines.append("# ============================================")
        lines.append("# Command execution")
        lines.append("# ============================================")
        lines.append("")

        for cmd_str, result in commands_and_summary:
            # Parse command from "Call to tool TOOL_NAME with arguments ARGS" format
            if cmd_str.startswith("Call to tool linux_terminal with arguments "):
                args_json = cmd_str[len("Call to tool linux_terminal with arguments "):]
                try:
                    args = json.loads(args_json)
                    command = args.get("command", "")
                    if command:
                        lines.append(_generate_terminal_command(command, container_started))
                except json.JSONDecodeError:
                    # Skip malformed commands
                    pass

    # Footer
    lines.append("# ============================================")
    lines.append("# Execution complete")
    lines.append("# ============================================")
    lines.append("")
    lines.append("echo '========================================='")
    lines.append("echo 'Execution agent trace replay complete'")
    lines.append("echo '========================================='")
    lines.append("")
    lines.append("# Note: Container is still running. Use 'docker exec $CONTAINER_ID bash' to access it.")
    lines.append("# To stop and remove the container, press Ctrl+C or let the script exit naturally.")

    return "\n".join(lines)


def save_bash_script_from_agent(
    agent: Any,
    output_path: str | Path,
) -> None:
    """
    Generate and save bash script from agent state.

    Args:
        agent: ExecutionAgent instance with commands_and_summary and written_files
        output_path: Path where to save the generated script
    """
    commands_and_summary = getattr(agent, "commands_and_summary", [])
    written_files = getattr(agent, "written_files", [])
    dockerfile_tag = getattr(agent, "docker_tag", "")
    project_path = getattr(agent, "project_path", "")

    script = generate_bash_script_from_trace(
        commands_and_summary=commands_and_summary,
        written_files=written_files,
        dockerfile_tag=dockerfile_tag,
        project_path=project_path,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(script, encoding="utf-8")

    # Make script executable
    import stat
    output_path.chmod(output_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
