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
import logging
import stat
from pathlib import Path
from typing import Any, List, Tuple, Optional

_LOG = logging.getLogger("execution_agent.exit_artifacts")


def _escape_bash_string(s: str) -> str:
    """Escape a string for safe use in bash single quotes."""
    return s.replace("'", "'\\''")


def _extract_dockerfile_content(agent: Any) -> Optional[str]:
    """
    Extract the Dockerfile content from the agent's written_files.

    Returns:
        The Dockerfile content if found, None otherwise.
    """
    written_files = getattr(agent, "written_files", [])

    for target_name, location, actual_path, content in written_files:
        # Check if this is a Dockerfile
        if target_name.lower() == "dockerfile" or target_name.lower().endswith(".dockerfile"):
            return content

    # Also check if stored directly on agent
    if hasattr(agent, "base_dockerfile") and agent.base_dockerfile:
        return agent.base_dockerfile

    return None


def _extract_container_commands(agent: Any) -> List[str]:
    """
    Extract the bash commands that were executed inside the container.

    Only includes linux_terminal commands that were executed after the container
    was created.

    Returns:
        List of command strings.
    """
    commands_and_summary = getattr(agent, "commands_and_summary", [])
    commands = []
    container_started = False

    for cmd_str, result in commands_and_summary:
        # Check if this is when the container started (Dockerfile was written)
        if "write_to_file" in cmd_str.lower() and "dockerfile" in cmd_str.lower():
            container_started = True
            continue

        # Extract linux_terminal commands after container started
        if container_started and cmd_str.startswith("Call to tool linux_terminal with arguments "):
            args_json = cmd_str[len("Call to tool linux_terminal with arguments "):]
            try:
                args = json.loads(args_json)
                command = args.get("command", "")
                if command and command.strip():
                    commands.append(command)
            except json.JSONDecodeError:
                pass

    return commands


def _generate_commands_script(commands: List[str], project_path: str) -> str:
    """
    Generate a bash script containing the commands executed inside the container.

    Args:
        commands: List of commands that were executed
        project_path: The project path for context

    Returns:
        Bash script content
    """
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("#")
    lines.append("# Commands executed inside the container")
    lines.append(f"# Project: {project_path}")
    lines.append("#")
    lines.append("# This script contains the commands that were successfully executed")
    lines.append("# inside the Docker container during the agent run.")
    lines.append("#")
    lines.append("")
    lines.append("set -e  # Exit on error")
    lines.append("")

    for i, cmd in enumerate(commands, 1):
        lines.append(f"# Command {i}")
        lines.append(cmd)
        lines.append("")

    return "\n".join(lines)


def _generate_launch_script(
    dockerfile_path: str,
    commands_script_path: str,
    project_path: str,
    docker_tag: str = "",
) -> str:
    """
    Generate a launch.sh script that builds the Docker image and runs the commands.

    Args:
        dockerfile_path: Path to the Dockerfile (relative to launch.sh)
        commands_script_path: Path to the commands script (relative to launch.sh)
        project_path: The project path for context
        docker_tag: Optional custom docker tag

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
    lines.append("set -e  # Exit on error")
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
    lines.append('docker build -t "$DOCKER_TAG" "$DOCKERFILE_DIR"')
    lines.append("")
    lines.append("BUILD_STATUS=$?")
    lines.append("if [ $BUILD_STATUS -ne 0 ]; then")
    lines.append("  echo 'ERROR: Docker build failed with exit code $BUILD_STATUS'")
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
    commands = _extract_container_commands(agent)
    if commands:
        commands_script = _generate_commands_script(commands, project_path)
        commands_path = artifacts_dir / "commands.sh"
        commands_path.write_text(commands_script, encoding="utf-8")
        commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        log.info(f"Saved commands script to: {commands_path} ({len(commands)} commands)")
    else:
        # Create an empty commands script with a placeholder
        commands_script = _generate_commands_script(["# No commands were executed after container creation"], project_path)
        commands_path = artifacts_dir / "commands.sh"
        commands_path.write_text(commands_script, encoding="utf-8")
        commands_path.chmod(commands_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        log.info(f"Saved empty commands script to: {commands_path}")

    # 3. Generate and save launch script
    docker_tag = getattr(agent, "docker_tag", "")
    launch_script = _generate_launch_script(
        dockerfile_path="Dockerfile",
        commands_script_path="commands.sh",
        project_path=project_path,
        docker_tag=docker_tag,
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
