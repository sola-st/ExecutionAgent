# execution_agent/state_persistence.py
"""
State persistence for agent recovery.

This module provides functionality to save and restore agent state,
allowing recovery from crashes or interruptions.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_LOG = logging.getLogger("execution_agent.state")


@dataclass
class AgentState:
    """
    Represents the persistent state of an execution agent.

    This captures all information needed to resume an agent run:
    - Execution history (commands run and their summaries)
    - Files written (both local and in-container)
    - Docker state (image tag, container ID if applicable)
    - Current cycle count and configuration
    """
    # Execution history
    commands_and_summary: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    written_files: List[Tuple[str, str, str, str]] = field(default_factory=list)  # (target, location, path, content)

    # Docker state
    docker_tag: str = ""
    container_id: Optional[str] = None

    # Agent configuration
    cycle_count: int = 0
    step_limit: int = 40

    # Project context
    project_path: str = ""
    project_url: str = ""
    workspace_path: str = ""

    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Status flags
    command_stuck: bool = False
    analysis_succeeded: bool = False

    # Stuck commands list
    stuck_commands: List[str] = field(default_factory=list)

    # Previous attempts (for retry logic)
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create state from dictionary."""
        # Handle tuple conversion for lists that should be tuples
        if "commands_and_summary" in data:
            data["commands_and_summary"] = [
                (cmd, summary) for cmd, summary in data["commands_and_summary"]
            ]
        if "written_files" in data:
            data["written_files"] = [
                tuple(item) for item in data["written_files"]
            ]
        return cls(**data)


class StatePersistence:
    """
    Handles saving and loading agent state to/from disk.

    State is saved as JSON in the run directory, allowing recovery
    if the agent crashes or is interrupted.
    """

    STATE_FILENAME = "agent_state.json"
    BACKUP_SUFFIX = ".backup"

    def __init__(self, run_dir: Path):
        """
        Initialize state persistence.

        Args:
            run_dir: Directory where state files will be saved
        """
        self.run_dir = Path(run_dir)
        self.state_file = self.run_dir / self.STATE_FILENAME
        self.backup_file = self.run_dir / f"{self.STATE_FILENAME}{self.BACKUP_SUFFIX}"

    def save_state(self, agent: Any) -> bool:
        """
        Save current agent state to disk.

        Creates a backup of the previous state before overwriting.

        Args:
            agent: The ExecutionAgent instance

        Returns:
            True if save was successful, False otherwise
        """
        try:
            state = self._extract_state_from_agent(agent)
            state.updated_at = time.time()

            # Create backup of existing state
            if self.state_file.exists():
                try:
                    self.state_file.rename(self.backup_file)
                except Exception as e:
                    _LOG.warning(f"Failed to create state backup: {e}")

            # Ensure directory exists
            self.run_dir.mkdir(parents=True, exist_ok=True)

            # Write new state
            with open(self.state_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2, default=str)

            _LOG.debug(f"Agent state saved to {self.state_file}")
            return True

        except Exception as e:
            _LOG.error(f"Failed to save agent state: {e}", exc_info=True)
            return False

    def load_state(self) -> Optional[AgentState]:
        """
        Load agent state from disk.

        Tries the main state file first, then falls back to backup.

        Returns:
            AgentState if loaded successfully, None otherwise
        """
        # Try main state file
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                state = AgentState.from_dict(data)
                _LOG.info(f"Loaded agent state from {self.state_file}")
                return state
            except Exception as e:
                _LOG.warning(f"Failed to load state from {self.state_file}: {e}")

        # Try backup
        if self.backup_file.exists():
            try:
                with open(self.backup_file, "r") as f:
                    data = json.load(f)
                state = AgentState.from_dict(data)
                _LOG.info(f"Loaded agent state from backup {self.backup_file}")
                return state
            except Exception as e:
                _LOG.warning(f"Failed to load state from backup: {e}")

        return None

    def restore_agent_state(self, agent: Any, state: AgentState) -> bool:
        """
        Restore agent state from a loaded AgentState object.

        Args:
            agent: The ExecutionAgent instance to restore into
            state: The AgentState to restore

        Returns:
            True if restoration was successful, False otherwise
        """
        try:
            # Restore execution history
            agent.commands_and_summary = list(state.commands_and_summary)
            agent.written_files = list(state.written_files)

            # Restore cycle count
            agent.cycle_count = state.cycle_count

            # Restore status flags
            agent.command_stuck = state.command_stuck
            agent.analysis_succeeded = state.analysis_succeeded

            # Restore stuck commands
            if hasattr(agent, "stuck_commands"):
                agent.stuck_commands = list(state.stuck_commands)

            # Restore previous attempts
            if hasattr(agent, "previous_attempts"):
                agent.previous_attempts = list(state.previous_attempts)

            # Restore Docker tag (container will need to be reconnected separately)
            agent.docker_tag = state.docker_tag

            _LOG.info(
                f"Restored agent state: {len(state.commands_and_summary)} commands, "
                f"cycle {state.cycle_count}"
            )
            return True

        except Exception as e:
            _LOG.error(f"Failed to restore agent state: {e}", exc_info=True)
            return False

    def _extract_state_from_agent(self, agent: Any) -> AgentState:
        """Extract current state from an agent instance."""
        container_id = None
        container = getattr(agent, "container", None)
        if container is not None:
            container_id = getattr(container, "id", None)

        return AgentState(
            commands_and_summary=list(getattr(agent, "commands_and_summary", [])),
            written_files=list(getattr(agent, "written_files", [])),
            docker_tag=getattr(agent, "docker_tag", ""),
            container_id=container_id,
            cycle_count=getattr(agent, "cycle_count", 0),
            step_limit=getattr(agent, "step_limit", 40),
            project_path=str(getattr(agent, "project_path", "")),
            project_url=str(getattr(agent, "project_url", "")),
            workspace_path=str(getattr(agent, "workspace_path", "")),
            command_stuck=getattr(agent, "command_stuck", False),
            analysis_succeeded=getattr(agent, "analysis_succeeded", False),
            stuck_commands=list(getattr(agent, "stuck_commands", [])),
            previous_attempts=list(getattr(agent, "previous_attempts", [])),
        )

    def has_saved_state(self) -> bool:
        """Check if there is a saved state file."""
        return self.state_file.exists() or self.backup_file.exists()

    def clear_state(self) -> None:
        """Remove saved state files."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            if self.backup_file.exists():
                self.backup_file.unlink()
            _LOG.info("Cleared saved agent state")
        except Exception as e:
            _LOG.warning(f"Failed to clear state files: {e}")


def create_state_persistence(run_dir: str | Path) -> StatePersistence:
    """
    Create a StatePersistence instance for the given run directory.

    Args:
        run_dir: Directory where state files will be saved

    Returns:
        StatePersistence instance
    """
    return StatePersistence(Path(run_dir))


def save_agent_state_periodically(agent: Any, persistence: StatePersistence, interval: int = 5) -> None:
    """
    Save agent state if enough cycles have passed since last save.

    This should be called after each cycle to ensure periodic state saves.

    Args:
        agent: The ExecutionAgent instance
        persistence: StatePersistence instance
        interval: Number of cycles between saves (default 5)
    """
    cycle_count = getattr(agent, "cycle_count", 0)
    last_save_cycle = getattr(agent, "_last_state_save_cycle", 0)

    if cycle_count - last_save_cycle >= interval:
        if persistence.save_state(agent):
            agent._last_state_save_cycle = cycle_count
