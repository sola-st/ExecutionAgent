# execution_agent/env.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class ExecutionEnvironment:
    """
    Execution substrate for the agent.

    Contract:
      execute(cmd) -> {"output": str, "returncode": int, ...}

    Notes:
      - Local mode uses the injected shell_interact_fn(cwd-aware).
      - If container is set, this env can still execute, but the primary
        interface remains the tools (linux_terminal uses advanced helpers).
    """

    def __init__(
        self,
        *,
        workspace_path: str,
        project_path: str,
        shell_interact_fn: Callable[[str], tuple[str, str]],
    ):
        self.workspace_path = workspace_path
        self.project_path = project_path
        self._shell_interact_fn = shell_interact_fn

        # container is set by tools.write_to_file -> env.set_container(...)
        self.container = None

    def set_container(self, container) -> None:
        self.container = container

    def execute(self, command: str) -> Dict[str, Any]:
        cmd = (command or "").strip()

        # local mode
        if self.container is None:
            try:
                out, cwd = self._shell_interact_fn(cmd)
                return {"output": out or "", "returncode": 0, "cwd": cwd}
            except Exception as e:
                return {"output": f"Local execution error: {type(e).__name__}: {e}", "returncode": 1}

        # container mode (best-effort fallback)
        # Prefer tools' screen-based execution; but if something calls env.execute directly,
        # keep it functional.
        try:
            from .docker_helpers_static import exec_in_screen_and_get_log

            rc, out, logfile, stuck = exec_in_screen_and_get_log(self.container, cmd)
            extra = {"logfile": logfile, "stuck": bool(stuck)}
            return {"output": out, "returncode": int(rc), **extra}
        except Exception as e:
            return {"output": f"Container execution error: {type(e).__name__}: {e}", "returncode": 1}
