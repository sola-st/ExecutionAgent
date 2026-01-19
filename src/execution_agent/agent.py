# execution_agent/agent.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .exceptions import BudgetExhausted, FormatError, GoalsAccomplished
from .repetition import is_repetition
from .state_persistence import StatePersistence, save_agent_state_periodically

if TYPE_CHECKING:
    from .tools import ToolRegistry  # type-only to avoid circular imports

_LOG = logging.getLogger("execution_agent.agent")


def _extract_json_substring(text: str) -> str:
    """
    Extract the JSON object substring from text by finding matching braces.
    Returns the substring from first '{' to its matching '}'.
    """
    start = text.find('{')
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # If we didn't find matching brace, return from start to end
    return text[start:]


def _fix_escaped_newlines(obj: Any) -> Any:
    """
    Recursively fix improperly escaped newlines in JSON object.

    When the model over-escapes and outputs \\n in JSON, json.loads() converts it to
    literal backslash-n string. This function converts those back to actual newlines.
    Also handles other common escape sequences: \\t, \\r, etc.
    """
    if isinstance(obj, dict):
        return {k: _fix_escaped_newlines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fix_escaped_newlines(item) for item in obj]
    elif isinstance(obj, str):
        # Fix common over-escaped sequences
        # Only fix if we see literal backslash-n (not already a newline)
        if '\\n' in obj or '\\t' in obj or '\\r' in obj or '\\\\' in obj:
            # Replace literal escape sequences with actual escape characters
            # Note: We don't need to handle \\\\ first because the over-escaping from
            # the model produces \\n not \\\\n, so the order doesn't matter here
            obj = obj.replace('\\n', '\n')
            obj = obj.replace('\\t', '\t')
            obj = obj.replace('\\r', '\r')
        return obj
    else:
        return obj


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract and parse a JSON object from text.
    Handles:
      - Raw JSON object
      - JSON inside a ```json ... ``` block
      - JSON with leading/trailing text
      - Fixes improperly escaped newlines (\\n -> \n)

    Returns parsed dict or raises FormatError.
    """
    text = (text or "").strip()

    # Prefer fenced block if present
    fence = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence[0].strip()
    else:
        # Extract JSON object by finding matching braces
        candidate = _extract_json_substring(text)

    if not candidate:
        raise FormatError("No JSON object found in model output.")

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            # Post-process to fix improperly escaped newlines in string values
            obj = _fix_escaped_newlines(obj)
            return obj
        raise FormatError("Top-level JSON must be an object (dict).")
    except json.JSONDecodeError as e:
        raise FormatError(f"Invalid JSON: {e}") from e


def _truncate(s: str, n: int = 1200) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _smart_truncate_output(
    s: str,
    head_chars: int = 10000,
    middle_chars: int = 2000,
    tail_chars: int = 10000,
    skip_marker: str = "\n\n... [output truncated: {skipped} characters omitted] ...\n\n",
) -> str:
    """
    Truncate large output while preserving the most useful parts.

    Keeps:
      - First `head_chars` characters (beginning of output - usually shows command and initial results)
      - `middle_chars` characters from the middle (can reveal patterns/progress)
      - Last `tail_chars` characters (end of output - usually shows final results/errors)

    This prevents context window overflow while maintaining diagnostic value.

    Args:
        s: The string to truncate
        head_chars: Number of characters to keep from the beginning
        middle_chars: Number of characters to keep from the middle
        tail_chars: Number of characters to keep from the end
        skip_marker: Template for the skip message (use {skipped} for char count)

    Returns:
        The truncated string with skip markers, or original if under limit.
    """
    s = s or ""
    total_keep = head_chars + middle_chars + tail_chars
    # Add buffer for skip markers (approximately 100 chars each)
    if len(s) <= total_keep + 200:
        return s

    # Calculate positions
    head_end = head_chars
    tail_start = len(s) - tail_chars

    # Middle section: centered between head and tail
    middle_region_start = head_end
    middle_region_end = tail_start
    middle_region_size = middle_region_end - middle_region_start

    if middle_region_size <= middle_chars:
        # Not enough gap for middle truncation, just do head + tail
        skipped = len(s) - head_chars - tail_chars
        return (
            s[:head_chars]
            + skip_marker.format(skipped=skipped)
            + s[-tail_chars:]
        )

    # Calculate middle sample position (centered)
    middle_sample_start = middle_region_start + (middle_region_size - middle_chars) // 2
    middle_sample_end = middle_sample_start + middle_chars

    # Calculate skipped amounts
    skipped_after_head = middle_sample_start - head_end
    skipped_after_middle = tail_start - middle_sample_end

    parts = [s[:head_end]]

    if skipped_after_head > 0:
        parts.append(skip_marker.format(skipped=skipped_after_head))

    parts.append(s[middle_sample_start:middle_sample_end])

    if skipped_after_middle > 0:
        parts.append(skip_marker.format(skipped=skipped_after_middle))

    parts.append(s[tail_start:])

    return "".join(parts)


@dataclass
class _Message:
    role: str
    content: str
    tag: Optional[str] = None


class ExecutionAgent:
    """
    Workflow (unchanged):
      - plan one tool call (strict JSON)
      - execute it
      - summarize output into compact memory
      - repeat until GoalsAccomplished or step limit
    """

    def __init__(
        self,
        model,
        env,
        *,
        tool_registry: "ToolRegistry",
        cycle_instruction: str,
        summarize_cycle: str,
        remove_progress_bars_prompt: str,
        search_workflows_summary_prompt: str,
        step_limit: int = 40,
    ):
        self.model = model
        self.env = env
        self.tool_registry = tool_registry

        # prompts
        self.cycle_instruction = cycle_instruction
        self.summarize_cycle = summarize_cycle
        self.remove_progress_bars_prompt = remove_progress_bars_prompt
        self.search_workflows_summary_prompt = search_workflows_summary_prompt

        # limits
        self.step_limit = int(step_limit)

        # state (carry-overs)
        self.commands_and_summary: List[tuple[str, Dict[str, Any]]] = []
        self.written_files: List[tuple[str, str, str, str]] = []  # (target, location, path, content)
        self.prompt_text: str = ""

        # runtime substrate (set by tools)
        self.container = None

        # injected by main.py after init
        self.workspace_path: str = ""
        self.project_path: str = ""
        self.project_url: str = ""
        self.hyperparams: Dict[str, Any] = {}
        self.repo_context = None
        self.tools_doc_string: str = ""
        self.language_guidelines: str = ""

        # conversation tail (for logging/observability)
        # NOTE: keep as list[dict]-like objects to interop with main.py wrappers.
        self.messages: List[Dict[str, Any]] = []

        # cycle counter (for logging)
        self.cycle_count: int = 0

        # last action, result, and thoughts (for prompt construction)
        self.last_action: Optional[Dict[str, Any]] = None
        self.last_result: Optional[str] = None
        self.last_thoughts: Optional[str] = None

        # last format error and failed response (for retry mechanism)
        self.last_format_error: Optional[str] = None
        self._last_failed_response: Optional[str] = None

        # optional stuck state (used by sophisticated linux_terminal implementations)
        # (tools/helpers may set these dynamically; having defaults avoids attribute errors)
        self.command_stuck: bool = False
        self.current_logfile: Optional[str] = None
        self.stuck_commands: List[str] = []  # Commands that got stuck and should be blocked

        # previous attempts (preserved across retries for learning)
        self.previous_attempts: List[Dict[str, Any]] = []

    # -------------------------
    # message helpers
    # -------------------------
    def add_message(self, role: str, content: str, *, tag: Optional[str] = None) -> None:
        self.messages.append({"role": role, "content": content, "tag": tag})

    # -------------------------
    # command history helpers
    # -------------------------
    def _last_commands(self) -> List[Dict[str, Any]]:
        cmds: List[Dict[str, Any]] = []
        for call_str, _ in self.commands_and_summary[-20:]:
            m = re.match(r"Call to tool ([^ ]+) with arguments (.*)$", call_str)
            if not m:
                continue
            name = m.group(1)
            raw_args = m.group(2)
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    args = {}
            except Exception:
                args = {}
            cmds.append({"name": name, "args": args})
        return cmds

    # -------------------------
    # prompt construction
    # -------------------------
    def _build_instance_prompt(self, task: str) -> str:
        ctx = self.repo_context

        parts: list[str] = []
        if ctx is not None:
            parts.append(f"Project path: {ctx.project_path}")
            parts.append(f"Project URL: {ctx.project_url}")
            parts.append(f"Primary language: {ctx.language}")
            parts.append("")

            # Add language-specific guidelines if available
            if self.language_guidelines:
                parts.append(f"Language-specific guidelines for {ctx.language}:")
                parts.append(self.language_guidelines)
                parts.append("")

            if getattr(ctx, "problems_memory", None):
                parts.append("Problems memory (from prior runs):")
                parts.append("```")
                parts.append(ctx.problems_memory)
                parts.append("```")
                parts.append("")

            # Include only the most relevant workflow file (Linux/Ubuntu build+test)
            # to avoid overwhelming the prompt. Full workflow info is in unified_summary.
            if getattr(ctx, "workflow_contents", None):
                # Filter for Linux/Ubuntu workflows and limit to 1-2 most relevant
                linux_workflows = []
                for path, content in ctx.workflow_contents:
                    if content and content.strip():
                        path_lower = path.lower()
                        content_lower = content.lower()
                        # Prioritize workflows that mention linux/ubuntu and test/build
                        is_linux = any(kw in content_lower for kw in ['ubuntu', 'linux', 'runs-on: ubuntu'])
                        is_test_build = any(kw in path_lower or kw in content_lower
                                           for kw in ['test', 'build', 'ci', 'check'])
                        if is_linux and is_test_build:
                            linux_workflows.append((path, content, 2))  # High priority
                        elif is_linux:
                            linux_workflows.append((path, content, 1))  # Medium priority
                        elif is_test_build:
                            linux_workflows.append((path, content, 0))  # Low priority

                # Sort by priority and take top 1-2
                linux_workflows.sort(key=lambda x: x[2], reverse=True)
                selected_workflows = linux_workflows[:2]

                if selected_workflows:
                    parts.append("Most relevant CI/CD workflow (Linux build/test):")
                    for path, content, _ in selected_workflows:
                        parts.append(f"\nFile: {path}\n```yaml\n{content}\n```")
                    parts.append("")

            if getattr(ctx, "dockerfile_contents", None):
                parts.append("Existing Dockerfiles found in the repository (may contain setup hints):")
                for path, content in ctx.dockerfile_contents:
                    if content and content.strip():
                        parts.append(f"\nFile: {path}\n```dockerfile\n{content}\n```")
                parts.append("")

            # NOTE: Requirement files and README content are NOT included verbatim in the prompt.
            # They are used as input to generate the unified summary below, which extracts
            # the relevant dependencies and installation instructions in a concise format.

            if getattr(ctx, "unified_summary", None):
                parts.append("Setup/Build/Test Summary (extracted from README, requirements, CI workflows, and web search):")
                parts.append("```")
                parts.append(ctx.unified_summary)
                parts.append("```")
                parts.append("")

        # Add previous attempts section if available
        if self.previous_attempts:
            parts.append("=" * 80)
            parts.append("CRITICAL: PREVIOUS ATTEMPT SUMMARIES")
            parts.append("=" * 80)
            parts.append("")
            parts.append("The following attempts have ALREADY been made to set up and run this project.")
            parts.append("YOU MUST learn from these to avoid repeating the same mistakes.")
            parts.append("Build upon previous progress and try DIFFERENT approaches where previous ones failed.")
            parts.append("")

            for idx, attempt in enumerate(self.previous_attempts, 1):
                parts.append(f"{'=' * 40}")
                parts.append(f"ATTEMPT {idx} OF {len(self.previous_attempts)}")
                parts.append(f"{'=' * 40}")
                parts.append("")

                # Helper to safely convert values to strings (handles dicts, lists, etc.)
                def to_str(val):
                    if val is None:
                        return ""
                    if isinstance(val, str):
                        return val
                    return json.dumps(val, ensure_ascii=False)

                if "problems" in attempt:
                    parts.append("## PROBLEMS ENCOUNTERED (where the agent got stuck):")
                    parts.append(to_str(attempt['problems']))
                    parts.append("")

                if "actions" in attempt:
                    parts.append("## SEQUENCE OF ACTIONS TAKEN:")
                    parts.append(to_str(attempt['actions']))
                    parts.append("")

                if "progress" in attempt:
                    parts.append("## PROGRESS MADE:")
                    parts.append(to_str(attempt['progress']))
                    parts.append("")

                if "lessons" in attempt:
                    parts.append("## KEY LESSONS LEARNED:")
                    parts.append(to_str(attempt['lessons']))
                    parts.append("")

                if "suggestions" in attempt:
                    parts.append("## SUGGESTIONS FOR THIS ATTEMPT (follow these!):")
                    parts.append(to_str(attempt['suggestions']))
                    parts.append("")

                if "dockerfile_recommendation" in attempt:
                    parts.append("## RECOMMENDED DOCKERFILE:")
                    parts.append("```dockerfile")
                    parts.append(to_str(attempt['dockerfile_recommendation']))
                    parts.append("```")
                    parts.append("")

                if "dockerfile_used" in attempt:
                    parts.append("## DOCKERFILE THAT WAS USED (for reference):")
                    parts.append("```dockerfile")
                    parts.append(to_str(attempt['dockerfile_used']))
                    parts.append("```")
                    parts.append("")

            parts.append("=" * 80)
            parts.append("END OF PREVIOUS ATTEMPTS - NOW START YOUR NEW ATTEMPT")
            parts.append("=" * 80)
            parts.append("")

        # Add container state awareness message
        parts.append("=" * 60)
        parts.append("CURRENT STATE:")
        if self.container is not None:
            parts.append("You are currently operating INSIDE a Docker container.")
            parts.append("The container is running and ready for commands.")
            parts.append("You have full access to execute commands inside the container.")
        else:
            parts.append("You have NOT yet created a Docker container.")
            parts.append("Your first priority should be to write a Dockerfile to create a container.")
            parts.append("Until a container is created, only limited commands are available (ls, cat, grep, find, etc.).")
            # Show local repo availability message only when no container exists
            if ctx is not None and getattr(ctx, "local_repo_available", False):
                parts.append("")
                parts.append("LOCAL REPOSITORY ACCESS:")
                parts.append(f"The target repository has been cloned locally to: {self.workspace_path}/{ctx.project_path}")
                parts.append("You can explore the codebase using terminal commands (ls, cat, grep, find, etc.) to understand")
                parts.append("the project structure, dependencies, and build system before creating your Dockerfile.")
                parts.append("Once you create a container, you should clone the repository inside the container.")
        parts.append("=" * 60)
        parts.append("")

        if self.commands_and_summary:
            parts.append("History of executed commands and summaries (most recent last):")
            for call_str, summ in self.commands_and_summary[-20:]:
                parts.append(f"- {call_str}\n  Summary: {json.dumps(summ, ensure_ascii=False)}")
            parts.append("")

        parts.append("High-level objective:")
        parts.append(task)
        parts.append("")

        parts.append("Tool interface reminder:")
        parts.append(self.tools_doc_string or "")
        parts.append("")

        parts.append("Planning instruction:")
        parts.append(self.cycle_instruction.strip())

        parts.append("\nIMPORTANT: Output ONLY a single JSON object for the Response schema. No extra text.")
        return "\n".join(parts)

    def _build_previous_cycle_messages(self) -> List[Dict[str, str]]:
        """
        Build messages representing the previous cycle's assistant response and result.
        Returns a list of messages to append after the initial user prompt.
        """
        messages: List[Dict[str, str]] = []

        if self.last_action is None:
            return messages

        # Build assistant message with thoughts and command from previous cycle
        assistant_parts = []
        if self.last_thoughts:
            assistant_parts.append(f"Here are my thoughts from the previous cycle:\n{self.last_thoughts}")
            assistant_parts.append("")

        assistant_parts.append(f"The last command I suggested was:")
        assistant_parts.append(f"Tool: '{self.last_action.get('name')}'")
        assistant_parts.append(f"Arguments: {json.dumps(self.last_action.get('args', {}), ensure_ascii=False)}")

        messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})

        # Build user message with result and guiding text
        user_parts = []
        user_parts.append("I have executed the last command and here is the result:")
        user_parts.append("")
        user_parts.append("```")
        user_parts.append(self.last_result or "(no output)")
        user_parts.append("```")
        user_parts.append("")

        # Add format error message if last response was invalid
        if self.last_format_error is not None:
            user_parts.append("=" * 60)
            user_parts.append("ERROR: Your last response was NOT valid and could NOT be parsed.")
            user_parts.append(f"Parsing error: {self.last_format_error}")
            user_parts.append("Please provide a new response with CORRECT JSON format.")
            user_parts.append("=" * 60)
            user_parts.append("")

        messages.append({"role": "user", "content": "\n".join(user_parts)})

        return messages

    def generate_attempt_summary(self, max_retries: int = 3) -> Dict[str, Any]:
        """
        Analyze the failed attempt using LLM to extract detailed lessons learned.

        Returns dict with:
          - problems: Detailed description of problems encountered and where agent got stuck
          - actions: Detailed sequence of actions taken to solve issues
          - progress: What was successfully completed vs what failed
          - lessons: Key technical insights about the project
          - suggestions: Specific, actionable next steps with commands
          - dockerfile_used: The Dockerfile content if one was created
        """
        # Build comprehensive analysis data from command history
        detailed_actions = []
        error_outputs = []
        successful_steps = []
        dockerfile_content = ""

        for idx, (call_str, summary) in enumerate(self.commands_and_summary):
            action_entry = {"step": idx + 1, "command": call_str}

            # Extract summary details
            if isinstance(summary, dict):
                action_entry["summary"] = summary.get("summary", "")
                action_entry["setup_details"] = summary.get("Setup details:", "")
                action_entry["next_steps"] = summary.get("Meaningful next setps", "")

                # Check for error indicators in summary
                summary_text = str(summary.get("summary", "")).lower()
                if any(err in summary_text for err in ["error", "failed", "not found", "cannot", "missing", "denied"]):
                    error_outputs.append({
                        "step": idx + 1,
                        "command": call_str,
                        "error_summary": summary.get("summary", "")
                    })
                else:
                    successful_steps.append({
                        "step": idx + 1,
                        "command": call_str,
                        "result": summary.get("summary", "")[:200]
                    })

            detailed_actions.append(action_entry)

        # Check for Dockerfile in written files
        for target, location, path, content in getattr(self, "written_files", []):
            if "dockerfile" in target.lower():
                dockerfile_content = content

        # Build comprehensive prompt
        prompt = f"""You are analyzing a failed attempt to set up and run tests for a project.
Your goal is to provide a DETAILED and COMPREHENSIVE summary that will help the next attempt succeed.

## PROJECT CONTEXT
- Project URL: {getattr(self, 'project_url', 'Unknown')}
- Project Path: {getattr(self, 'project_path', 'Unknown')}
- Total cycles executed: {len(self.commands_and_summary)}

## DOCKERFILE USED (if any)
```dockerfile
{dockerfile_content if dockerfile_content else "No Dockerfile was created in this attempt"}
```

## DETAILED ACTION SEQUENCE
The following actions were taken in order:

{chr(10).join(self._format_action_for_summary(action) for action in detailed_actions[-40:])}

## ERRORS AND FAILURES ENCOUNTERED
{chr(10).join(f"- Step {e['step']}: {e['command'][:100]}... -> {e['error_summary'][:300]}" for e in error_outputs[-15:]) if error_outputs else "No explicit errors captured in summaries"}

## SUCCESSFUL STEPS
{chr(10).join(f"- Step {s['step']}: {s['command'][:80]}... -> {s['result']}" for s in successful_steps[-10:]) if successful_steps else "No clearly successful steps identified"}

## ANALYSIS TASK
Based on the above information, provide a DETAILED JSON analysis with the following fields:

1. **problems**: A DETAILED description of:
   - The specific technical problems encountered (with exact error messages if available)
   - Where exactly the agent got stuck (which step, which command)
   - Root causes of failures (missing dependencies, wrong versions, configuration issues, etc.)

2. **actions**: A DETAILED sequence of actions taken, including:
   - What was tried first and why
   - How the approach evolved as problems were encountered
   - What workarounds were attempted

3. **progress**: What was successfully completed vs what remains:
   - List what worked (e.g., "Docker image built successfully", "Dependencies installed")
   - List what failed or is incomplete (e.g., "Tests failed to run", "Build step failed")
   - Percentage estimate of overall progress (e.g., "70% complete - environment ready but tests not running")

4. **lessons**: Key technical insights about this specific project:
   - Required dependencies and versions discovered
   - Configuration requirements discovered
   - Build/test commands that work or don't work
   - Project-specific quirks or requirements

5. **suggestions**: SPECIFIC, ACTIONABLE next steps with exact commands:
   - What should be done differently in the next attempt
   - Exact commands to try
   - Alternative approaches if the current one is not working
   - Things to avoid based on what failed

6. **dockerfile_recommendation**: If a Dockerfile was used, suggest improvements or a completely new Dockerfile content

IMPORTANT: Be DETAILED and SPECIFIC. Vague summaries are not helpful.
Respond ONLY with a valid JSON object. Do not include any text before or after the JSON.

Example of the expected detail level:
{{
  "problems": "Build failed at step 15 with error 'ModuleNotFoundError: No module named scipy.special'. The Dockerfile installed scipy 1.9.0 but the project requires scipy>=1.11.0 for the special submodule. Additionally, numpy version conflict detected - project needs numpy>=1.24 but got 1.21.",
  "actions": "1. Created Dockerfile with python:3.11-slim base. 2. Installed build dependencies (gcc, gfortran). 3. Cloned repository. 4. Ran pip install -r requirements.txt which succeeded. 5. Attempted pytest which failed on scipy import. 6. Tried upgrading scipy with pip install --upgrade scipy but hit numpy conflict. 7. Attempted to resolve with pip install numpy>=1.24 scipy>=1.11 but still failed.",
  "progress": "70% complete. Environment setup done, dependencies partially installed, but test execution blocked by version conflicts. Docker container is running and accessible.",
  "lessons": "This project requires scipy>=1.11.0 with the special submodule. The requirements.txt has loose version constraints that cause conflicts. Need to pin exact versions. Project uses pytest with custom conftest.py.",
  "suggestions": "1. Use python:3.11 (not slim) for better compatibility. 2. Install specific versions: pip install 'numpy==1.24.3' 'scipy==1.11.4'. 3. Run tests with: python -m pytest tests/ -v. 4. If scipy.special still fails, try building scipy from source with gfortran.",
  "dockerfile_recommendation": "FROM python:3.11\\nRUN apt-get update && apt-get install -y gcc gfortran libopenblas-dev\\nWORKDIR /app\\nRUN git clone <repo_url> .\\nRUN pip install 'numpy==1.24.3' 'scipy==1.11.4'\\nRUN pip install -e .[test]"
}}
"""

        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                resp = self.model.query(messages=[{"role": "user", "content": prompt}], temperature=0.7)
                content = (resp.get("content") or "").strip()

                # Try to extract JSON
                summary = _extract_json_object(content)

                # Validate required fields (progress and dockerfile_recommendation are optional)
                required_fields = ['problems', 'actions', 'lessons', 'suggestions']
                if all(field in summary for field in required_fields):
                    # Add dockerfile content to summary for reference
                    if dockerfile_content and 'dockerfile_used' not in summary:
                        summary['dockerfile_used'] = dockerfile_content
                    return summary

                # Missing fields - try again
                last_error = ValueError(f"Missing required fields. Got: {list(summary.keys())}")
                _LOG.warning(f"Attempt summary retry {attempt + 1}/{max_retries}: {last_error}")
                continue

            except Exception as e:
                last_error = e
                _LOG.warning(f"Failed to generate attempt summary (attempt {attempt + 1}/{max_retries}): {e}")
                continue

        # All retries exhausted - return detailed fallback summary
        _LOG.error(f"Failed to generate proper attempt summary after {max_retries} retries. Using fallback.")

        # Build a detailed fallback from available data
        fallback_actions = []
        for action in detailed_actions[-10:]:
            cmd = action.get("command", "")[:100]
            summary_text = action.get("summary", "")[:150]
            fallback_actions.append(f"- {cmd}: {summary_text}")

        fallback_errors = []
        for err in error_outputs[-5:]:
            fallback_errors.append(f"- Step {err['step']}: {err['error_summary'][:200]}")

        return {
            "problems": f"Errors encountered: {chr(10).join(fallback_errors) if fallback_errors else 'Unable to extract specific errors'}",
            "actions": f"Actions taken ({len(detailed_actions)} total):{chr(10)}{chr(10).join(fallback_actions) if fallback_actions else 'No actions recorded'}",
            "progress": f"Executed {len(detailed_actions)} steps. {len(successful_steps)} succeeded, {len(error_outputs)} had errors.",
            "lessons": f"Unable to extract detailed lessons (error: {last_error}). Review the action sequence above.",
            "suggestions": "Review the errors above and try a different approach. Check dependency versions and build requirements.",
            "dockerfile_used": dockerfile_content if dockerfile_content else "No Dockerfile was created"
        }

    def _format_action_for_summary(self, action: Dict[str, Any]) -> str:
        """Format a single action entry for the summary prompt."""
        step = action.get("step", "?")
        cmd = action.get("command", "Unknown command")
        summary = action.get("summary", "")
        setup_details = action.get("setup_details", "")
        next_steps = action.get("next_steps", "")

        parts = [f"### Step {step}"]
        parts.append(f"**Command**: {cmd}")

        if summary:
            parts.append(f"**Result Summary**: {summary[:500]}")
        if setup_details:
            parts.append(f"**Setup Details**: {setup_details[:300]}")
        if next_steps:
            parts.append(f"**Suggested Next Steps**: {next_steps[:200]}")

        parts.append("")  # Empty line between steps
        return "\n".join(parts)

    # -------------------------
    # planning/execution/summarization
    # -------------------------
    def _plan_next_action(self, task: str, max_retries: int = 5) -> Dict[str, Any]:
        system_msg = """
You are an AI assistant specialized in automatically setting up a given project and making it ready to run (by installing dependencies and making the correct configurations). Your role involves automating the process of gathering project information/requirements and dependencies, setting up the execution environment, and running test suites. You should always gather essential details such as language and version, dependencies, and testing frameworks; Following that you set up the environment and execute test suites based on collected information; Finally, you assess test outcomes, identify failing cases. Your personality is characterized by efficiency, attention to detail, and a commitment to streamlining the installation and tests execution of the given project. You are also good at analyzing feedback from environment and always produce very long detailed thoughts that analyze the situation and feedback from different angles to the point where an outsider walking in would understand everything from just reading that. Focus on your previous list of actions and on the information and context given with and use your reasoning and analytical skill to take suitable actions. In summary, MAIN TASK := SETUP AND RUN TESTS OF THE TARGET PROJECT INSIDE A DOCKER CONTAINER."""
        user_msg = self._build_instance_prompt(task)

        # keep prompt text for debugging
        self.prompt_text = system_msg + "\n\n" + user_msg

        # Build base messages: system, initial user prompt, then previous cycle's assistant/user messages
        base_messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        base_messages.extend(self._build_previous_cycle_messages())

        last_error: Optional[FormatError] = None

        for attempt in range(max_retries):
            # Build messages for this attempt
            messages = list(base_messages)

            # If this is a retry, add the failed response and error feedback
            if last_error is not None:
                messages.append({"role": "assistant", "content": self._last_failed_response or ""})
                error_msg = (
                    f"ERROR: Your response was NOT valid and could NOT be parsed.\n"
                    f"Parsing error: {last_error}\n"
                    f"Please provide a new response with CORRECT JSON format. "
                    f"Output ONLY a single JSON object matching the Response schema."
                )
                messages.append({"role": "user", "content": error_msg})

            _LOG.info(
                f"Querying LLM for next action (cycle {self.cycle_count + 1}, "
                f"attempt {attempt + 1}/{max_retries}, messages={len(messages)})..."
            )
            resp = self.model.query(messages=messages, temperature=1)
            _LOG.info(f"LLM response received for cycle {self.cycle_count + 1}, extracting content...")
            content = resp.get("content", "") or ""
            _LOG.info(f"Content extracted ({len(content)} chars), adding to messages...")

            # record assistant content for cycle logging (does not change workflow)
            self.add_message("assistant", content)
            _LOG.info(f"Message added, parsing JSON...")

            try:
                payload = _extract_json_object(content)
                _LOG.info(f"JSON parsed successfully, extracting command...")

                cmd = payload.get("command") or {}
                name = cmd.get("name")
                args = cmd.get("args", {})
                _LOG.info(f"Command extracted: name={name}, args_keys={list(args.keys()) if isinstance(args, dict) else 'not-dict'}")

                if not isinstance(name, str) or not name.strip():
                    raise FormatError("Missing or invalid command.name")

                if not isinstance(args, dict):
                    args = {}

                _LOG.info(f"Normalizing and validating args for tool '{name}'...")
                norm_args = self.tool_registry.normalize_and_validate(name, args)
                _LOG.info(f"Args normalized, returning planned action")

                # Clear format error on success
                self.last_format_error = None
                return {"tool_name": name, "tool_args": norm_args, "raw": payload}

            except FormatError as e:
                last_error = e
                self._last_failed_response = content
                _LOG.warning(f"Format error on attempt {attempt + 1}/{max_retries}: {e}")
                continue

        # All retries exhausted - store error and raise
        self.last_format_error = str(last_error)
        raise FormatError(f"Failed after {max_retries} attempts. Last error: {last_error}")

    def _summarize_last_command(self, last_output: str) -> Dict[str, Any]:
        # Use larger limits for summarization (30K head, 5K middle, 25K tail = ~60K total)
        # Summarizer benefits from more context to produce accurate summaries
        truncated_output = _smart_truncate_output(
            last_output or "",
            head_chars=30000,
            middle_chars=5000,
            tail_chars=25000,
        )
        prompt = self.summarize_cycle.strip() + "\n\n" + truncated_output
        _LOG.info(f"Querying LLM for summarization (cycle {self.cycle_count})...")
        resp = self.model.query(messages=[{"role": "user", "content": prompt}], temperature=1)
        _LOG.info(f"LLM summarization response received for cycle {self.cycle_count}")
        txt = (resp.get("content") or "").strip()

        try:
            return _extract_json_object(txt)
        except Exception:
            return {
                "summary": txt[:2000],
                "Setup details:": "",
                "Meaningful next setps": "",
            }

    def _execute_action(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        _LOG.info(f"[EXECUTE] Starting execution of tool '{name}'...")
        candidate = {"name": name, "args": args}
        if is_repetition(self._last_commands(), candidate):
            raise FormatError(
                "Repetition detected in the last commands. Choose a different tool and/or a different approach."
            )

        # tool may raise GoalsAccomplished; allow it to propagate
        _LOG.info(f"[EXECUTE] Calling tool registry for '{name}'...")
        result = self.tool_registry.call(name, args, agent=self)
        _LOG.info(f"[EXECUTE] Tool '{name}' completed, result length: {len(str(result))} chars")

        return {
            "output": str(result),
            "returncode": 0,
            "tool_name": name,
            "tool_args": args,
        }

    # -------------------------
    # cycle API (for logging wrappers)
    # -------------------------
    def run_one_cycle(self, task: str) -> Dict[str, Any]:
        """
        Execute exactly one plan→tool→summarize cycle.

        This method is intentionally thin and does not change workflow.
        It exists to make cycle logging clean and non-invasive.
        """
        self.cycle_count += 1
        _LOG.info(f"[CYCLE] Starting cycle {self.cycle_count}")

        _LOG.info(f"[CYCLE] Phase 1: Planning next action...")
        planned = self._plan_next_action(task)
        name = planned["tool_name"]
        args = planned["tool_args"]
        _LOG.info(f"[CYCLE] Phase 1 complete: tool='{name}'")

        _LOG.info(f"[CYCLE] Phase 2: Executing action...")
        raw = self._execute_action(name, args)
        _LOG.info(f"[CYCLE] Phase 2 complete: output length={len(raw.get('output', ''))}")

        _LOG.info(f"[CYCLE] Phase 3: Summarizing output...")
        summary = self._summarize_last_command(raw["output"])
        _LOG.info(f"[CYCLE] Phase 3 complete: summary keys={list(summary.keys()) if isinstance(summary, dict) else 'not-dict'}")

        # Store last action, result, and thoughts for next prompt
        # Truncate output to prevent context window overflow (keeps first 10K, middle 2K, last 10K)
        _LOG.info(f"[CYCLE] Storing state for next cycle...")
        self.last_action = {"name": name, "args": args}
        self.last_result = _smart_truncate_output(raw["output"])
        self.last_thoughts = planned["raw"].get("thoughts", "")

        self.commands_and_summary.append(
            (f"Call to tool {name} with arguments {json.dumps(args)}", summary)
        )
        _LOG.info(f"[CYCLE] Cycle {self.cycle_count} complete")

        return {
            "tool_call": {"command": {"name": name, "args": args}},
            "result": raw,
            "summary": summary,
        }

    # -------------------------
    # main loop
    # -------------------------
    def run(self, task: str, **_: Any) -> None:
        # Get state persistence if available
        state_persistence: Optional[StatePersistence] = getattr(self, "_state_persistence", None)

        for step in range(1, self.step_limit + 1):
            try:
                self.run_one_cycle(task)
                # last_format_error is cleared in _plan_next_action on success

                # Save state after every cycle for better recovery from crashes/hangs
                if state_persistence:
                    save_agent_state_periodically(self, state_persistence, interval=1)

                continue

            except GoalsAccomplished:
                # Save final state before returning
                if state_persistence:
                    state_persistence.save_state(self)
                return

            except FormatError as e:
                # FormatError here means all retries in _plan_next_action were exhausted
                # last_format_error is already set by _plan_next_action
                # Record into compact memory and continue to next step
                note = f"Internal note (format error after retries exhausted) at step {step}"
                self.commands_and_summary.append((note, {"summary": str(e)}))
                self.add_message("developer", str(e), tag="format_error")
                continue

            except Exception as e:
                # Clear format error for non-format exceptions
                self.last_format_error = None
                # hard safety: do not crash silently; preserve as memory and continue
                note = f"Internal note (unexpected error) at step {step}"
                self.commands_and_summary.append((note, {"summary": f"{type(e).__name__}: {e}"}))
                self.add_message("developer", f"{type(e).__name__}: {e}", tag="unexpected_error")
                _LOG.error(f"Unexpected error at step {step}: {e}", exc_info=True)

                # Save state on errors for recovery
                if state_persistence:
                    state_persistence.save_state(self)

                continue

        # Step limit reached without accomplishing goals - raise BudgetExhausted
        self.commands_and_summary.append(("Run stopped", {"summary": f"Step limit ({self.step_limit}) reached."}))

        # Save final state before raising
        if state_persistence:
            state_persistence.save_state(self)

        raise BudgetExhausted(f"Step limit ({self.step_limit}) reached without accomplishing goals.")
