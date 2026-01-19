# execution_agent/context.py
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    import requests
except ImportError:
    requests = None  # type: ignore

_LOG = logging.getLogger("execution_agent.context")


def _llm_filter_cicd_file(
    file_path: str,
    file_content: str,
    model,
) -> Optional[str]:
    """
    Use an LLM to filter a CI/CD file and extract only the relevant parts.

    The LLM extracts parts related to:
    - Installing the repository from source code on Linux
    - Running test cases

    Args:
        file_path: Path to the CI/CD file (for context)
        file_content: Raw content of the CI/CD file
        model: The model instance to use for filtering

    Returns:
        Filtered content with only relevant parts, or None if the file is not relevant
    """
    if not file_content or not file_content.strip():
        return None

    if model is None:
        _LOG.warning("No model provided for LLM filtering, falling back to heuristic extraction")
        return None

    prompt = f"""You are helping with a specific task: INSTALLING A SOFTWARE PROJECT FROM SOURCE CODE AND RUNNING ITS TEST SUITE inside a fresh Linux Docker container.

You are analyzing a CI/CD configuration file to extract ONLY the parts that would help accomplish this task.

FILE PATH: {file_path}

FILE CONTENT:
```
{file_content[:50000]}
```

CONTEXT - THE OVERALL GOAL:
We need to:
1. Clone a repository into a fresh Docker container (Ubuntu-based)
2. Install all necessary system dependencies and language runtimes
3. Build/install the project from source code
4. Run the project's test suite successfully

YOUR TASK:
1. First, determine if this CI/CD file contains information useful for the above goal.

   A file IS RELEVANT if it contains:
   - Commands to install system packages or language runtimes needed to build/run the project
   - Commands to install project dependencies (pip install, npm install, cargo build, etc.)
   - Commands to build or compile the project from source
   - Commands to run tests (pytest, npm test, cargo test, make test, etc.)
   - Environment setup needed before building/testing (env vars, services like databases)
   - Container/Docker image specifications that show what base image or packages are needed

   A file is NOT RELEVANT if it ONLY contains:
   - Documentation building/deployment (mkdocs, sphinx, readthedocs, GitHub Pages)
   - Code formatting/linting checks only (black, flake8, prettier, eslint) without build/test
   - Release/publishing workflows (PyPI uploads, npm publish, Docker Hub push)
   - Badge/status updates, notifications
   - Security scanning without actual build/test steps
   - Production/staging deployment (we only care about building and testing locally)

2. If the file is NOT relevant to building from source and running tests, respond with exactly: NOT_RELEVANT

3. If the file IS relevant, extract and output ONLY the parts useful for our goal:
   - Shell commands (run:) that install dependencies, build the project, or run tests
   - The specific test commands used (this is crucial - we need to know how to run tests)
   - Required system packages and their installation commands
   - Language/runtime version requirements (Python version, Node version, etc.)
   - Environment variables needed for building or testing
   - Container images or services (databases, redis, etc.) required for tests
   - Any special setup steps or scripts that must run before tests

   REMOVE (not useful for our goal):
   - Trigger configurations (on:, push:, pull_request:, schedule:)
   - Permissions blocks
   - Caching configurations (we'll run fresh each time)
   - Artifact upload/download (CI-specific)
   - Secret/token references (we won't have these)
   - GitHub-specific output commands
   - Matrix configurations (just note "runs on multiple Python versions" if present)
   - Jobs related to docs, linting-only, releases, or deployments

Output the extracted content in a clean, readable format. PRESERVE SHELL COMMANDS EXACTLY as written - these are the most valuable information for reproducing the build and test process."""

    try:
        response = model.query([{"role": "user", "content": prompt}])
        result = response.get("content", "").strip()

        if not result or result == "NOT_RELEVANT":
            _LOG.info(f"LLM determined '{file_path}' is not relevant for build/test")
            return None

        _LOG.info(f"LLM filtered '{file_path}' - extracted relevant content")
        return result

    except Exception as e:
        _LOG.warning(f"LLM filtering failed for '{file_path}': {e}, falling back to heuristic extraction")
        return None


def _extract_relevant_workflow_parts(workflow_content: str) -> str:
    """
    Extract only the relevant parts from a GitHub Actions workflow YAML.

    Keeps:
    - Job names and their 'runs-on' values
    - 'run:' commands (the actual shell commands)
    - 'uses:' actions that are relevant (setup-*, install-*, build-*, test-*)
    - Environment variables that might be relevant
    - Container/services definitions

    Removes:
    - Trigger configurations (on:, push:, pull_request:, schedule:, etc.)
    - Permissions blocks
    - Concurrency settings
    - Most 'with:' blocks (except for relevant setup actions)
    - Matrix configurations (just keeps the concept, not all variants)
    - Caching configurations
    - Artifact upload/download details
    - GitHub-specific tokens and secrets references
    """
    if not workflow_content or not workflow_content.strip():
        return ""

    # If yaml is not available, do basic text filtering
    if yaml is None:
        return _extract_workflow_parts_regex(workflow_content)

    try:
        data = yaml.safe_load(workflow_content)
        if not isinstance(data, dict):
            return _extract_workflow_parts_regex(workflow_content)
        return _extract_workflow_parts_structured(data)
    except Exception:
        return _extract_workflow_parts_regex(workflow_content)


def _extract_workflow_parts_structured(data: Dict[str, Any]) -> str:
    """Extract relevant parts from parsed YAML workflow."""
    lines = []

    # Extract workflow name
    if data.get("name"):
        lines.append(f"Workflow: {data['name']}")

    # Extract environment variables at workflow level
    if data.get("env"):
        env_vars = data["env"]
        relevant_env = {k: v for k, v in env_vars.items()
                       if not any(secret in str(v).upper() for secret in ["SECRET", "TOKEN", "KEY", "PASSWORD"])}
        if relevant_env:
            lines.append(f"Environment: {relevant_env}")

    jobs = data.get("jobs", {})
    if not jobs:
        return "\n".join(lines)

    for job_name, job_data in jobs.items():
        if not isinstance(job_data, dict):
            continue

        lines.append(f"\nJob: {job_name}")

        # Runs-on is useful to know the target OS
        if job_data.get("runs-on"):
            runs_on = job_data["runs-on"]
            # Simplify matrix expressions
            if isinstance(runs_on, str) and "${{" in runs_on:
                runs_on = "linux/macos/windows (matrix)"
            lines.append(f"  runs-on: {runs_on}")

        # Container info is very relevant
        if job_data.get("container"):
            container = job_data["container"]
            if isinstance(container, str):
                lines.append(f"  container: {container}")
            elif isinstance(container, dict) and container.get("image"):
                lines.append(f"  container: {container['image']}")

        # Services (like databases) are relevant
        if job_data.get("services"):
            services = job_data["services"]
            service_names = list(services.keys()) if isinstance(services, dict) else []
            if service_names:
                lines.append(f"  services: {', '.join(service_names)}")

        # Extract steps
        steps = job_data.get("steps", [])
        if not steps:
            continue

        lines.append("  steps:")
        for step in steps:
            if not isinstance(step, dict):
                continue

            step_name = step.get("name", "")

            # Handle 'uses:' actions
            if step.get("uses"):
                uses = step["uses"]
                # Only include relevant setup/build/test actions
                relevant_actions = ["setup-", "install-", "build-", "test-", "python", "node", "java", "go", "rust", "docker"]
                if any(action in uses.lower() for action in relevant_actions):
                    action_line = f"    - uses: {uses}"
                    # Include relevant 'with' parameters for setup actions
                    if step.get("with"):
                        with_data = step["with"]
                        # Filter out tokens, keys, etc.
                        relevant_with = {}
                        for k, v in with_data.items():
                            k_lower = k.lower()
                            v_str = str(v).upper()
                            if any(secret in k_lower or secret in v_str for secret in ["token", "secret", "key", "password", "credential"]):
                                continue
                            if k_lower in ["version", "python-version", "node-version", "java-version", "go-version", "architecture"]:
                                relevant_with[k] = v
                        if relevant_with:
                            action_line += f" (with: {relevant_with})"
                    lines.append(action_line)

            # Handle 'run:' commands - these are the most valuable
            if step.get("run"):
                run_cmd = step["run"].strip()
                # Skip if it's just echoing or setting outputs
                if run_cmd.startswith("echo ") and "GITHUB_OUTPUT" in run_cmd:
                    continue
                if "::set-output" in run_cmd or "::add-mask" in run_cmd:
                    continue

                if step_name:
                    lines.append(f"    - {step_name}:")
                lines.append(f"      run: |")
                for cmd_line in run_cmd.split("\n"):
                    cmd_line = cmd_line.strip()
                    if cmd_line:
                        lines.append(f"        {cmd_line}")

    return "\n".join(lines)


def _extract_workflow_parts_regex(workflow_content: str) -> str:
    """Fallback regex-based extraction when YAML parsing is unavailable."""
    lines = []
    in_run_block = False
    run_indent = 0

    # Patterns to skip
    skip_patterns = [
        r"^\s*on:\s*$",
        r"^\s*push:\s*$",
        r"^\s*pull_request",
        r"^\s*schedule:",
        r"^\s*workflow_dispatch",
        r"^\s*permissions:",
        r"^\s*concurrency:",
        r"^\s*branches:",
        r"^\s*tags:",
        r"^\s*paths:",
        r"^\s*types:",
        r"^\s*cron:",
        r"^\s*#",  # Comments
        r"\$\{\{\s*secrets\.",  # Secret references
        r"\$\{\{\s*github\.token",
    ]

    # Patterns to keep
    keep_patterns = [
        r"^\s*name:",
        r"^\s*jobs:",
        r"^\s*runs-on:",
        r"^\s*container:",
        r"^\s*services:",
        r"^\s*steps:",
        r"^\s*run:\s*[|>]?\s*$",
        r"^\s*run:\s+\S",
        r"^\s*-\s+name:",
        r"^\s*uses:\s+.*(setup|install|build|test|python|node|java|docker)",
        r"^\s*env:",
        r"^\s*(python|node|java|go)-version:",
    ]

    for line in workflow_content.split("\n"):
        # Check if we should skip this line
        should_skip = any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns)
        if should_skip and not in_run_block:
            continue

        # Track run blocks
        if re.match(r"^\s*run:\s*[|>]?\s*$", line):
            in_run_block = True
            run_indent = len(line) - len(line.lstrip())
            lines.append(line)
            continue

        if in_run_block:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else run_indent + 1
            if current_indent > run_indent or not line.strip():
                lines.append(line)
                continue
            else:
                in_run_block = False

        # Check if we should keep this line
        should_keep = any(re.match(pattern, line, re.IGNORECASE) for pattern in keep_patterns)
        if should_keep:
            lines.append(line)

    return "\n".join(lines)

@dataclass
class RepoContext:
    project_path: str
    project_url: str
    language: str
    workflows: List[str]
    workflow_contents: List[Tuple[str, str]]          # (path, content)
    dockerfiles: List[str]
    dockerfile_contents: List[Tuple[str, str]]        # (path, content)
    search_results: List[Dict[str, Any]]              # loaded from cache or produced elsewhere
    requirement_files: List[Tuple[str, str]] = None   # (path, content) - dependency/requirement files
    readme_content: Optional[str] = None              # README content (installation/build instructions)
    unified_summary: Optional[str] = None
    problems_memory: Optional[str] = None
    local_repo_available: bool = False                # True if repo was cloned locally during preprocessing

    def __post_init__(self):
        if self.requirement_files is None:
            self.requirement_files = []

class ContextBuilder:
    KEYWORDS = ["test", "build", "linux", "unittest", "integration", "deploy"]

    def __init__(
        self,
        *,
        workspace_root: str = "execution_agent_workspace",
        search_logs_root: str = "search_logs",
        problems_memory_root: str = "problems_memory",
    ):
        self.workspace_root = workspace_root
        self.search_logs_root = search_logs_root
        self.problems_memory_root = problems_memory_root

    def _shorten_path(self, full_path: str, project_name: str) -> str:
        """
        Shorten an absolute path to just workspace/project/relative_path.

        For example:
            /home/user/mini_execution_agent/execution_agent_workspace/pandas/.github/workflows/test.yml
        becomes:
            execution_agent_workspace/pandas/.github/workflows/test.yml
        """
        # Try to find the workspace_root in the path and extract from there
        workspace_base = os.path.basename(self.workspace_root.rstrip(os.sep))

        # Find where workspace_root appears in the path
        try:
            # Get the absolute workspace root for comparison
            abs_workspace = os.path.abspath(self.workspace_root)
            abs_path = os.path.abspath(full_path)

            if abs_path.startswith(abs_workspace):
                # Extract the relative part after workspace_root
                rel_path = os.path.relpath(abs_path, os.path.dirname(abs_workspace))
                return rel_path
        except (ValueError, OSError):
            pass

        # Fallback: try to find workspace_base/project_name pattern
        pattern = os.path.join(workspace_base, project_name)
        idx = full_path.find(pattern)
        if idx != -1:
            return full_path[idx:]

        # Last resort: just return the original path
        return full_path

    # ---------- faithful file discovery ----------
    def find_workflows(self, project_name: str, filter_by_keywords: bool = False) -> List[str]:
        """
        Find workflow files in the project's .github/workflows directory.

        Args:
            project_name: Name of the project (subdirectory in workspace_root)
            filter_by_keywords: If True, only return files with keywords in filename.
                              If False (default), return all workflow files and let
                              LLM filtering determine relevance later.

        Returns:
            List of paths to workflow YAML files
        """
        found_files: List[str] = []
        workflow_dir = os.path.join(self.workspace_root, project_name, ".github", "workflows")
        if not os.path.isdir(workflow_dir):
            return []

        result = subprocess.run(
            ["find", workflow_dir, "-name", "*.yml", "-o", "-name", "*.yaml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return []

        for path in result.stdout.splitlines():
            if not path.strip():
                continue
            if filter_by_keywords:
                filename = os.path.basename(path).lower()
                if any(k in filename for k in self.KEYWORDS):
                    found_files.append(path)
            else:
                found_files.append(path)
        return found_files

    def find_dockerfiles(self, project_path: str) -> List[str]:
        proj_dir = os.path.join(self.workspace_root, project_path)
        result = subprocess.run(
            ["find", proj_dir, "-name", "Dockerfile"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return []
        return result.stdout.splitlines()

    # ---------- requirement files discovery (language-dependent) ----------

    # Language-specific requirement file patterns
    REQUIREMENT_FILE_PATTERNS: Dict[str, List[str]] = {
        # Python patterns
        "python": [
            "requirements.txt", "requirements*.txt", "requirements/*.txt",
            "setup.py", "setup.cfg", "pyproject.toml",
            "Pipfile", "Pipfile.lock", "poetry.lock",
            "environment.yml", "environment.yaml", "conda.yml", "conda.yaml",
            "tox.ini", ".python-version",
        ],
        # JavaScript/TypeScript patterns
        "javascript": [
            "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            "bun.lockb", ".nvmrc", ".node-version",
        ],
        "typescript": [
            "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
            "bun.lockb", ".nvmrc", ".node-version", "tsconfig.json",
        ],
        # Java/JVM patterns
        "java": [
            "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
            "gradle.properties", ".java-version", "gradlew",
        ],
        "kotlin": [
            "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
            "gradle.properties",
        ],
        "scala": [
            "build.sbt", "pom.xml", "build.gradle",
        ],
        # C/C++ patterns
        "c": [
            "CMakeLists.txt", "Makefile", "makefile", "GNUmakefile",
            "configure", "configure.ac", "configure.in",
            "meson.build", "conanfile.txt", "conanfile.py",
            "vcpkg.json", "WORKSPACE", "BUILD", "BUILD.bazel",
        ],
        "cpp": [
            "CMakeLists.txt", "Makefile", "makefile", "GNUmakefile",
            "configure", "configure.ac", "configure.in",
            "meson.build", "conanfile.txt", "conanfile.py",
            "vcpkg.json", "WORKSPACE", "BUILD", "BUILD.bazel",
        ],
        "c++": [
            "CMakeLists.txt", "Makefile", "makefile", "GNUmakefile",
            "configure", "configure.ac", "configure.in",
            "meson.build", "conanfile.txt", "conanfile.py",
            "vcpkg.json", "WORKSPACE", "BUILD", "BUILD.bazel",
        ],
        # Rust patterns
        "rust": [
            "Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml",
        ],
        # Go patterns
        "go": [
            "go.mod", "go.sum", "Gopkg.toml", "Gopkg.lock", "glide.yaml",
        ],
        # Ruby patterns
        "ruby": [
            "Gemfile", "Gemfile.lock", ".ruby-version", ".ruby-gemset",
            "*.gemspec",
        ],
        # PHP patterns
        "php": [
            "composer.json", "composer.lock",
        ],
        # .NET patterns
        "csharp": [
            "*.csproj", "*.sln", "packages.config", "nuget.config",
            "Directory.Build.props", "global.json",
        ],
        "fsharp": [
            "*.fsproj", "*.sln", "packages.config", "nuget.config",
        ],
        # Elixir patterns
        "elixir": [
            "mix.exs", "mix.lock",
        ],
        # Haskell patterns
        "haskell": [
            "stack.yaml", "cabal.project", "*.cabal", "package.yaml",
        ],
    }

    # Common patterns across all languages (always check these)
    COMMON_REQUIREMENT_PATTERNS: List[str] = [
        "Makefile", "makefile", "GNUmakefile",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        ".tool-versions",  # asdf version manager
    ]

    def find_requirement_files(self, project_path: str, language: str) -> List[str]:
        """
        Find requirement/dependency files in the repository based on language.

        This uses a heuristic approach:
        1. Look for language-specific requirement files
        2. Also look for common build system files
        3. Search at the root level and one level deep

        Args:
            project_path: Name of the project (subdirectory in workspace_root)
            language: Primary language of the project (e.g., 'python', 'java', 'c')

        Returns:
            List of paths to requirement/dependency files
        """
        found_files: List[str] = []
        proj_dir = os.path.join(self.workspace_root, project_path)

        if not os.path.isdir(proj_dir):
            return []

        # Get language-specific patterns
        lang_lower = language.lower().strip() if language else ""
        patterns = list(self.COMMON_REQUIREMENT_PATTERNS)

        # Add language-specific patterns
        if lang_lower in self.REQUIREMENT_FILE_PATTERNS:
            patterns.extend(self.REQUIREMENT_FILE_PATTERNS[lang_lower])
        else:
            # If language is unknown, check for common ones across multiple languages
            for lang_patterns in self.REQUIREMENT_FILE_PATTERNS.values():
                patterns.extend(lang_patterns)
            patterns = list(set(patterns))  # Remove duplicates

        # Search for files matching patterns
        for pattern in patterns:
            # Handle glob patterns (files with *)
            if "*" in pattern:
                # Use find with -name for glob patterns
                try:
                    result = subprocess.run(
                        ["find", proj_dir, "-maxdepth", "2", "-name", pattern, "-type", "f"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        for path in result.stdout.splitlines():
                            if path.strip() and path not in found_files:
                                found_files.append(path.strip())
                except Exception:
                    pass
            else:
                # Check for exact filename at root and one level deep
                root_path = os.path.join(proj_dir, pattern)
                if os.path.isfile(root_path) and root_path not in found_files:
                    found_files.append(root_path)

                # Check one level deep (common subdirs)
                for subdir in ["", "src", "lib", "pkg", "config", "build"]:
                    if subdir:
                        check_path = os.path.join(proj_dir, subdir, pattern)
                    else:
                        continue  # Already checked root
                    if os.path.isfile(check_path) and check_path not in found_files:
                        found_files.append(check_path)

        return found_files

    def find_readme(self, project_path: str) -> Optional[str]:
        """
        Find the main README file in the repository.

        Looks for common README file names at the root level.

        Args:
            project_path: Name of the project (subdirectory in workspace_root)

        Returns:
            Path to the README file if found, None otherwise
        """
        proj_dir = os.path.join(self.workspace_root, project_path)

        if not os.path.isdir(proj_dir):
            return None

        # Common README file names (in priority order)
        readme_names = [
            "README.md", "README.rst", "README.txt", "README",
            "readme.md", "readme.rst", "readme.txt", "readme",
            "Readme.md", "Readme.rst", "Readme.txt", "Readme",
            "INSTALL.md", "INSTALL.txt", "INSTALL",
            "CONTRIBUTING.md", "BUILDING.md",
        ]

        for name in readme_names:
            path = os.path.join(proj_dir, name)
            if os.path.isfile(path):
                return path

        return None

    def load_requirement_files(
        self,
        requirement_paths: List[str],
        project_name: str,
        max_file_size: int = 50_000,
    ) -> List[Tuple[str, str]]:
        """
        Load requirement/dependency file contents.

        Args:
            requirement_paths: List of paths to requirement files
            project_name: Name of the project (for shortening paths in output)
            max_file_size: Maximum file size to read (bytes)

        Returns:
            List of (shortened_path, content) tuples
        """
        out = []
        for p in requirement_paths:
            # Skip very large files (like lock files)
            try:
                file_size = os.path.getsize(p)
                if file_size > max_file_size:
                    _LOG.debug(f"Skipping large file {p} ({file_size} bytes)")
                    # Still include a note about the file
                    short_path = self._shorten_path(p, project_name)
                    out.append((short_path, f"[File too large: {file_size} bytes. This is likely a lock file.]"))
                    continue
            except Exception:
                pass

            short_path = self._shorten_path(p, project_name)
            content = self._read_text_file(p, max_chars=max_file_size)
            if content:
                out.append((short_path, content))

        return out

    def load_readme_content(
        self,
        readme_path: Optional[str],
        project_name: str,
        max_chars: int = 30_000,
    ) -> Optional[str]:
        """
        Load and extract relevant sections from README file.

        Focuses on sections related to:
        - Installation
        - Building from source
        - Running tests
        - Dependencies/Requirements

        Args:
            readme_path: Path to the README file
            project_name: Name of the project (for logging)
            max_chars: Maximum characters to read

        Returns:
            Extracted README content or None if not found
        """
        if not readme_path or not os.path.isfile(readme_path):
            return None

        content = self._read_text_file(readme_path, max_chars=max_chars * 2)  # Read more, then filter
        if not content:
            return None

        # Keywords that indicate relevant sections
        relevant_keywords = [
            "install", "setup", "getting started", "build", "compile",
            "test", "testing", "development", "contributing", "requirements",
            "dependencies", "prerequisite", "quick start", "usage",
            "from source", "docker", "container", "environment",
        ]

        # Try to extract relevant sections
        lines = content.split('\n')
        relevant_sections = []
        in_relevant_section = False
        current_section = []
        section_header_level = 0

        for line in lines:
            # Check if this is a header (markdown # or underlined)
            is_header = False
            header_level = 0

            if line.startswith('#'):
                is_header = True
                header_level = len(line) - len(line.lstrip('#'))
            elif len(lines) > 1 and lines.index(line) < len(lines) - 1:
                next_line = lines[lines.index(line) + 1] if lines.index(line) + 1 < len(lines) else ""
                if next_line and (set(next_line.strip()) == {'='} or set(next_line.strip()) == {'-'}):
                    is_header = True
                    header_level = 1 if '=' in next_line else 2

            if is_header:
                # Save previous section if it was relevant
                if in_relevant_section and current_section:
                    relevant_sections.append('\n'.join(current_section))

                # Check if new section is relevant
                line_lower = line.lower()
                in_relevant_section = any(kw in line_lower for kw in relevant_keywords)
                current_section = [line] if in_relevant_section else []
                section_header_level = header_level

            elif in_relevant_section:
                current_section.append(line)

        # Don't forget the last section
        if in_relevant_section and current_section:
            relevant_sections.append('\n'.join(current_section))

        if relevant_sections:
            extracted = '\n\n'.join(relevant_sections)
            if len(extracted) > max_chars:
                extracted = extracted[:max_chars] + "\n... [README content truncated]"
            return extracted

        # If no relevant sections found, return the first part of the README
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [README content truncated]"
        return content

    # ---------- content loaders ----------
    def _read_text_file(self, path: str, max_chars: int = 200_000) -> str:
        try:
            with open(path, "r", errors="ignore") as f:
                data = f.read()
            return data[:max_chars]
        except Exception:
            return ""

    def load_workflow_contents(
        self,
        workflow_paths: List[str],
        project_name: str,
        filter_relevant: bool = True,
        model=None,
        use_llm_filter: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        Load workflow file contents.

        Args:
            workflow_paths: List of paths to workflow YAML files
            project_name: Name of the project (for shortening paths in output)
            filter_relevant: If True, extract only relevant parts (build/test commands)
                           and remove CI/CD platform-specific configuration
            model: Optional model instance for LLM-based filtering
            use_llm_filter: If True and model is provided, use LLM to filter and extract
                          only relevant CI/CD files (those related to installing from
                          source and running tests on Linux)

        Returns:
            List of (shortened_path, content) tuples (only relevant files if LLM filtering is enabled)
        """
        out = []
        for p in workflow_paths:
            raw_content = self._read_text_file(p)
            if not raw_content:
                continue

            # Shorten the path for output
            short_path = self._shorten_path(p, project_name)

            # Try LLM filtering first if enabled and model is available
            if use_llm_filter and model is not None:
                llm_filtered = _llm_filter_cicd_file(short_path, raw_content, model)
                if llm_filtered:
                    out.append((short_path, llm_filtered))
                # If LLM returns None, the file is not relevant - skip it
                continue

            # Fall back to heuristic filtering if LLM not available
            if filter_relevant:
                filtered = _extract_relevant_workflow_parts(raw_content)
                # Only include if we extracted something meaningful
                if filtered and filtered.strip():
                    out.append((short_path, filtered))
                else:
                    # Fall back to raw if filtering removed everything
                    out.append((short_path, raw_content))
            else:
                out.append((short_path, raw_content))
        return out

    def load_dockerfile_contents(self, dockerfile_paths: List[str], project_name: str) -> List[Tuple[str, str]]:
        """
        Load Dockerfile contents.

        Args:
            dockerfile_paths: List of paths to Dockerfiles
            project_name: Name of the project (for shortening paths in output)

        Returns:
            List of (shortened_path, content) tuples
        """
        out = []
        for p in dockerfile_paths:
            short_path = self._shorten_path(p, project_name)
            out.append((short_path, self._read_text_file(p)))
        return out

    def load_problems_memory(self, project_path: str) -> Optional[str]:
        # replicates your old "problems_memory/<project_path>" behavior
        p = os.path.join(self.problems_memory_root, project_path)
        if not os.path.exists(p):
            return None
        txt = self._read_text_file(p, max_chars=80_000)
        return txt if txt.strip() else None

    # ---------- repository cloning ----------
    def clone_repo(self, project_path: str, project_url: str) -> bool:
        """
        Clone the repository into workspace_root/project_path if not already present.
        Returns True if the repo is available (either already existed or was cloned successfully).
        """
        target_dir = os.path.join(self.workspace_root, project_path)

        # Check if already cloned (has .git directory)
        if os.path.isdir(os.path.join(target_dir, ".git")):
            _LOG.info(f"Repository already exists at {target_dir}")
            return True

        # Create parent directory if needed
        os.makedirs(self.workspace_root, exist_ok=True)

        # Clone the repository
        _LOG.info(f"Cloning repository {project_url} to {target_dir}...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", project_url, target_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,  # 5 minute timeout for large repos
            )
            if result.returncode == 0:
                _LOG.info(f"Successfully cloned {project_url} to {target_dir}")
                return True
            else:
                _LOG.warning(f"Failed to clone {project_url}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            _LOG.warning(f"Cloning {project_url} timed out after 5 minutes")
            return False
        except Exception as e:
            _LOG.warning(f"Failed to clone {project_url}: {e}")
            return False

    # ---------- web search cache loader ----------
    def load_cached_search_results(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Old code reads:
          search_logs/<project_id>/<project_id>_build_install_from_source.json
        but your “extra search doc code” saves:
          search_logs/<project_id>/<query>.json
        Here we support both conventions.
        """
        folder = os.path.join(self.search_logs_root, project_id)
        if not os.path.isdir(folder):
            return []

        preferred = os.path.join(folder, f"{project_id}_build_install_from_source.json")
        if os.path.exists(preferred):
            try:
                with open(preferred, "r") as f:
                    return json.loads(f.read())
            except Exception:
                return []

        # fallback: load any json in folder (best-effort)
        results: List[Dict[str, Any]] = []
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(folder, fn), "r") as f:
                    results.extend(json.loads(f.read()))
            except Exception:
                continue
        return results

    # ---------- web search functionality ----------
    def _duckduckgo_search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Perform a DuckDuckGo search and return a list of {url, title, snippet} dicts.

        Reliability improvements:
        - requests.Session() + more complete headers (cookies / consistency)
        - Prefer POST to html endpoint (often more stable for scraping)
        - Fallback to GET html endpoint, then lite endpoint
        - More tolerant parsing (BeautifulSoup if installed; regex fallback otherwise)
        - Detect likely bot-check / interstitial HTML and warn clearly
        - Better redirect URL cleanup (uddg=) + HTML entity unescape
        """
        if requests is None:
            _LOG.warning("requests library not available, skipping web search")
            return []

        import time
        import re
        import html as htmllib
        from urllib.parse import quote_plus, unquote, parse_qs, urlparse

        def _extract_actual_url(href: str) -> str:
            """Best-effort cleanup of DDG redirect URLs and normalization."""
            if not href:
                return href

            # Some links may be protocol-relative
            if href.startswith("//"):
                href = "https:" + href

            # DDG often wraps URLs in a redirect containing uddg=
            if "uddg=" in href:
                try:
                    parsed = urlparse(href)
                    actual_url = parse_qs(parsed.query).get("uddg", [href])[0]
                    return unquote(actual_url)
                except Exception:
                    return href

            return href

        def _looks_like_block_or_interstitial(page_text: str) -> bool:
            """Heuristics for bot-check / interstitial pages that return 200 but no results."""
            t = (page_text or "").lower()
            # Keep this list broad; you want to detect, log, and bail quickly.
            indicators = [
                "captcha",
                "unusual traffic",
                "automated requests",
                "sorry",
                "verify you are a human",
                "enable javascript",
                "temporarily unavailable",
            ]
            return any(s in t for s in indicators)

        def _parse_results_bs4(page_text: str) -> List[Dict[str, str]]:
            """Parse results using BeautifulSoup. Raises ImportError if bs4 is not installed."""
            from bs4 import BeautifulSoup  # type: ignore

            soup = BeautifulSoup(page_text, "html.parser")
            results: List[Dict[str, str]] = []

            # Primary selector for html.duckduckgo.com results
            anchors = soup.select("a.result__a")
            if anchors:
                for a in anchors:
                    href = _extract_actual_url(a.get("href", "").strip())
                    title = a.get_text(" ", strip=True)

                    # Try to find a nearby snippet inside the same result container
                    snippet = ""
                    container = a.find_parent(class_=re.compile(r"\bresult\b"))
                    if container is not None:
                        snip_el = container.select_one(".result__snippet")
                        if snip_el is None:
                            # Sometimes snippet class variants appear
                            snip_el = container.select_one("[class*='result__snippet']")
                        if snip_el is not None:
                            snippet = snip_el.get_text(" ", strip=True)

                    if href and title:
                        results.append({"url": href, "title": title, "snippet": snippet})

                    if len(results) >= max_results:
                        break

            # Fallback parsing for lite endpoint (structure differs)
            if not results:
                # Lite pages often have result links without result__a
                lite_candidates = soup.select("a.result-link")
                if not lite_candidates:
                    # Heuristic fallback: keep only outbound links; exclude obvious DDG navigation
                    for a in soup.find_all("a", href=True):
                        h = a.get("href", "")
                        if not h:
                            continue
                        if "duckduckgo.com" in h and "uddg=" not in h:
                            continue
                        if h.startswith("/"):
                            continue
                        text = a.get_text(" ", strip=True)
                        if not text:
                            continue
                        lite_candidates.append(a)

                for a in lite_candidates:
                    href = _extract_actual_url(a.get("href", "").strip())
                    title = a.get_text(" ", strip=True)

                    if href and title:
                        results.append({"url": href, "title": title, "snippet": ""})

                    if len(results) >= max_results:
                        break

            return results

        def _parse_results_regex(page_text: str) -> List[Dict[str, str]]:
            """Regex-based parser with looser matching (still more brittle than bs4)."""
            results: List[Dict[str, str]] = []

            # Match result__a with class containing result__a (not exact match), allow nested tags in title.
            link_pattern = re.compile(
                r'<a[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )

            # Snippets can be <a>, <div>, or <span>, and class may include additional tokens.
            snippet_pattern = re.compile(
                r'<(?:a|div|span)[^>]*class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
                re.IGNORECASE | re.DOTALL,
            )

            links = link_pattern.findall(page_text)
            snippets = snippet_pattern.findall(page_text)

            def _strip_tags(s: str) -> str:
                s = re.sub(r"<[^>]+>", "", s)
                return htmllib.unescape(s).strip()

            for i, (href, title_html) in enumerate(links[:max_results]):
                href = _extract_actual_url(htmllib.unescape(href))
                title = _strip_tags(title_html)
                snippet = _strip_tags(snippets[i]) if i < len(snippets) else ""

                if href and title:
                    results.append({"url": href, "title": title, "snippet": snippet})

            # Lite fallback: pull any anchors that look like outbound results
            if not results:
                generic_anchor = re.compile(
                    r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    re.IGNORECASE | re.DOTALL,
                )
                for href, text_html in generic_anchor.findall(page_text):
                    href = htmllib.unescape(href)
                    if "duckduckgo.com" in href and "uddg=" not in href:
                        continue
                    if href.startswith("/"):
                        continue
                    title = _strip_tags(text_html)
                    href = _extract_actual_url(href)
                    if href and title:
                        results.append({"url": href, "title": title, "snippet": ""})
                    if len(results) >= max_results:
                        break

            return results[:max_results]

        # Use a session for cookies + consistent behavior
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "DNT": "1",
                "Referer": "https://duckduckgo.com/",
            }
        )

        # Try a few approaches: POST html, GET html, GET lite
        attempts = [
            ("POST", "https://html.duckduckgo.com/html/", {"q": query}),
            ("GET", f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", None),
            ("GET", f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}", None),
        ]

        last_error: Exception | None = None

        for method, url, data in attempts:
            try:
                # Small backoff between endpoint fallbacks (helps if rate-limited)
                time.sleep(0.25)

                if method == "POST":
                    resp = session.post(url, data=data, timeout=30)
                else:
                    resp = session.get(url, timeout=30)

                # If rate-limited, try next endpoint
                if resp.status_code in (429, 503):
                    _LOG.warning("DuckDuckGo responded with %s for %s; trying fallback", resp.status_code, url)
                    continue

                resp.raise_for_status()
                page_text = resp.text or ""

                # Quick “blocked/interstitial” detection
                if _looks_like_block_or_interstitial(page_text):
                    _LOG.warning(
                        "DuckDuckGo returned a likely interstitial/bot-check page (status=%s, url=%s). "
                        "This commonly yields 0 parsed results.",
                        resp.status_code,
                        resp.url,
                    )
                    # Try the next endpoint, because lite sometimes works when html is blocked (and vice versa)
                    continue

                # Parse results (prefer bs4 if installed)
                try:
                    results = _parse_results_bs4(page_text)
                except ImportError:
                    results = _parse_results_regex(page_text)

                # Final sanity: if results are empty but page contains markers of results, log for visibility
                if not results and "result__a" in page_text:
                    _LOG.warning(
                        "DDG page appears to contain result markers but parser returned 0 results. "
                        "HTML structure may have changed."
                    )

                if results:
                    _LOG.info("DuckDuckGo search for %r returned %d results", query, len(results))
                    return results[:max_results]

            except Exception as e:
                last_error = e
                _LOG.warning("DuckDuckGo search attempt failed (%s %s): %s", method, url, e)
                continue

        if last_error:
            _LOG.warning("DuckDuckGo search failed after fallbacks: %s", last_error)

        return []

    def _fetch_and_extract_page(self, url: str, max_chars: int = 15000) -> str:
        """
        Fetch a web page and extract its main text content.
        Returns empty string on failure.
        """
        if requests is None:
            return ""

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()

            html = resp.text

            # Simple HTML to text extraction
            import re
            # Remove script and style elements
            html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', ' ', html)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            # Decode HTML entities
            try:
                import html as html_module
                text = html_module.unescape(text)
            except Exception:
                pass

            return text[:max_chars]

        except Exception as e:
            _LOG.debug(f"Failed to fetch {url}: {e}")
            return ""

    def perform_web_search(
        self,
        project_name: str,
        *,
        model=None,
        knowledge_model=None,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search for 'how to install <project_name> on linux and run tests'.
        Fetches top results, extracts content, and optionally uses LLM to analyze/summarize each page.

        Args:
            project_name: Name of the project to search for
            model: Default model (used if knowledge_model is not provided)
            knowledge_model: Separate model for web content analysis (e.g., gpt-5-mini).
                           This model should be up-to-date and knowledgeable about
                           current technologies, build systems, and best practices.
            max_results: Maximum number of search results to fetch

        Returns list of {url, title, snippet, content, analysis} dicts.
        """
        # Use knowledge_model for analysis if provided, otherwise fall back to model
        analysis_model = knowledge_model if knowledge_model is not None else model
        query = f"how to install {project_name} on linux and run tests"
        _LOG.info(f"Performing web search: '{query}'")

        search_results = self._duckduckgo_search(query, max_results=max_results)
        if not search_results:
            _LOG.warning(f"No search results found for '{query}'")
            return []

        enriched_results: List[Dict[str, Any]] = []

        for result in search_results:
            url = result.get("url", "")
            title = result.get("title", "")
            snippet = result.get("snippet", "")

            # Fetch page content
            content = self._fetch_and_extract_page(url)

            # Build result entry
            entry: Dict[str, Any] = {
                "url": url,
                "title": title,
                "snippet": snippet,
                "content": content,
                "analysis": "",
            }

            # If analysis_model is provided, use it to analyze/summarize the page content
            if analysis_model is not None and content:
                try:
                    analysis_prompt = f"""You are helping with a specific task: INSTALLING THE PROJECT '{project_name}' FROM SOURCE CODE AND RUNNING ITS TEST SUITE inside a fresh Linux (Ubuntu) Docker container.

Analyze this web page and extract ONLY information that helps accomplish this goal.

Web page URL: {url}
Web page content:
{content[:10000]}

WHAT WE NEED TO KNOW (extract if present):
1. SYSTEM DEPENDENCIES: What system packages need to be installed (apt-get install ...)?
   - Include specific package names, not just general descriptions

2. LANGUAGE/RUNTIME REQUIREMENTS: What version of Python/Node/Java/etc. is required?
   - Specific version numbers are very helpful

3. PROJECT DEPENDENCIES: How to install the project's dependencies?
   - Exact commands (pip install -e ., npm install, cargo build, etc.)
   - Any special flags or environment variables needed

4. BUILD COMMANDS: How to build/compile the project from source?
   - Exact commands in order
   - Any configuration steps needed first

5. TEST COMMANDS: How to run the test suite? (THIS IS CRUCIAL)
   - The exact command to run tests (pytest, npm test, make test, etc.)
   - Any test-specific setup required
   - How to run a subset of tests if the full suite takes too long

6. COMMON ISSUES: Any known problems when building/testing and their solutions?
   - Missing dependencies that aren't documented
   - Platform-specific issues on Linux
   - Version conflicts and how to resolve them

7. DOCKER/CONTAINER HINTS: Any Dockerfile examples or container setup instructions?

IMPORTANT:
- Focus ONLY on building from source and running tests - ignore deployment, production setup, or usage documentation
- Preserve exact commands as written - don't paraphrase shell commands
- Note any prerequisites or assumptions the documentation makes
- If information seems incomplete or might not work in a fresh container, mention what might be missing

Provide a concise, actionable summary focused on our goal of installing from source and running tests."""

                    resp = analysis_model.query([{"role": "user", "content": analysis_prompt}])
                    entry["analysis"] = resp.get("content", "").strip()
                except Exception as e:
                    _LOG.warning(f"Failed to analyze page {url}: {e}")

            enriched_results.append(entry)
            time.sleep(0.5)  # Be polite to servers

        return enriched_results

    def save_search_results(self, project_id: str, results: List[Dict[str, Any]]) -> None:
        """
        Save search results to cache for future use.
        """
        folder = os.path.join(self.search_logs_root, project_id)
        os.makedirs(folder, exist_ok=True)

        cache_file = os.path.join(folder, f"{project_id}_build_install_from_source.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(results, f, indent=2)
            _LOG.info(f"Saved {len(results)} search results to {cache_file}")
        except Exception as e:
            _LOG.warning(f"Failed to save search results: {e}")

    # ---------- unified summary generator ----------
    def build_unified_summary(
        self,
        *,
        model,
        knowledge_model=None,
        search_workflows_summary_prompt: str,
        project_name: str,
        language: str,
        search_results: List[Dict[str, Any]],
        dockerfile_contents: List[Tuple[str, str]],
        requirement_files: List[Tuple[str, str]],
        readme_content: Optional[str] = None,
        workflow_contents: List[Tuple[str, str]] = None,
        cache_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Ask LLM to consolidate all available information into a structured "prompt section".

        Args:
            model: Default model (used if knowledge_model is not provided)
            knowledge_model: Separate model for summary generation (e.g., gpt-5-mini).
                           This model should be up-to-date and knowledgeable about
                           current technologies, build systems, and best practices.
            search_workflows_summary_prompt: Prompt template for summary generation
            project_name: Name of the project
            language: Primary programming language of the project
            search_results: Web search results with analysis
            dockerfile_contents: List of (path, content) tuples for Dockerfiles
            requirement_files: List of (path, content) tuples for requirement/dependency files
            readme_content: Extracted README content (installation/build instructions)
            workflow_contents: List of (path, content) tuples for CI/CD workflow files
            cache_path: Optional path to cache the summary
        """
        # Use knowledge_model for summary if provided, otherwise fall back to model
        summary_model = knowledge_model if knowledge_model is not None else model

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    txt = f.read()
                return txt if txt.strip() else None
            except Exception:
                pass

        # Assemble evidence from all sources
        pages = []
        for r in search_results[:10]:
            url = r.get("url", "")
            analysis = r.get("analysis", "")
            if not analysis:
                continue
            pages.append({"url": url, "content": analysis[:12000]})

        docker_bits = []
        for path, content in dockerfile_contents[:5]:
            if content.strip():
                docker_bits.append({"path": path, "content": content[:8000]})

        # Add requirement files (limit size to prevent token overflow)
        req_files = []
        for path, content in (requirement_files or [])[:10]:
            if content.strip():
                req_files.append({"path": path, "content": content[:15000]})

        # Add workflow files (CI/CD)
        workflow_bits = []
        for path, content in (workflow_contents or [])[:5]:
            if content.strip():
                workflow_bits.append({"path": path, "content": content[:10000]})

        # Check if we have any meaningful content
        has_content = pages or docker_bits or req_files or workflow_bits or readme_content
        if not has_content:
            return None

        query = search_workflows_summary_prompt.format(project_name)

        payload = {
            "project": project_name,
            "language": language,
            "web_pages": pages,
            "dockerfiles": docker_bits,
            "requirement_files": req_files,
            "workflow_files": workflow_bits,
            "readme_content": (readme_content[:20000] if readme_content else ""),
        }

        resp = summary_model.query([
            {"role": "user", "content": query + "\n\nEvidence (JSON):\n" + json.dumps(payload, indent=2)}
        ])
        summary = resp.get("content", "").strip()
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            try:
                with open(cache_path, "w") as f:
                    f.write(summary)
            except Exception:
                pass
        return summary if summary else None

    # ---------- main entry ----------
    def build_repo_context(
        self,
        *,
        model,
        knowledge_model=None,
        project_path: str,
        project_url: str,
        language: str,
        search_workflows_summary_prompt: str,
        unified_summary_cache_root: str = "search_logs_unified",
        perform_web_search_if_missing: bool = True,
    ) -> RepoContext:
        """
        Build the repository context including workflows, dockerfiles, requirement files,
        README, and web search results.

        Args:
            model: Main model used for CI/CD file filtering
            knowledge_model: Separate model for web search analysis and unified summary
                           generation (e.g., gpt-5-mini). This model should be up-to-date
                           and knowledgeable about current technologies and best practices.
                           If not provided, falls back to the main model.
            project_path: Path/name of the project
            project_url: Git URL of the project
            language: Programming language of the project
            search_workflows_summary_prompt: Prompt template for summary generation
            unified_summary_cache_root: Root directory for caching unified summaries
            perform_web_search_if_missing: Whether to perform web search if no cached results
        """
        # Clone the repository first so the agent can explore it before creating a container
        local_repo_available = self.clone_repo(project_path, project_url)

        # Find and load workflow files (CI/CD)
        workflows = self.find_workflows(project_path, filter_by_keywords=False)
        workflow_contents = self.load_workflow_contents(
            workflows, project_name=project_path, model=model, use_llm_filter=True
        )

        # Find and load Dockerfiles
        dockerfiles = self.find_dockerfiles(project_path)
        dockerfile_contents = self.load_dockerfile_contents(dockerfiles, project_name=project_path)

        # Find and load requirement/dependency files (language-specific heuristic)
        _LOG.info(f"Searching for requirement files for language: {language}")
        requirement_paths = self.find_requirement_files(project_path, language)
        requirement_files = self.load_requirement_files(requirement_paths, project_name=project_path)
        _LOG.info(f"Found {len(requirement_files)} requirement/dependency files")

        # Find and load README content
        readme_path = self.find_readme(project_path)
        readme_content = self.load_readme_content(readme_path, project_name=project_path)
        if readme_content:
            _LOG.info(f"Loaded README content ({len(readme_content)} chars)")
        else:
            _LOG.info("No README file found or content extracted")

        # Try to load cached search results first
        search_results = self.load_cached_search_results(project_path)

        # If no cached results and web search is enabled, perform web search
        # Use knowledge_model for web content analysis (more up-to-date knowledge)
        if not search_results and perform_web_search_if_missing:
            _LOG.info(f"No cached search results for '{project_path}', performing web search...")
            search_results = self.perform_web_search(
                project_name=project_path,
                model=model,
                knowledge_model=knowledge_model,
                max_results=5,
            )
            # Save results for future use
            if search_results:
                self.save_search_results(project_path, search_results)

        problems_memory = self.load_problems_memory(project_path)

        # Use knowledge_model for unified summary (more up-to-date knowledge)
        # Pass all collected information to create a comprehensive summary
        cache_path = os.path.join(unified_summary_cache_root, project_path, "unified_summary.txt")
        unified_summary = self.build_unified_summary(
            model=model,
            knowledge_model=knowledge_model,
            search_workflows_summary_prompt=search_workflows_summary_prompt,
            project_name=project_path,
            language=language,
            search_results=search_results,
            dockerfile_contents=dockerfile_contents,
            requirement_files=requirement_files,
            readme_content=readme_content,
            workflow_contents=workflow_contents,
            cache_path=cache_path,
        )

        return RepoContext(
            project_path=project_path,
            project_url=project_url,
            language=language,
            workflows=workflows,
            workflow_contents=workflow_contents,
            dockerfiles=dockerfiles,
            dockerfile_contents=dockerfile_contents,
            search_results=search_results,
            requirement_files=requirement_files,
            readme_content=readme_content,
            unified_summary=unified_summary,
            problems_memory=problems_memory,
            local_repo_available=local_repo_available,
        )
