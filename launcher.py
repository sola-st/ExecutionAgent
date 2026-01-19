#!/usr/bin/env python3
"""
Execution Agent Launcher

A comprehensive launcher for running the execution agent on multiple projects.
Supports launching all, one, or selected projects with various configuration options.

Usage:
    python launcher.py --help                     # Show help
    python launcher.py --list                     # List all available projects
    python launcher.py --run all                  # Run all projects
    python launcher.py --run scipy                # Run single project
    python launcher.py --run scipy,pandas,numpy   # Run multiple projects
    python launcher.py --run python               # Run all Python projects
    python launcher.py --run java                 # Run all Java projects
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import shutil
import time
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# Project Definitions
# =============================================================================

@dataclass
class Project:
    """Represents a project to be tested."""
    name: str
    url: str
    language: str
    image_tag: str

    @property
    def safe_name(self) -> str:
        """Return a filesystem-safe name."""
        return self.name.lower().replace("-", "_").replace(".", "_")


# All 50 projects
PROJECTS: List[Project] = [
    # Python Projects (14)
    Project("pandas", "https://github.com/pandas-dev/pandas", "Python", "pandas_image:ExecutionAgent"),
    Project("scikit-learn", "https://github.com/scikit-learn/scikit-learn", "Python", "scikit_image:ExecutionAgent"),
    Project("scipy", "https://github.com/scipy/scipy", "Python", "scipy_image:ExecutionAgent"),
    Project("numpy", "https://github.com/numpy/numpy", "Python", "numpy_image:ExecutionAgent"),
    Project("django", "https://github.com/django/django", "Python", "django_image:ExecutionAgent"),
    Project("langchain", "https://github.com/langchain-ai/langchain", "Python", "langchain_image:ExecutionAgent"),
    Project("pytest", "https://github.com/pytest-dev/pytest", "Python", "pytest_image:ExecutionAgent"),
    Project("cpython", "https://github.com/python/cpython", "Python", "cpython_image:ExecutionAgent"),
    Project("ansible", "https://github.com/ansible/ansible", "Python", "ansible_image:ExecutionAgent"),
    Project("flask", "https://github.com/pallets/flask", "Python", "flask_image:ExecutionAgent"),
    Project("keras", "https://github.com/keras-team/keras", "Python", "keras_image:ExecutionAgent"),
    Project("tensorflow", "https://github.com/tensorflow/tensorflow", "C++", "tensorflow_image:ExecutionAgent"),

    # Java Projects (10)
    Project("flink", "https://github.com/apache/flink", "Java", "flink_image:ExecutionAgent"),
    Project("commons-csv", "https://github.com/apache/commons-csv", "Java", "commonscsv_image:ExecutionAgent"),
    Project("dubbo", "https://github.com/apache/dubbo", "Java", "dubbo_image:ExecutionAgent"),
    Project("mybatis-3", "https://github.com/mybatis/mybatis-3", "Java", "mybatis_image:ExecutionAgent"),
    Project("rocketmq", "https://github.com/apache/rocketmq", "Java", "rocketmq_image:ExecutionAgent"),
    Project("guava", "https://github.com/google/guava", "Java", "guava_image:ExecutionAgent"),
    Project("RxJava", "https://github.com/ReactiveX/RxJava", "Java", "rxjava_image:ExecutionAgent"),
    Project("Activiti", "https://github.com/Activiti/Activiti", "Java", "activiti_image:ExecutionAgent"),
    Project("spring-security", "https://github.com/spring-projects/spring-security", "Java", "spring_image:ExecutionAgent"),

    # JavaScript Projects (12)
    Project("react", "https://github.com/facebook/react", "Javascript", "reactjs_image:ExecutionAgent"),
    Project("vue", "https://github.com/vuejs/vue", "Javascript", "vuejs_image:ExecutionAgent"),
    Project("bootstrap", "https://github.com/twbs/bootstrap", "Javascript", "bootstrap_image:ExecutionAgent"),
    Project("node", "https://github.com/nodejs/node", "Javascript", "nodejs_image:ExecutionAgent"),
    Project("axios", "https://github.com/axios/axios", "Javascript", "axios_image:ExecutionAgent"),
    Project("typescript", "https://github.com/microsoft/TypeScript", "Javascript", "typescript_image:ExecutionAgent"),
    Project("deno", "https://github.com/denoland/deno", "Javascript", "deno_image:ExecutionAgent"),
    Project("mermaid", "https://github.com/mermaid-js/mermaid", "Javascript", "mermaid_image:ExecutionAgent"),
    Project("nest", "https://github.com/nestjs/nest", "Javascript", "nest_image:ExecutionAgent"),
    Project("webpack", "https://github.com/webpack/webpack", "Javascript", "webpack_image:ExecutionAgent"),
    Project("express", "https://github.com/expressjs/express", "Javascript", "express_image:ExecutionAgent"),
    Project("Chart.js", "https://github.com/chartjs/Chart.js", "Javascript", "chart_image:ExecutionAgent"),

    # C Projects (10)
    Project("git", "https://github.com/git/git", "C", "git_image:ExecutionAgent"),
    Project("mpv", "https://github.com/mpv-player/mpv", "C", "mpv_image:ExecutionAgent"),
    Project("FreeRTOS-Kernel", "https://github.com/FreeRTOS/FreeRTOS-Kernel", "C", "freertos_image:ExecutionAgent"),
    Project("ccache", "https://github.com/ccache/ccache", "C", "ccache_image:ExecutionAgent"),
    Project("msgpack-c", "https://github.com/msgpack/msgpack-c", "C", "msgpack_image:ExecutionAgent"),
    Project("openvpn", "https://github.com/OpenVPN/openvpn", "C", "openvpn_image:ExecutionAgent"),
    Project("distcc", "https://github.com/distcc/distcc", "C", "distcc_image:ExecutionAgent"),
    Project("xrdp", "https://github.com/neutrinolabs/xrdp", "C", "xrdp_image:ExecutionAgent"),
    Project("libevent", "https://github.com/libevent/libevent", "C", "libevent_image:ExecutionAgent"),
    Project("json-c", "https://github.com/json-c/json-c", "C", "jsonc_image:ExecutionAgent"),

    # C++ Projects (8)
    Project("react-native", "https://github.com/facebook/react-native", "C++", "react_image:ExecutionAgent"),
    Project("opencv", "https://github.com/opencv/opencv", "C++", "opencv_image:ExecutionAgent"),
    Project("imgui_test_engine", "https://github.com/ocornut/imgui_test_engine", "C++", "imgui_image:ExecutionAgent"),
    Project("folly", "https://github.com/facebook/folly", "C++", "folly_image:ExecutionAgent"),
    Project("xgboost", "https://github.com/dmlc/xgboost", "C++", "xgboost_image:ExecutionAgent"),
    Project("webview", "https://github.com/webview/webview", "C++", "webview_image:ExecutionAgent"),
    Project("json", "https://github.com/nlohmann/json", "C++", "json_image:ExecutionAgent"),
]

# Create lookup dictionaries
PROJECTS_BY_NAME: Dict[str, Project] = {p.name.lower(): p for p in PROJECTS}
PROJECTS_BY_LANGUAGE: Dict[str, List[Project]] = {}
for p in PROJECTS:
    lang = p.language.lower()
    if lang not in PROJECTS_BY_LANGUAGE:
        PROJECTS_BY_LANGUAGE[lang] = []
    PROJECTS_BY_LANGUAGE[lang].append(p)


# =============================================================================
# Utility Functions
# =============================================================================

def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def create_metadata_file(project: Project, output_dir: Path, budget: int = 40) -> Path:
    """Create metadata JSON file for a project."""
    metadata = {
        "project_path": project.safe_name,  # Used by main.py as the project identifier
        "project_name": project.name,       # Human-readable name
        "project_url": project.url,
        "language": project.language,
        "image_tag": project.image_tag,
        "budget": budget,                   # Step limit per attempt
        "created_at": datetime.now().isoformat(),
    }

    metadata_path = output_dir / f"meta_{project.safe_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def print_header(text: str, char: str = "=", width: int = 80) -> None:
    """Print a formatted header."""
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}\n")


def print_project_table(projects: List[Project]) -> None:
    """Print a formatted table of projects."""
    print(f"{'#':<4} {'Name':<25} {'Language':<12} {'URL':<50}")
    print("-" * 95)
    for i, p in enumerate(projects, 1):
        url_short = p.url[:47] + "..." if len(p.url) > 50 else p.url
        print(f"{i:<4} {p.name:<25} {p.language:<12} {url_short:<50}")


def resolve_project_selection(selection: str) -> List[Project]:
    """
    Resolve a project selection string to a list of projects.

    Supports:
    - 'all': All projects
    - Language name (python, java, c, c++, javascript): All projects of that language
    - Project name: Single project
    - Comma-separated list: Multiple projects
    """
    selection = selection.strip().lower()

    if selection == "all":
        return PROJECTS.copy()

    # Check if it's a language
    if selection in PROJECTS_BY_LANGUAGE:
        return PROJECTS_BY_LANGUAGE[selection].copy()

    # Handle c++ specially
    if selection in ("c++", "cpp"):
        return PROJECTS_BY_LANGUAGE.get("c++", []).copy()

    # Check for comma-separated list
    if "," in selection:
        names = [n.strip() for n in selection.split(",")]
        projects = []
        for name in names:
            # Check if it's a language
            if name in PROJECTS_BY_LANGUAGE:
                projects.extend(PROJECTS_BY_LANGUAGE[name])
            elif name in ("c++", "cpp"):
                projects.extend(PROJECTS_BY_LANGUAGE.get("c++", []))
            elif name in PROJECTS_BY_NAME:
                projects.append(PROJECTS_BY_NAME[name])
            else:
                print(f"Warning: Unknown project or language '{name}', skipping...")
        return projects

    # Single project
    if selection in PROJECTS_BY_NAME:
        return [PROJECTS_BY_NAME[selection]]

    # Fuzzy match
    matches = [p for p in PROJECTS if selection in p.name.lower()]
    if matches:
        return matches

    return []


# =============================================================================
# Runner Class
# =============================================================================

class AgentRunner:
    """Manages running the execution agent on projects."""

    def __init__(
        self,
        workspace_root: Path,
        model: str = "gpt-4o-mini",
        step_limit: int = 40,
        max_retries: int = 2,
        parallel: int = 1,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.workspace_root = workspace_root
        self.model = model
        self.step_limit = step_limit
        self.max_retries = max_retries
        self.parallel = parallel
        self.dry_run = dry_run
        self.verbose = verbose
        self.script_dir = get_script_dir()
        self.results: Dict[str, dict] = {}
        self._stop_event = threading.Event()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def handler(signum, frame):
            print("\n\nReceived interrupt signal. Stopping...")
            self._stop_event.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def run_single_project(self, project: Project) -> dict:
        """Run the agent on a single project."""
        if self._stop_event.is_set():
            return {"status": "cancelled", "project": project.name}

        start_time = datetime.now()
        result = {
            "project": project.name,
            "language": project.language,
            "url": project.url,
            "start_time": start_time.isoformat(),
            "status": "unknown",
            "exit_code": -1,
            "duration_seconds": 0,
        }

        # Create metadata file
        metadata_dir = self.workspace_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = create_metadata_file(project, metadata_dir, budget=self.step_limit)

        # Build command
        cmd = [
            sys.executable,
            "-m", "execution_agent.main",
            "--experiment-file", str(metadata_path),
            "--workspace-root", str(self.workspace_root),
            "--model", self.model,
            "--max-retries", str(self.max_retries),
        ]

        print(f"\n{'=' * 60}")
        print(f"Starting: {project.name} ({project.language})")
        print(f"URL: {project.url}")
        print(f"{'=' * 60}")

        if self.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            result["status"] = "dry_run"
            result["exit_code"] = 0
            return result

        # Create log file for capturing output (avoids pipe buffer deadlock)
        log_dir = self.workspace_root / "_run_logs" / project.safe_name
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"launcher_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
        log_file_handle = None

        try:
            if self.verbose:
                # In verbose mode, output goes directly to terminal
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.script_dir),
                    stdout=None,
                    stderr=None,
                )
            else:
                # Redirect stdout/stderr to a file instead of using PIPE
                # This avoids the pipe buffer deadlock issue
                log_file_handle = open(log_file, "w")
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.script_dir),
                    stdout=log_file_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                result["log_file"] = str(log_file)

            # Wait for completion or stop signal
            while process.poll() is None:
                if self._stop_event.is_set():
                    print(f"Terminating {project.name}...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    result["status"] = "cancelled"
                    result["exit_code"] = -1
                    break
                time.sleep(1)

            if not self._stop_event.is_set():
                result["exit_code"] = process.returncode
                result["status"] = "success" if process.returncode == 0 else "failed"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"Error running {project.name}: {e}")
        finally:
            # Ensure log file handle is closed
            if log_file_handle is not None:
                log_file_handle.close()

        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()

        status_emoji = "✅" if result["status"] == "success" else "❌" if result["status"] == "failed" else "⚠️"
        print(f"\n{status_emoji} {project.name}: {result['status']} (took {result['duration_seconds']:.1f}s)")

        return result

    def run_projects(self, projects: List[Project]) -> Dict[str, dict]:
        """Run the agent on multiple projects."""
        self.setup_signal_handlers()

        print_header(f"Running {len(projects)} project(s)")
        print(f"Model: {self.model}")
        print(f"Step limit: {self.step_limit}")
        print(f"Max retries: {self.max_retries}")
        print(f"Parallel: {self.parallel}")
        print(f"Workspace: {self.workspace_root}")
        print()

        if self.parallel > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {executor.submit(self.run_single_project, p): p for p in projects}

                for future in as_completed(futures):
                    project = futures[future]
                    try:
                        result = future.result()
                        self.results[project.name] = result
                    except Exception as e:
                        self.results[project.name] = {
                            "project": project.name,
                            "status": "error",
                            "error": str(e),
                        }

                    if self._stop_event.is_set():
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
        else:
            # Sequential execution
            for project in projects:
                if self._stop_event.is_set():
                    break
                result = self.run_single_project(project)
                self.results[project.name] = result

        return self.results

    def print_summary(self) -> None:
        """Print a summary of all results."""
        if not self.results:
            return

        print_header("Execution Summary")

        success = [r for r in self.results.values() if r.get("status") == "success"]
        failed = [r for r in self.results.values() if r.get("status") == "failed"]
        cancelled = [r for r in self.results.values() if r.get("status") == "cancelled"]
        errors = [r for r in self.results.values() if r.get("status") == "error"]

        print(f"Total: {len(self.results)}")
        print(f"  ✅ Success: {len(success)}")
        print(f"  ❌ Failed: {len(failed)}")
        print(f"  ⚠️  Cancelled: {len(cancelled)}")
        print(f"  💥 Errors: {len(errors)}")
        print()

        if failed:
            print("Failed projects:")
            for r in failed:
                print(f"  - {r['project']}: exit code {r.get('exit_code', 'N/A')}")
            print()

        if errors:
            print("Error projects:")
            for r in errors:
                print(f"  - {r['project']}: {r.get('error', 'Unknown error')}")
            print()

        total_duration = sum(r.get("duration_seconds", 0) for r in self.results.values())
        print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")

        # Save summary to file
        summary_path = self.workspace_root / "launcher_summary.json"
        summary_path.write_text(json.dumps(self.results, indent=2))
        print(f"\nSummary saved to: {summary_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Execution Agent Launcher - Run the agent on multiple projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                        # List all available projects
  %(prog)s --list --language python      # List only Python projects
  %(prog)s --run all                     # Run all 50 projects
  %(prog)s --run scipy                   # Run single project
  %(prog)s --run scipy,pandas,numpy      # Run multiple projects
  %(prog)s --run python                  # Run all Python projects
  %(prog)s --run java,c++                # Run all Java and C++ projects
  %(prog)s --run scipy --model gpt-4o    # Use specific model
  %(prog)s --run all --parallel 4        # Run 4 projects in parallel
  %(prog)s --run all --dry-run           # Show what would be run
        """,
    )

    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available projects",
    )
    action_group.add_argument(
        "--run", "-r",
        type=str,
        metavar="SELECTION",
        help="Run agent on selected projects (all, language name, project name, or comma-separated list)",
    )
    action_group.add_argument(
        "--create-meta",
        type=str,
        metavar="SELECTION",
        help="Create metadata files only (without running)",
    )

    # Filter arguments
    parser.add_argument(
        "--language", "-L",
        type=str,
        choices=["python", "java", "javascript", "c", "c++"],
        help="Filter by language (for --list)",
    )

    # Agent configuration
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--step-limit", "-s",
        type=int,
        default=40,
        help="Step limit per attempt (default: 40)",
    )
    parser.add_argument(
        "--max-retries", "-R",
        type=int,
        default=2,
        help="Maximum retries after budget exhaustion (default: 2)",
    )

    # Execution options
    parser.add_argument(
        "--workspace-root", "-w",
        type=str,
        default=None,
        help="Workspace root directory (default: ./execution_agent_workspace)",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of projects to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be run without actually running",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output from agent",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    script_dir = get_script_dir()
    workspace_root = Path(args.workspace_root) if args.workspace_root else script_dir / "execution_agent_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    # Handle --list
    if args.list:
        print_header("Available Projects")

        if args.language:
            lang = args.language.lower()
            if lang == "c++":
                lang = "c++"
            projects = PROJECTS_BY_LANGUAGE.get(lang, [])
            if not projects:
                print(f"No projects found for language: {args.language}")
                return 1
            print(f"Language: {args.language} ({len(projects)} projects)\n")
        else:
            projects = PROJECTS
            print(f"All projects ({len(projects)} total)\n")

            # Show breakdown by language
            print("By language:")
            for lang, projs in sorted(PROJECTS_BY_LANGUAGE.items()):
                print(f"  {lang}: {len(projs)}")
            print()

        print_project_table(projects)
        return 0

    # Handle --create-meta
    if args.create_meta:
        projects = resolve_project_selection(args.create_meta)
        if not projects:
            print(f"No projects found matching: {args.create_meta}")
            return 1

        print_header(f"Creating metadata for {len(projects)} project(s)")

        metadata_dir = workspace_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for project in projects:
            meta_path = create_metadata_file(project, metadata_dir)
            print(f"Created: {meta_path}")

        print(f"\nMetadata files created in: {metadata_dir}")
        return 0

    # Handle --run
    if args.run:
        projects = resolve_project_selection(args.run)
        if not projects:
            print(f"No projects found matching: {args.run}")
            print("\nUse --list to see available projects")
            return 1

        print(f"Selected {len(projects)} project(s):")
        for p in projects:
            print(f"  - {p.name} ({p.language})")

        runner = AgentRunner(
            workspace_root=workspace_root,
            model=args.model,
            step_limit=args.step_limit,
            max_retries=args.max_retries,
            parallel=args.parallel,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        runner.run_projects(projects)
        runner.print_summary()

        # Return non-zero if any failed
        failed = [r for r in runner.results.values() if r.get("status") in ("failed", "error")]
        return 1 if failed else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
