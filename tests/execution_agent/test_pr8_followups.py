"""
Tests for the follow-ups to PR #8 (Dockerfile build args + commit pinning).

Run from the repository root with:
    pytest tests/execution_agent --confcutdir=tests/execution_agent -p no:cacheprovider

(--confcutdir keeps pytest from loading tests/conftest.py, which belongs to the
inherited mini-swe-agent suite and imports packages this project does not install.)
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from execution_agent.agent import ExecutionAgent  # noqa: E402
from execution_agent.context import ContextBuilder, RepoContext  # noqa: E402
from execution_agent import exit_artifacts  # noqa: E402


# --------------------------------------------------------------------------- helpers
def _ctx(**kw) -> RepoContext:
    base = dict(project_path="proj", project_url="https://x/y.git", language="Python",
                workflows=[], workflow_contents=[], dockerfiles=[], dockerfile_contents=[],
                search_results=[])
    base.update(kw)
    return RepoContext(**base)


def _make_agent() -> ExecutionAgent:
    return ExecutionAgent(
        model=None,
        env=None,
        tool_registry=None,
        cycle_instruction="CYCLE-INSTRUCTION",
        summarize_cycle="",
        remove_progress_bars_prompt="",
        search_workflows_summary_prompt="",
    )


SUMMARY_WITH_SHALLOW_CLONE = (
    "## 2. DOCKERFILE TEMPLATE\n"
    "```dockerfile\n"
    "FROM ubuntu:24.04\n"
    "ARG REPO_URL\n"
    'RUN test -n "$REPO_URL" && git clone --depth 1 "$REPO_URL" "$PROJECT_DIR"\n'
    "```\n"
)


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
    }
    res = subprocess.run(["git", "-C", str(cwd), *args], check=True, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return res.stdout.strip()


@pytest.fixture
def remote(tmp_path: Path):
    """A bare 'origin' with two commits on main and one more on a 'feature' branch."""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    (work / "a.txt").write_text("1\n")
    _git(work, "add", "."); _git(work, "commit", "-q", "-m", "c1")
    (work / "a.txt").write_text("2\n")
    _git(work, "add", "."); _git(work, "commit", "-q", "-m", "c2")
    main_sha = _git(work, "rev-parse", "HEAD")
    _git(work, "checkout", "-q", "-b", "feature")
    (work / "b.txt").write_text("f\n")
    _git(work, "add", "."); _git(work, "commit", "-q", "-m", "feature-only")
    feature_sha = _git(work, "rev-parse", "HEAD")
    _git(work, "checkout", "-q", "main")

    bare = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(work), str(bare))
    # file:// rather than a bare path: git ignores --depth for local-path clones,
    # and the shallow-clone cases below are the whole point.
    return {"work": work, "url": bare.as_uri(), "main": main_sha, "feature": feature_sha}


# =========================================================================== Fix 1
class TestPinnedCommitPromptOrdering:
    def test_commit_section_comes_after_summary_template(self):
        agent = _make_agent()
        agent.repo_context = _ctx(unified_summary=SUMMARY_WITH_SHALLOW_CLONE, commit="deadbeefcafe",
        )
        prompt = agent._build_instance_prompt("task")

        shallow_at = prompt.index("git clone --depth 1")
        required_at = prompt.index("REQUIRED COMMIT: deadbeefcafe")
        assert required_at > shallow_at, "pinned-commit block must follow the summary template"
        assert "OVERRIDES the clone step shown in the setup summary above" in prompt
        assert 'checkout --detach "$COMMIT_SHA"' in prompt
        # a short pointer still appears in the header so the model sees it early
        assert prompt.index("Required commit: deadbeefcafe") < shallow_at

    def test_no_commit_means_no_section(self):
        agent = _make_agent()
        agent.repo_context = _ctx(project_url="u", unified_summary=SUMMARY_WITH_SHALLOW_CLONE,
        )
        prompt = agent._build_instance_prompt("task")
        assert "REQUIRED COMMIT" not in prompt
        assert "Required commit" not in prompt

    def test_agent_attr_fallback_when_context_lacks_commit(self):
        agent = _make_agent()
        agent.repo_context = _ctx(project_path="p", project_url="u", language="C")
        agent.commit = "abc123"
        assert "REQUIRED COMMIT: abc123" in agent._build_instance_prompt("t")


class TestUnifiedSummaryIsCommitAware:
    class _CaptureModel:
        def __init__(self):
            self.queries = []

        def query(self, messages):
            self.queries.append(messages[0]["content"])
            return {"content": "SUMMARY"}

    def _run(self, commit: str) -> str:
        model = self._CaptureModel()
        cb = ContextBuilder(workspace_root="unused")
        out = cb.build_unified_summary(
            model=model, search_workflows_summary_prompt="Summarize {}.",
            project_name="proj", language="Python", search_results=[],
            dockerfile_contents=[], requirement_files=[], readme_content="README",
            workflow_contents=[], cache_path=None, commit=commit,
        )
        assert out == "SUMMARY"
        assert len(model.queries) == 1
        return model.queries[0]

    def test_pinned_commit_instructs_full_clone(self):
        q = self._run("0123abcd")
        assert "PINNED COMMIT" in q and "0123abcd" in q
        assert 'checkout --detach "$COMMIT_SHA"' in q
        assert "Do not use `--depth 1`" in q

    def test_unpinned_query_unchanged(self):
        q = self._run("")
        assert "PINNED COMMIT" not in q
        assert q.startswith("Summarize proj.")

    def test_pinned_template_is_rendered_not_contradicted(self):
        from execution_agent import shared_utils as su
        tpl = 'WORKDIR /app\n\nARG REPO_URL\nARG PROJECT_DIR=project\n\nRUN test -n "$REPO_URL" && git clone --depth 1 "$REPO_URL" "$PROJECT_DIR"\n\nWORKDIR /app/${PROJECT_DIR}\n'
        out = su.render_pinned_template(tpl, "abc123")
        assert "--depth 1" not in out and 'ARG COMMIT_SHA=abc123' in out and 'checkout --detach "$COMMIT_SHA"' in out
        assert su.render_pinned_template(tpl, "") == tpl
        q = self._run("0123abcd")
        assert "--depth 1" not in q.split("IMPORTANT - PINNED COMMIT")[0]   # template part is clean

    def test_failed_pinned_checkout_does_not_scan_or_cache(self, tmp_path, remote):
        """Review finding: a transient checkout failure summarised the wrong tree and
        cached it under the commit key, poisoning every later run for that commit."""
        ws = tmp_path / "ws"; cache = tmp_path / "cache"
        cb = ContextBuilder(workspace_root=str(ws), search_logs_root=str(tmp_path / "sl"),
                            problems_memory_root=str(tmp_path / "pm"))
        model = self._CaptureModel()
        ctx = cb.build_repo_context(model=model, project_path="proj", project_url=remote["url"], language="Python",
                                    search_workflows_summary_prompt="S {}", commit="0" * 40,
                                    unified_summary_cache_root=str(cache), perform_web_search_if_missing=False)
        assert ctx.local_repo_available is False
        assert ctx.unified_summary is None and model.queries == []
        assert not cache.exists() or not list(cache.rglob("unified_summary*"))

    def test_cache_path_is_keyed_by_commit(self):
        """A pinned summary embeds the SHA, so it must not be served to an unpinned
        run of the same project_path (observed live: --commit on a built-in name)."""
        f = ContextBuilder.unified_summary_cache_path
        assert f("root", "proj") == os.path.join("root", "proj", "unified_summary.txt")
        pinned = f("root", "proj", "7dfe9a8af6ffb49f1b2d1790ca1ef5f025709e40")
        assert pinned == os.path.join("root", "proj", "unified_summary_7dfe9a8af6ff.txt")
        assert f("root", "proj", "0" * 40) != pinned

    def test_pinned_summary_read_from_commit_cache(self, tmp_path):
        root = tmp_path / "cache"
        (root / "proj").mkdir(parents=True)
        (root / "proj" / "unified_summary.txt").write_text("UNPINNED")
        (root / "proj" / "unified_summary_abcdef012345.txt").write_text("PINNED")
        cb = ContextBuilder(workspace_root="unused")
        model = self._CaptureModel()
        common = dict(model=model, search_workflows_summary_prompt="S {}", project_name="proj",
                      language="Python", search_results=[], dockerfile_contents=[],
                      requirement_files=[], readme_content="R", workflow_contents=[])
        assert cb.build_unified_summary(cache_path=cb.unified_summary_cache_path(str(root), "proj"), **common) == "UNPINNED"
        assert cb.build_unified_summary(cache_path=cb.unified_summary_cache_path(str(root), "proj", "abcdef0123456789"),
                                        commit="abcdef0123456789", **common) == "PINNED"
        assert model.queries == []   # both served from cache, no model call


# =========================================================================== Fix 2
class TestBaseDockerfileReset:
    def test_agent_defaults_are_empty(self):
        agent = _make_agent()
        assert agent.base_dockerfile == ""

    def test_retry_reset_clears_base_dockerfile(self):
        """Guard on main.py's inline reset block: it must clear the field that
        exit_artifacts now treats as authoritative."""
        src = (REPO_ROOT / "src/execution_agent/main.py").read_text()
        start = src.index("# Reset volatile state")
        end = src.index("# Restore preserved state", start)
        block = src[start:end]
        assert re.search(r'agent\.base_dockerfile\s*=\s*""', block)
        assert re.search(r'agent\.written_files\s*=\s*\[\]', block)  # sanity: right block

    def test_exit_artifact_prefers_built_dockerfile(self):
        class A: pass
        a = A()
        a.written_files = [("Dockerfile", "local", "/x", "FROM a\n# broken"),
                           ("Dockerfile", "local", "/x", "FROM a\n# built")]
        a.base_dockerfile = "FROM a\n# built"
        assert exit_artifacts._extract_dockerfile_content(a).endswith("# built")
        a.base_dockerfile = ""            # e.g. after the retry reset, before a new build
        assert exit_artifacts._extract_dockerfile_content(a).endswith("# built")  # newest written


# =========================================================================== Fix 3
class TestCloneRepoCommitRecovery:
    def _head(self, path: Path) -> str:
        return _git(path, "rev-parse", "HEAD")

    def test_fresh_pinned_clone_is_full_depth_and_checked_out(self, tmp_path, remote):
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"], commit=remote["feature"]) is True
        assert self._head(ws / "proj") == remote["feature"]
        assert not (ws / "proj/.git/shallow").exists()

    def test_unpinned_clone_stays_shallow(self, tmp_path, remote):
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"]) is True
        assert (ws / "proj/.git/shallow").exists()
        assert self._head(ws / "proj") == remote["main"]

    def test_existing_shallow_clone_recovers_commit_on_other_branch(self, tmp_path, remote):
        """The case the PR could not handle: an earlier unpinned run left a
        `--depth 1` single-branch clone, and the pinned commit is on another branch."""
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"]) is True          # shallow, main only
        assert (ws / "proj/.git/shallow").exists()

        assert cb.clone_repo("proj", remote["url"], commit=remote["feature"]) is True
        assert self._head(ws / "proj") == remote["feature"]

    def test_existing_full_clone_recovers_commit_pushed_later(self, tmp_path, remote):
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"], commit=remote["main"]) is True

        # remote moves on after our clone
        work = remote["work"]
        (work / "a.txt").write_text("3\n")
        _git(work, "add", "."); _git(work, "commit", "-q", "-m", "c3")
        new_sha = _git(work, "rev-parse", "HEAD")
        _git(work, "push", "-q", remote["url"], "main")

        assert cb.clone_repo("proj", remote["url"], commit=new_sha) is True
        assert self._head(ws / "proj") == new_sha

    def test_branch_name_pin_is_checked_out_via_origin(self, tmp_path, remote):
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"], commit="feature") is True
        assert self._head(ws / "proj") == remote["feature"]

    def test_unpinned_run_reattaches_default_branch_after_a_pinned_one(self, tmp_path, remote):
        """Review finding: the same project_path left detached at an old pin fed that
        old tree to the next unpinned run."""
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"], commit=remote["feature"]) is True
        assert self._head(ws / "proj") == remote["feature"]
        assert cb.clone_repo("proj", remote["url"]) is True
        assert self._head(ws / "proj") == remote["main"]
        assert _git(ws / "proj", "symbolic-ref", "-q", "HEAD") == "refs/heads/main"

    def test_recovering_a_full_clone_does_not_make_it_shallow(self, tmp_path, remote):
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"], commit=remote["main"]) is True
        work = remote["work"]
        (work / "a.txt").write_text("3\n"); _git(work, "add", "."); _git(work, "commit", "-q", "-m", "c3")
        new_sha = _git(work, "rev-parse", "HEAD"); _git(work, "push", "-q", remote["url"], "main")
        assert cb.clone_repo("proj", remote["url"], commit=new_sha) is True
        assert self._head(ws / "proj") == new_sha
        assert not (ws / "proj/.git/shallow").exists()

    def test_fetch_timeout_returns_false_instead_of_raising(self, tmp_path, remote, monkeypatch):
        """Review finding: a hung fetch raised TimeoutExpired out of clone_repo."""
        import subprocess as sp
        from execution_agent import context as ctx_mod
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"]) is True            # shallow, main only
        real_run = sp.run
        def flaky(cmd, *a, **kw):
            if "fetch" in cmd:
                raise sp.TimeoutExpired(cmd, kw.get("timeout", 1))
            return real_run(cmd, *a, **kw)
        monkeypatch.setattr(ctx_mod.subprocess, "run", flaky)
        assert cb.clone_repo("proj", remote["url"], commit=remote["feature"]) is False

    def test_unknown_commit_returns_false(self, tmp_path, remote):
        ws = tmp_path / "ws"
        cb = ContextBuilder(workspace_root=str(ws))
        assert cb.clone_repo("proj", remote["url"]) is True
        assert cb.clone_repo("proj", remote["url"], commit="0" * 40) is False


# =========================================================================== Fix 4
@pytest.fixture
def launcher():
    """Fresh launcher module per test: register_projects mutates module globals."""
    spec = importlib.util.spec_from_file_location("launcher_under_test", REPO_ROOT / "launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_projects(tmp_path: Path, entries, wrap=False) -> Path:
    p = tmp_path / "projects.json"
    p.write_text(json.dumps({"projects": entries} if wrap else entries))
    return p


class TestLauncherProjectsFile:
    def test_builtin_safe_names_unchanged(self, launcher):
        # safe_name now also sanitises characters like '@' and '/'; the built-in
        # names must keep their historical project_path (it is a workspace dir).
        for p in launcher.PROJECTS:
            legacy = p.name.lower().replace("-", "_").replace(".", "_")
            assert p.safe_name == legacy, p.name

    def test_safe_name_sanitises_pinned_names(self, launcher):
        p = launcher.Project("commons-csv@a1b2c3d", "u", "Java", "t")
        assert p.safe_name == "commons_csv_a1b2c3d"

    def test_load_defaults_and_wrapped_form(self, launcher, tmp_path):
        f = _write_projects(tmp_path, [
            {"name": "commons-csv@abc", "url": "https://github.com/apache/commons-csv",
             "language": "Java", "commit": " abc123 \n"},
        ], wrap=True)
        [p] = launcher.load_projects_file(f)
        assert p.commit == "abc123"
        assert p.image_tag == "commons_csv_abc_image:ExecutionAgent"

    @pytest.mark.parametrize("bad", [
        [{"name": "x", "language": "Java"}],           # missing url
        [{"url": "u", "language": "Java"}],            # missing name
        ["not-an-object"],
        {"nope": []},
    ])
    def test_load_rejects_malformed(self, launcher, tmp_path, bad):
        f = _write_projects(tmp_path, bad)
        with pytest.raises(ValueError):
            launcher.load_projects_file(f)

    def test_register_replaces_builtin_by_name_case_insensitive(self, launcher):
        n = len(launcher.PROJECTS)
        launcher.register_projects([launcher.Project("PANDAS", "u", "Python", "t", commit="c0ffee")])
        assert len(launcher.PROJECTS) == n
        assert launcher.PROJECTS_BY_NAME["pandas"].commit == "c0ffee"
        assert launcher.PROJECTS_BY_NAME["pandas"].url == "u"
        assert sum(1 for p in launcher.PROJECTS_BY_LANGUAGE["python"] if p.name.lower() == "pandas") == 1

    def test_two_file_entries_naming_the_same_builtin_are_reported(self, launcher):
        with pytest.raises(ValueError, match="safe_name"):
            launcher.register_projects([launcher.Project("pandas", "u", "Python", "t", commit="a"),
                                        launcher.Project("PANDAS", "u", "Python", "t", commit="b")])

    def test_register_rejects_safe_name_collision(self, launcher):
        # "Chart.js" is built in and maps to chart_js, as would "chart-js"
        with pytest.raises(ValueError, match="safe_name"):
            launcher.register_projects([launcher.Project("chart-js", "u", "Javascript", "t")])

    def test_same_repo_two_commits_are_two_projects(self, launcher, tmp_path):
        f = _write_projects(tmp_path, [
            {"name": "csv@aaa", "url": "https://x/csv", "language": "Java", "commit": "aaa"},
            {"name": "csv@bbb", "url": "https://x/csv", "language": "Java", "commit": "bbb"},
        ])
        launcher.register_projects(launcher.load_projects_file(f))
        sel = launcher.resolve_project_selection("csv@aaa,csv@bbb")
        assert [p.commit for p in sel] == ["aaa", "bbb"]
        assert len({p.safe_name for p in sel}) == 2
        assert any(p.name == "csv@bbb" for p in launcher.PROJECTS_BY_LANGUAGE["java"])

    def test_commit_override_requires_single_project(self, launcher, capsys):
        two = launcher.resolve_project_selection("pandas,scipy")
        assert launcher.apply_commit_override(two, "abc") is None
        assert "exactly one project" in capsys.readouterr().out
        [one] = launcher.apply_commit_override([two[0]], " abc ")
        assert one.commit == "abc" and one.name == two[0].name
        assert launcher.apply_commit_override(two, None) is two


class TestLauncherCli:
    def _run(self, launcher, monkeypatch, *argv) -> int:
        monkeypatch.setattr(sys, "argv", ["launcher.py", *argv])
        return launcher.main()

    def test_create_meta_writes_commit_from_file(self, launcher, tmp_path, monkeypatch):
        f = _write_projects(tmp_path, [
            {"name": "commons-csv@abc", "url": "https://x/csv", "language": "Java", "commit": "abc123"},
        ])
        ws = tmp_path / "ws"
        rc = self._run(launcher, monkeypatch, "--projects-file", str(f),
                       "--create-meta", "commons-csv@abc", "--workspace-root", str(ws))
        assert rc == 0
        [meta] = list((ws / "metadata").glob("*.json"))
        data = json.loads(meta.read_text())
        assert data["commit"] == "abc123"
        assert data["project_path"] == "commons_csv_abc"
        assert data["project_url"] == "https://x/csv"

    def test_create_meta_commit_flag_pins_builtin(self, launcher, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        rc = self._run(launcher, monkeypatch, "--create-meta", "pandas",
                       "--commit", "feedface", "--workspace-root", str(ws))
        assert rc == 0
        data = json.loads(next((ws / "metadata").glob("*.json")).read_text())
        assert data["commit"] == "feedface"
        assert data["project_url"] == "https://github.com/pandas-dev/pandas"

    def test_unpinned_metadata_has_empty_commit(self, launcher, tmp_path, monkeypatch):
        ws = tmp_path / "ws"
        assert self._run(launcher, monkeypatch, "--create-meta", "pandas", "--workspace-root", str(ws)) == 0
        data = json.loads(next((ws / "metadata").glob("*.json")).read_text())
        assert data["commit"] == ""

    def test_commit_flag_with_many_projects_fails_before_running(self, launcher, tmp_path, monkeypatch, capsys):
        rc = self._run(launcher, monkeypatch, "--run", "pandas,scipy", "--commit", "abc",
                       "--dry-run", "--workspace-root", str(tmp_path / "ws"))
        assert rc == 1
        assert "exactly one project" in capsys.readouterr().out

    def test_bad_projects_file_fails_cleanly(self, launcher, tmp_path, monkeypatch, capsys):
        f = tmp_path / "p.json"; f.write_text("{not json")
        rc = self._run(launcher, monkeypatch, "--projects-file", str(f), "--list",
                       "--workspace-root", str(tmp_path / "ws"))
        assert rc == 1
        assert "Could not load --projects-file" in capsys.readouterr().out

    def test_list_shows_commit_column(self, launcher, tmp_path, monkeypatch, capsys):
        f = _write_projects(tmp_path, [
            {"name": "csv@abc", "url": "https://x/csv", "language": "Java", "commit": "abcdef0123456789"},
        ])
        assert self._run(launcher, monkeypatch, "--projects-file", str(f), "--list",
                         "--workspace-root", str(tmp_path / "ws")) == 0
        out = capsys.readouterr().out
        assert re.search(r"csv@abc\s+Java\s+https://x/csv\s+abcdef012345\b", out)


# =========================================================================== PR change 2 (REPO_URL build arg), real docker
@pytest.fixture(scope="module")
def docker_client():
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"docker daemon not reachable: {e}")
    return client


class TestRepoUrlBuildArg:
    @pytest.fixture
    def dockerfile_dir(self, tmp_path):
        (tmp_path / "Dockerfile").write_text(
            "FROM busybox:latest\n"
            "ARG REPO_URL\n"
            'RUN test -n "$REPO_URL" && echo "url=$REPO_URL" > /url.txt\n'
            'CMD ["cat", "/url.txt"]\n'
        )
        return tmp_path

    @pytest.mark.slow
    def test_build_arg_is_passed_and_absent_without_it(self, docker_client, dockerfile_dir):
        from execution_agent import tools
        tag = "executionagent-test-repourl:pr8"
        try:
            ok, log = tools._docker_build_image(str(dockerfile_dir), tag, repo_url="https://example.com/r.git")
            assert ok, log
            assert "build_args={'REPO_URL': 'https://example.com/r.git'}" in log
            out = docker_client.containers.run(tag, remove=True).decode()
            assert out.strip() == "url=https://example.com/r.git"

            ok2, log2 = tools._docker_build_image(str(dockerfile_dir), tag + "-none")
            assert not ok2, "template's `test -n \"$REPO_URL\"` must fail when no build arg is given"
        finally:
            for t in (tag, tag + "-none"):
                try:
                    docker_client.images.remove(t, force=True)
                except Exception:
                    pass

    def test_launch_script_forwards_repo_url(self):
        s = exit_artifacts._generate_launch_script("Dockerfile", "commands.sh", "proj",
                                                   docker_tag="t:1", project_url="https://x/y.git")
        assert "REPO_URL='https://x/y.git'" in s
        assert 'docker build --build-arg REPO_URL="$REPO_URL" -t' in s
        assert "set -e" not in s and "set +e" in s          # commands.sh may legitimately exit non-zero
        assert 'echo "ERROR: Docker build failed with exit code $BUILD_STATUS"' in s
        s2 = exit_artifacts._generate_launch_script("Dockerfile", "commands.sh", "proj", docker_tag="t:1")
        assert "--build-arg" not in s2
        s3 = exit_artifacts._generate_launch_script("Dockerfile", "commands.sh", "proj", docker_tag="t:1",
                                                    project_url="https://x/y.git", commit="abc123")
        assert "COMMIT_SHA='abc123'" in s3
        assert '--build-arg REPO_URL="$REPO_URL" --build-arg COMMIT_SHA="$COMMIT_SHA"' in s3


# =========================================================================== harness-side pin verification
class TestVerifyPinnedCommit:
    """The prompt alone is advisory: live with gpt-5-mini, a Dockerfile cloned fine but
    skipped the checkout, so the container silently held HEAD. The harness now checks."""

    class _Exec:
        def __init__(self, exit_code, output):
            self.exit_code, self.output = exit_code, output

    class _Container:
        """Answers the checkout probe with a fixed (HEAD, resolved-pin) pair."""
        def __init__(self, exit_code, output):
            self._r = TestVerifyPinnedCommit._Exec(exit_code, output)
            self.removed = False
            self.id = "cid"
            self.probes = []

        def exec_run(self, cmd, tty=False):
            self.probes.append(cmd[-1])
            return self._r

        def remove(self, force=False):
            self.removed = True

    SHA = "330123258e8c3dc391cbe55ab1ed94891ca83af3"
    OTHER = "d318b683471101618febed18996405ad26462110"

    @classmethod
    def _at(cls, head, pin, where=".", url="https://github.com/o/r"):
        return cls._Container(0, f"DIR={where} HEAD={head} PIN={pin or ''} URL={url}\nPROBE-DONE\n")

    @classmethod
    def _many(cls, *rows):
        return cls._Container(0, "".join(f"DIR={w} HEAD={h} PIN={p or ''} URL={u}\n" for w, h, p, u in rows) + "PROBE-DONE\n")

    def test_dependency_checkout_sorting_first_is_not_mistaken_for_the_project(self):
        """Review finding: the first `.git` found was compared, so a dependency clone
        under /app got a correct build destroyed as a 'mismatch'."""
        from execution_agent import tools
        c = self._many(("/app/abseil-cpp", self.OTHER, None, "https://github.com/abseil/abseil-cpp"),
                       ("/app/project", self.SHA, self.SHA, "https://github.com/o/r"))
        assert tools._verify_pinned_commit(c, self.SHA, project_url="https://github.com/o/r") == ("ok", self.SHA)
        # even without a URL match, the checkout where the pin resolves wins
        assert tools._verify_pinned_commit(c, self.SHA)[0] == "ok"

    def test_project_checkout_is_found_by_origin_url_when_pin_is_unresolvable(self):
        from execution_agent import tools
        c = self._many(("/app/dep", self.OTHER, None, "git@github.com:x/dep.git"),
                       ("/app/project", self.OTHER, None, "https://github.com/O/R.git"))
        status, detail = tools._verify_pinned_commit(c, self.SHA, project_url="https://github.com/o/r")
        assert status == "mismatch" and "/app/project" in detail

    def test_ambiguous_layout_is_unverifiable_not_punished(self):
        from execution_agent import tools
        c = self._many(("/app/a", self.OTHER, None, "https://x/a"), ("/app/b", self.OTHER, None, "https://x/b"))
        status, detail = tools._verify_pinned_commit(c, self.SHA, project_url="https://github.com/o/r")
        assert status == "unverifiable" and "several git checkouts" in detail

    def test_ok_when_head_matches(self):
        from execution_agent import tools
        assert tools._verify_pinned_commit(self._at(self.SHA, self.SHA), self.SHA) == ("ok", self.SHA)

    def test_tag_branch_and_uppercase_pins_are_resolved_by_git(self):
        """Review finding: the old prefix comparison force-removed correct containers
        for tag/branch/uppercase pins. git resolves the ref inside the checkout now."""
        from execution_agent import tools
        for pin in ("v2.3.1", "release/2.x", self.SHA.upper(), self.SHA[:8].upper()):
            assert tools._verify_pinned_commit(self._at(self.SHA, self.SHA), pin) == ("ok", self.SHA), pin

    def test_unresolvable_hex_prefix_falls_back_to_prefix_match(self):
        from execution_agent import tools
        assert tools._verify_pinned_commit(self._at(self.SHA, None), self.SHA[:10])[0] == "ok"
        assert tools._verify_pinned_commit(self._at(self.OTHER, None), self.SHA[:10])[0] == "mismatch"

    def test_unresolvable_tag_is_a_mismatch(self):
        from execution_agent import tools
        status, detail = tools._verify_pinned_commit(self._at(self.OTHER, None), "v9.9.9")
        assert status == "mismatch" and "not a known ref" in detail

    def test_mismatch_when_head_is_default_branch(self):
        from execution_agent import tools
        status, detail = tools._verify_pinned_commit(self._at(self.OTHER, self.SHA, where="/app/project"), self.SHA)
        assert status == "mismatch" and detail.startswith(self.OTHER) and "/app/project" in detail

    def test_unverifiable_when_no_checkout_anywhere(self):
        from execution_agent import tools
        status, detail = tools._verify_pinned_commit(self._Container(0, "PROBE-DONE\n"), self.SHA)
        assert status == "unverifiable" and "no git checkout found" in detail

    def test_probe_searches_workdir_then_project_dir_then_app_then_subdirs(self):
        """Review finding: a Dockerfile that clones into $PROJECT_DIR without a WORKDIR
        left the pin 'unverifiable'. The probe now looks where generated Dockerfiles
        put the checkout."""
        from execution_agent import tools
        c = self._at(self.SHA, self.SHA)
        tools._verify_pinned_commit(c, self.SHA, project_path="commons_csv")
        probe = c.probes[-1]
        assert probe.startswith("for d in . /app/commons_csv /app/* */; do")
        assert "rev-parse --verify HEAD" in probe and "^{commit}" in probe and "remote.origin.url" in probe

    def test_write_to_file_discards_mismatched_container_and_image(self, tmp_path, monkeypatch):
        """End-to-end through write_to_file with the docker build/start stubbed out."""
        from execution_agent import tools
        wrong = self._at(self.OTHER, self.SHA)
        removed_images = []
        class Images:
            def remove(self, tag, force=False): removed_images.append(tag)
        class Client:
            images = Images()
        monkeypatch.setattr(tools, "_docker_client", lambda: Client())
        monkeypatch.setattr(tools, "_docker_build_image", lambda d, tag, repo_url="", commit="": (True, "built"))
        monkeypatch.setattr(tools, "_docker_start_container", lambda tag: wrong)

        class Agent: pass
        a = Agent()
        a.workspace_path = str(tmp_path); a.project_path = "proj"; a.project_url = "https://x/y"
        a.commit = self.SHA; a.written_files = []; a.container = None; a.env = Agent()
        res = tools.write_to_file(file_path="Dockerfile", content="FROM ubuntu:24.04\nRUN true\n", agent=a)
        assert res["returncode"] == 1
        assert "NOT at the required commit" in res["output"]
        assert self.OTHER in res["output"]
        assert wrong.removed, "the wrong container must be discarded"
        assert removed_images and removed_images[0].startswith("tool"), "the built image must be removed too"
        assert a.docker_tag == "", "a discarded image's tag must not leak into exit artifacts"
        assert a.container is None, "agent must not adopt the wrong container"
        assert getattr(a, "base_dockerfile", "") == "", "a mismatched Dockerfile must not become the exit artifact"

    def test_write_to_file_accepts_matching_container(self, tmp_path, monkeypatch):
        from execution_agent import tools
        right = self._at(self.SHA, self.SHA)
        seen = {}
        def fake_build(d, tag, repo_url="", commit=""):
            seen["commit"] = commit; return (True, "built")
        monkeypatch.setattr(tools, "_docker_build_image", fake_build)
        monkeypatch.setattr(tools, "_docker_start_container", lambda tag: right)

        class Agent: pass
        a = Agent()
        a.workspace_path = str(tmp_path); a.project_path = "proj"; a.project_url = "https://x/y"
        a.commit = self.SHA; a.written_files = []; a.container = None; a.env = Agent()
        res = tools.write_to_file(file_path="Dockerfile", content="FROM ubuntu:24.04\nRUN true\n", agent=a)
        assert res["returncode"] == 0, res["output"]
        assert "Verified: container HEAD is the required commit" in res["output"]
        assert a.container is right and a.base_dockerfile.startswith("FROM ubuntu")
        assert seen["commit"] == self.SHA, "COMMIT_SHA must be passed as a build arg"

    def test_write_to_file_unpinned_does_not_check(self, tmp_path, monkeypatch):
        from execution_agent import tools
        class NoGit(self._Container):
            def exec_run(self, cmd, tty=False):
                if "rev-parse" in cmd[-1]:
                    raise AssertionError("must not verify when no commit is pinned")
                return TestVerifyPinnedCommit._Exec(0, "/app/project\n")
        c = NoGit(0, "")
        monkeypatch.setattr(tools, "_docker_build_image", lambda d, tag, repo_url="", commit="": (True, "built"))
        monkeypatch.setattr(tools, "_docker_start_container", lambda tag: c)
        class Agent: pass
        a = Agent()
        a.workspace_path = str(tmp_path); a.project_path = "proj"; a.project_url = "https://x/y"
        a.commit = ""; a.written_files = []; a.container = None; a.env = Agent()
        res = tools.write_to_file(file_path="Dockerfile", content="FROM ubuntu:24.04\nRUN true\n", agent=a)
        assert res["returncode"] == 0, res["output"]
        assert "Verified" not in res["output"] and "WARNING" not in res["output"]


# =========================================================================== commands.sh: exit codes + keep going
class TestCommandsScript:
    """Live on 4/4 artifacts, launch.sh stopped at the first recorded command that had
    failed during the run, and the later command that actually succeeded depended on
    what the failing one installed. The script must keep going and report exit codes."""

    def _agent_with_history(self):
        class A: pass
        a = A()
        a.command_history = [
            {"cycle": 1, "tool": "write_to_file", "args": {"filename": "Dockerfile", "text": "FROM a"}, "returncode": 1},  # build failed
            {"cycle": 2, "tool": "linux_terminal", "args": {"command": "cat README"}, "returncode": 0},             # pre-container
            {"cycle": 3, "tool": "write_to_file", "args": {"file_path": "Dockerfile", "content": "FROM b"}, "returncode": 0},  # built
            {"cycle": 4, "tool": "linux_terminal", "args": {"command": "apt-get install -y wget && wget bad-url"}, "returncode": 8},
            {"cycle": 5, "tool": "read_file", "args": {"file_path": "pom.xml"}, "returncode": 0},
            {"cycle": 6, "tool": "linux_terminal", "args": {"command": "wget good-url && mvn test"}, "returncode": 0},
            # rejected: "Cannot write a Dockerfile after the container is running" (container untouched)
            {"cycle": 7, "tool": "write_to_file", "args": {"path": "docker/Dockerfile/"}, "returncode": 1},
            {"cycle": 8, "tool": "linux_terminal", "args": {"command": "mvn verify"}, "returncode": 0},
            {"cycle": 9, "tool": "linux_terminal", "args": {"command": "./gradlew test"}, "returncode": 124},   # stuck
            {"cycle": 10, "tool": "linux_terminal", "args": {"command": "WAIT"}, "returncode": 0},             # not shell
            {"cycle": 11, "tool": "linux_terminal", "args": {"command": "WRITE:y"}, "returncode": 0},          # not shell
            {"cycle": 12, "tool": "linux_terminal", "args": {"command": "TERMINATE"}, "returncode": 0},        # not shell
            {"cycle": 13, "tool": "linux_terminal", "args": {"command": "apt-get install curl"},
             "executed_command": "DEBIAN_FRONTEND=noninteractive apt-get install -y curl", "returncode": 0},
            {"cycle": 14, "tool": "goals_accomplished", "args": {"reason": "done"}, "returncode": 0},
        ]
        a.commands_and_summary = []
        return a

    def test_extract_uses_history_and_exit_codes(self):
        cmds = exit_artifacts._extract_container_commands(self._agent_with_history())
        assert cmds == [("apt-get install -y wget && wget bad-url", 8),
                        ("wget good-url && mvn test", 0),
                        ("mvn verify", 0),                                          # kept despite the rejected rewrite
                        ("./gradlew test", 124),                                    # stuck commands are kept, annotated
                        ("DEBIAN_FRONTEND=noninteractive apt-get install -y curl", 0)]   # what actually ran
        s = exit_artifacts._generate_commands_script(cmds, "proj")
        assert "(exit code during the run: 124, timed out / stuck)" in s

    def test_container_file_writes_are_replayed_in_order(self, tmp_path):
        """Live (flask, pinned): the model patched tests/conftest.py and a pytest
        internal with write_to_file before its final pytest run; replaying only the
        shell commands could not reproduce the result."""
        class A: pass
        a = A()
        target_dir = tmp_path / "app" / "project"
        conftest = str(target_dir / "tests" / "conftest.py")
        a.command_history = [
            {"cycle": 1, "tool": "write_to_file", "args": {"filename": "Dockerfile"}, "returncode": 0},
            {"cycle": 2, "tool": "linux_terminal", "args": {"command": "echo before"}, "returncode": 0},
            {"cycle": 3, "tool": "write_to_file", "args": {"file_path": "tests/conftest.py", "content": "<40 chars omitted>"}, "returncode": 0},
            {"cycle": 4, "tool": "linux_terminal", "args": {"command": f"grep -q PATCHED {conftest} && echo after"}, "returncode": 0},
        ]
        a.written_files = [
            ("Dockerfile", "local", "/x/Dockerfile", "FROM a"),
            ("tests/conftest.py", "container", conftest, "# PATCHED\nline with 'quotes' and $dollars\n"),
        ]
        steps = exit_artifacts._extract_container_steps(a)
        assert [s["kind"] for s in steps] == ["command", "write", "command"]
        assert steps[1]["path"] == conftest and "PATCHED" in steps[1]["content"]
        script = exit_artifacts._generate_commands_script(steps, "proj")
        assert "1 file write(s)" in script and f"cat > '{conftest}' <<'__EA_FILE_2__'" in script
        f = tmp_path / "commands.sh"; f.write_text(script)
        r = subprocess.run(["bash", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert r.stdout.splitlines() == ["before", "after"]
        assert (target_dir / "tests" / "conftest.py").read_text() == "# PATCHED\nline with 'quotes' and $dollars\n"
        assert "command 2 (write" in r.stderr

    def test_dockerfile_write_detection_matches_the_tool(self):
        f = exit_artifacts._is_dockerfile_write
        assert f("write_to_file", {"file_path": "docker/Dockerfile/"})      # trailing slash, as normpath sees it
        assert f("write_to_file", {"path": "build.Dockerfile"})
        assert f("write_to_file", {"filename": "DOCKERFILE"})
        assert not f("write_to_file", {"file_path": "Dockerfile.md"})
        assert not f("linux_terminal", {"command": "cat Dockerfile"})

    def test_extract_falls_back_to_summaries_without_history(self):
        class A: pass
        a = A(); a.command_history = []
        a.commands_and_summary = [
            ('Call to tool write_to_file with arguments {"filename": "Dockerfile"}', {}),
            ('Call to tool linux_terminal with arguments {"command": "mvn test"}', {}),
        ]
        assert exit_artifacts._extract_container_commands(a) == [("mvn test", None)]

    def test_script_has_no_set_e_and_annotates_exit_codes(self):
        s = exit_artifacts._generate_commands_script([("false", 1), ("echo ok", 0), ("mystery", None)], "proj")
        assert "set -e" not in s and "set +e" in s
        assert "# Command 1 (exit code during the run: 1)" in s
        assert "# Command 2 (exit code during the run: 0)" in s
        assert "# Command 3 (exit code during the run: unknown)" in s
        assert "1 with a non-zero exit during the run" in s
        assert s.rstrip().endswith("exit $_rc")

    def test_empty_command_list_is_a_valid_noop_script(self, tmp_path):
        s = exit_artifacts._generate_commands_script([], "proj")
        assert "# Command 1" not in s and "No commands were executed" in s
        f = tmp_path / "c.sh"; f.write_text(s)
        assert subprocess.run(["bash", str(f)], capture_output=True).returncode == 0

    def test_script_keeps_going_after_failure_and_exits_with_last_status(self, tmp_path):
        s = exit_artifacts._generate_commands_script(
            [("cd /tmp && export MARK=set && false", 1),          # fails, but cd/export must persist
             ("test \"$MARK\" = set && test \"$PWD\" = /tmp && echo REACHED", 0)], "proj")
        f = tmp_path / "commands.sh"; f.write_text(s)
        r = subprocess.run(["bash", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "REACHED" in r.stdout
        assert "command 1 exited with 1 (during the run: 1)" in r.stderr
        assert "command 2 exited with 0 (during the run: 0)" in r.stderr

    def test_script_exit_status_is_last_commands(self, tmp_path):
        s = exit_artifacts._generate_commands_script([("echo first", 0), ("exit 3", 3)], "proj")
        f = tmp_path / "commands.sh"; f.write_text(s)
        assert subprocess.run(["bash", str(f)], capture_output=True).returncode == 3

    def test_agent_records_history_with_returncode(self, monkeypatch):
        agent = _make_agent()
        agent.tool_registry = None
        monkeypatch.setattr(agent, "_plan_next_action", lambda task: {"tool_name": "linux_terminal", "tool_args": {"command": "false"}, "raw": {"thoughts": ""}}, raising=False)
        monkeypatch.setattr(agent, "_execute_action", lambda name, args: {"output": "x", "returncode": 1}, raising=False)
        monkeypatch.setattr(agent, "_summarize_last_command", lambda out: {"summary": "s"}, raising=False)
        agent.run_one_cycle("task")
        assert agent.command_history[-1]["tool"] == "linux_terminal"
        assert agent.command_history[-1]["returncode"] == 1

    def test_history_is_recorded_even_if_the_summariser_fails(self, monkeypatch):
        """Review finding: the entry was appended after summarising, so a summariser
        exception on the Dockerfile cycle lost the 'container started' boundary."""
        agent = _make_agent()
        monkeypatch.setattr(agent, "_plan_next_action", lambda task: {"tool_name": "write_to_file", "tool_args": {"file_path": "Dockerfile", "content": "FROM x"}, "raw": {"thoughts": ""}}, raising=False)
        monkeypatch.setattr(agent, "_execute_action", lambda name, args: {"output": "built", "returncode": 0}, raising=False)
        def boom(out): raise RuntimeError("LLM timed out")
        monkeypatch.setattr(agent, "_summarize_last_command", boom, raising=False)
        with pytest.raises(RuntimeError):
            agent.run_one_cycle("task")
        assert agent.command_history[-1]["tool"] == "write_to_file" and agent.command_history[-1]["returncode"] == 0
        assert agent.command_history[-1]["args"]["content"].endswith("chars omitted>")   # bulk payload slimmed

    def test_history_prefers_the_command_that_actually_ran(self, monkeypatch):
        agent = _make_agent()
        monkeypatch.setattr(agent, "_plan_next_action", lambda task: {"tool_name": "linux_terminal", "tool_args": {"command": "apt-get install curl"}, "raw": {"thoughts": ""}}, raising=False)
        def exec_action(name, args):
            agent.last_executed_command = "DEBIAN_FRONTEND=noninteractive apt-get install -y curl"   # as linux_terminal does
            return {"output": "ok", "returncode": 0}
        monkeypatch.setattr(agent, "_execute_action", exec_action, raising=False)
        monkeypatch.setattr(agent, "_summarize_last_command", lambda out: {"summary": "s"}, raising=False)
        agent.run_one_cycle("task")
        assert agent.command_history[-1]["executed_command"] == "DEBIAN_FRONTEND=noninteractive apt-get install -y curl"
        assert agent.last_executed_command is None

    def test_execute_action_propagates_tool_returncode(self):
        """Live: the executor logged exit code 1 but command_history recorded 0, because
        _execute_action returned a constant 0. Go through the registry, not around it."""
        agent = _make_agent()
        class Registry:
            def call(self, name, args, agent=None):
                return {"output": "Output in terminal after executing the command:\nBUILD FAILURE", "returncode": 1}
        agent.tool_registry = Registry()
        raw = agent._execute_action("linux_terminal", {"command": "mvn test"})
        assert raw["returncode"] == 1
        assert raw["output"].startswith("{'output': 'Output in terminal")   # what the model sees is unchanged

    def test_execute_action_defaults_to_zero_for_non_dict_results(self):
        agent = _make_agent()
        class Registry:
            def call(self, name, args, agent=None): return "plain text"
        agent.tool_registry = Registry()
        assert agent._execute_action("read_file", {"file_path": "x"})["returncode"] == 0

    def test_state_roundtrip_keeps_history(self, tmp_path):
        from execution_agent import state_persistence as sp
        st = sp.AgentState(command_history=[{"cycle": 1, "tool": "linux_terminal", "args": {"command": "x"}, "returncode": 2}])
        assert hasattr(sp.AgentState, "from_dict") and hasattr(st, "to_dict")
        assert sp.AgentState.from_dict(st.to_dict()).command_history[0]["returncode"] == 2


# =========================================================================== screen exit codes
class TestScreenExitCodes:
    """Every in-container command was reported as exit code 0: screen's `stuff` expands
    `$__rc` in the typed payload. The status capture now lives in a file that is
    written via docker exec and merely sourced."""

    class _Rec:
        def __init__(self): self.cmds = []
        def exec_run(self, cmd, tty=False):
            self.cmds.append(cmd[-1])
            class R: exit_code, output = 0, b""
            return R()

    def test_staging_script_uses_literal_paths_and_no_positional_params(self):
        """Review finding: a wrapper sourced with arguments leaked $1..$4 into the user's
        command (`source activate` forwards "$@") and depended on them afterwards."""
        from execution_agent import shared_utils as su
        wrapper, staging = su.screen_staging_script("abc", "/tmp/l.log", "/tmp/s.sh", "echo hi", source=True)
        assert wrapper == "/tmp/screen_wrap_abc.sh"
        assert staging.startswith("set -e\n")                                          # any staging failure -> non-zero
        assert "cat > /tmp/s.sh <<'__SRC_abc__'\necho hi\n__SRC_abc__" in staging
        assert "if . /tmp/s.sh >> /tmp/l.log 2>&1; then __rc=0; else __rc=$?; fi" in staging   # `if`: errexit-safe
        assert 'printf "<<RC:abc:%d>>\\n" "$__rc" >> /tmp/l.log' in staging
        assert "rm -f /tmp/s.sh /tmp/screen_wrap_abc.sh" in staging                      # self-cleaning
        assert "$1" not in staging and "$2" not in staging and "$3" not in staging
        _, probe = su.screen_staging_script("x", "/l", "/s", "pwd", source=False)
        assert "if /usr/bin/env bash /s >> /l 2>&1" in probe

    def test_stuff_escape(self):
        from execution_agent import shared_utils as su
        assert su.stuff_escape("$HOME it's a\\b") == "\\$HOME it'\\''s a\\\\b"
        assert su.stuff_escape("grep ^foo") == "grep \\^foo"          # screen turns ^X into Ctrl-X otherwise

    def test_stateful_executor_never_types_a_dollar_into_screen(self, monkeypatch):
        """Drive exec_in_screen_and_get_log with a fake container and inspect what gets
        stuffed. The fake answers the RC marker on the first tail read."""
        from execution_agent import tools
        typed = []
        class Fake:
            def exec_run(self, cmd, tty=False):
                text = cmd[-1]
                if "-X stuff" in text:
                    typed.append(text)
                class R: exit_code, output = 0, b""
                return R()
        def fake_tail(container, path, max_bytes=0):
            import re
            rid = re.search(r"screen_exec_stateful_([0-9a-f]+)\.log", path).group(1)
            return f"<<BEGIN:{rid}>>\nhello\n<<END:{rid}>>\n<<RC:{rid}:7>>\n"
        monkeypatch.setattr(tools, "read_file_tail", fake_tail)
        rc, out, _, stuck = tools.exec_in_screen_and_get_log(Fake(), "echo hello; exit 7")
        assert (rc, stuck) == (7, False) and out.strip() == "hello"
        stuffed = [t for t in typed if "screen_wrap_" in t]
        assert stuffed, typed
        payload = stuffed[0].split("-X stuff ", 1)[1]
        assert "$" not in payload, payload
        import re as _re
        assert _re.fullmatch(r"'\. /tmp/screen_wrap_[0-9a-f]+\.sh\\r'", payload), payload   # bare `. <wrapper>`, no arguments

    def test_staging_failure_fails_fast(self, monkeypatch):
        """Review finding: a failed wrapper/script write used to mean a 300 s 'stuck' verdict."""
        from execution_agent import tools
        class Fake:
            def exec_run(self, cmd, tty=False):
                class R: exit_code, output = (1, b"sh: /tmp: Read-only file system") if "cat >" in cmd[-1] else (0, b"")
                return R()
        rc, out, _, stuck = tools.exec_in_screen_and_get_log(Fake(), "true")
        assert rc == 1 and not stuck and "Could not stage the command" in out

    @pytest.mark.slow
    def test_live_exit_codes_through_screen(self, docker_client):
        """Real container, real screen session, real sourced commands."""
        from execution_agent import tools
        c = docker_client.containers.run("ubuntu:24.04", "sleep infinity", detach=True, remove=True)
        try:
            msg = tools.create_screen_session(c)
            assert not (isinstance(msg, str) and msg.startswith("{")), msg   # a JSON string would be an error result
            cases = [("true", 0), ("false", 1), ("cd /nonexistent && echo never", 1),
                     ("sh -c 'exit 2'", 2), ("cd /tmp && export MARK=1 && false", 1),
                     ('test "$MARK" = 1 && test "$PWD" = /tmp', 0),         # state persisted from the previous command
                     ("set -e; false", 1),                                  # errexit must not kill the shell...
                     ("echo still-alive", 0), ("set +e", 0),               # ...so this still runs in the same shell
                     ("set -- X Y Z", 0), ('test "$#" = 3', 0),           # positional params belong to the user...
                     ("shift 3", 0), ("echo n=$#", 0)]                     # ...and cannot break the markers
            got = [(cmd, tools.exec_in_screen_and_get_log(c, cmd)[0]) for cmd, _ in cases]
            assert got == cases
            rc, out = tools._exec(c, "ls /tmp/screen_src_* /tmp/screen_wrap_* /tmp/screen_health_*.sh 2>/dev/null | wc -l")
            assert out.strip() == "0", f"litter left in /tmp: {out}"
        finally:
            c.remove(force=True)
