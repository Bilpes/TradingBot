"""Repository hygiene: nothing importable may be invisible to git.

This exists because of a real failure. ``.gitignore`` contained an unanchored
``data/``, which matches at *any* depth, so the whole ``tradingbot/data/``
package -- schema, providers, the Dhan client -- was silently excluded from
every commit. The working tree looked fine, tests passed locally, and the
pushed repository failed at ``import tradingbot``.

An unanchored ignore pattern is easy to write and impossible to notice from the
inside, so it gets checked explicitly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories that legitimately hold non-source files.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist",
             ".pytest_cache", ".mypy_cache", ".ruff_cache", "runs", "data"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture(scope="module")
def git_available() -> bool:
    try:
        _git("rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _tracked() -> set[str]:
    return set(_git("ls-files").splitlines())


def test_no_python_source_file_is_untracked(git_available):
    """Every .py in the package must be committed, or the clone is broken."""
    if not git_available:
        pytest.skip("git not available")

    tracked = _tracked()
    on_disk = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in REPO_ROOT.rglob("*.py")
        if not any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts[:-1])
    }
    missing = sorted(on_disk - tracked)
    assert not missing, (
        f"{len(missing)} source file(s) exist on disk but are not tracked by git "
        f"-- a fresh clone cannot import the package: {missing}"
    )


def test_the_package_directory_itself_is_not_ignored(git_available):
    """Regression: `data/` in .gitignore matched tradingbot/data/ too."""
    if not git_available:
        pytest.skip("git not available")

    for relpath in ("tradingbot/data/schema.py", "tradingbot/data/providers.py",
                    "tradingbot/data/universe.py", "tradingbot/data/dhan.py",
                    "tradingbot/data/__init__.py"):
        result = subprocess.run(["git", "check-ignore", "-q", relpath], cwd=REPO_ROOT)
        assert result.returncode != 0, f"{relpath} is excluded by .gitignore"


def test_importing_the_package_works():
    """The failure mode this guards against: a clone that cannot be imported."""
    import importlib

    for module in ("tradingbot", "tradingbot.data.schema", "tradingbot.data.providers",
                   "tradingbot.data.dhan", "tradingbot.data.universe",
                   "tradingbot.workers.coordinator", "tradingbot.web.app"):
        importlib.import_module(module)


def test_runtime_artifacts_are_ignored(git_available):
    """The ignore rules must still keep the database and run state out of git."""
    if not git_available:
        pytest.skip("git not available")

    for relpath in ("runs/paper.sqlite3", "data/prices.csv", "profiles.json"):
        result = subprocess.run(["git", "check-ignore", "-q", relpath], cwd=REPO_ROOT)
        assert result.returncode == 0, f"{relpath} should be gitignored but is not"
