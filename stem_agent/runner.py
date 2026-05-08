"""Sandboxed test runner.

Each evaluation copies the task into a fresh temp directory and runs the
tests via the subprocess harness (`_harness.py`). The original benchmark
files are never mutated. A hard subprocess timeout protects against
runaway variants.

The test files are pytest-compatible; the harness just executes the
subset pytest uses (top-level `test_*` callables with plain `assert`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_HARNESS_PATH = Path(__file__).with_name("_harness.py")
_REQUIRED_FILES = ("solution.py", "test_solution.py")


@dataclass(frozen=True)
class TestResult:
    passed: bool
    exit_code: int
    duration_s: float
    stdout: str
    stderr: str


@contextmanager
def task_workspace(task_dir: Path) -> Iterator[Path]:
    """Copy a task into a fresh tempdir; yield the path; cleanup on exit."""
    task_dir = Path(task_dir)
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task dir not found: {task_dir}")
    for name in _REQUIRED_FILES:
        if not (task_dir / name).exists():
            raise FileNotFoundError(f"missing {name} in {task_dir}")
    tmp = Path(tempfile.mkdtemp(prefix="stem-"))
    try:
        for name in _REQUIRED_FILES:
            shutil.copy2(task_dir / name, tmp / name)
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_solution(workspace: Path, source: str) -> None:
    (Path(workspace) / "solution.py").write_text(source, encoding="utf-8")


def read_solution(workspace: Path) -> str:
    return (Path(workspace) / "solution.py").read_text(encoding="utf-8")


def run_tests(workspace: Path, timeout_s: float = 5.0) -> TestResult:
    workspace = Path(workspace)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(workspace) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, str(_HARNESS_PATH), str(workspace)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        return TestResult(
            passed=proc.returncode == 0,
            exit_code=proc.returncode,
            duration_s=elapsed,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - t0
        return TestResult(
            passed=False,
            exit_code=-1,
            duration_s=elapsed,
            stdout=e.stdout or "",
            stderr=(e.stderr or "") + "\nTIMEOUT",
        )
