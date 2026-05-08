"""Subprocess test harness used by `runner.run_tests`.

Imports solution.py and test_solution.py from the workspace directory
passed as argv[1], runs every `test_*` callable in the test module, and
exits 0 if all pass, 1 if any fail, 2 on import error.

This is deliberately minimal — no plugin system, no conftest discovery,
no rootdir search — so per-call wall time is dominated by Python
interpreter startup rather than pytest infrastructure. The benchmark
test files remain pytest-compatible; only the runtime differs.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: _harness.py <workspace>", file=sys.stderr)
        return 2
    ws = Path(sys.argv[1])
    if str(ws) not in sys.path:
        sys.path.insert(0, str(ws))
    try:
        _load_module("solution", ws / "solution.py")
        test_mod = _load_module("test_solution", ws / "test_solution.py")
    except Exception:
        traceback.print_exc()
        return 2

    failures = []
    for name in sorted(dir(test_mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(test_mod, name)
        if not callable(fn):
            continue
        try:
            fn()
        except BaseException:  # includes AssertionError, KeyboardInterrupt
            failures.append((name, traceback.format_exc()))

    if failures:
        for name, tb in failures:
            print(f"FAIL solution.py::{name}")
            print(tb)
        return 1
    print(f"OK {len([n for n in dir(test_mod) if n.startswith('test_')])} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
