import textwrap
from pathlib import Path

from blueprint_repair.runner import read_solution, run_tests, task_workspace, write_solution


def _make_task(parent: Path, sol: str, test: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "solution.py").write_text(textwrap.dedent(sol), encoding="utf-8")
    (parent / "test_solution.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return parent


def test_runner_pass(tmp_path):
    td = _make_task(
        tmp_path / "task",
        """\
        def add(a, b):
            return a + b
        """,
        """\
        from solution import add


        def test_basic():
            assert add(2, 3) == 5
        """,
    )
    with task_workspace(td) as ws:
        r = run_tests(ws)
    assert r.passed
    assert r.exit_code == 0


def test_runner_fail(tmp_path):
    td = _make_task(
        tmp_path / "task",
        """\
        def add(a, b):
            return a - b
        """,
        """\
        from solution import add


        def test_basic():
            assert add(2, 3) == 5
        """,
    )
    with task_workspace(td) as ws:
        r = run_tests(ws)
    assert not r.passed
    assert r.exit_code != 0


def test_runner_timeout(tmp_path):
    td = _make_task(
        tmp_path / "task",
        """\
        def hang():
            while True:
                pass
        """,
        """\
        from solution import hang


        def test_basic():
            hang()
        """,
    )
    with task_workspace(td) as ws:
        r = run_tests(ws, timeout_s=2.0)
    assert not r.passed
    assert "TIMEOUT" in r.stderr


def test_runner_does_not_mutate_task_dir(tmp_path):
    sol = "def add(a, b):\n    return a + b\n"
    test = "from solution import add\n\n\ndef test_basic():\n    assert add(1, 1) == 2\n"
    td = tmp_path / "task"
    td.mkdir()
    (td / "solution.py").write_text(sol)
    (td / "test_solution.py").write_text(test)
    with task_workspace(td) as ws:
        write_solution(ws, "def add(a, b):\n    return 0\n")
        run_tests(ws)
        assert read_solution(ws) != sol
    assert (td / "solution.py").read_text() == sol


def test_workspace_missing_files_raises(tmp_path):
    td = tmp_path / "task"
    td.mkdir()
    (td / "solution.py").write_text("x = 1\n")
    try:
        with task_workspace(td) as _:
            pass
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
