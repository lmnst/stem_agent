import textwrap
from pathlib import Path

from stem_agent.agent import solve_task
from stem_agent.evolve import stem_blueprint


def _make_task(parent: Path, name: str, sol: str, test: str) -> Path:
    td = parent / name
    td.mkdir()
    (td / "solution.py").write_text(textwrap.dedent(sol), encoding="utf-8")
    (td / "test_solution.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return td


def test_solve_simple_compare_strict_bug(tmp_path):
    td = _make_task(
        tmp_path,
        "task",
        """\
        def upto(n):
            out = []
            i = 1
            while i < n:
                out.append(i)
                i += 1
            return out
        """,
        """\
        from solution import upto


        def test_basic():
            assert upto(3) == [1, 2, 3]
        """,
    )
    bp = stem_blueprint()
    res = solve_task(td, bp, task_id="task")
    assert res.solved is True
    assert res.fixing_primitive == "swap_compare_strict"
    assert res.iterations >= 1


def test_solve_unsolvable(tmp_path):
    td = _make_task(
        tmp_path,
        "task",
        """\
        def f():
            return "wrong"
        """,
        """\
        from solution import f


        def test_basic():
            assert f() == "right"
        """,
    )
    bp = stem_blueprint()
    res = solve_task(td, bp, task_id="task")
    assert res.solved is False
    assert res.fixing_primitive is None


def test_solve_already_passing(tmp_path):
    td = _make_task(
        tmp_path,
        "task",
        """\
        def add(a, b):
            return a + b
        """,
        """\
        from solution import add


        def test_basic():
            assert add(1, 1) == 2
        """,
    )
    res = solve_task(td, stem_blueprint(), task_id="task")
    assert res.solved is True
    assert res.iterations == 0
