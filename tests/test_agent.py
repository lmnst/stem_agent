import textwrap
from pathlib import Path

import pytest

from blueprint_repair.agent import solve_task
from blueprint_repair.blueprint import PRIMITIVE_NAMES, Blueprint
from blueprint_repair.evolve import stem_blueprint


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
    assert res.note == "already passing"


def test_priority_change_alters_pass_at_tight_budget(tmp_path):
    """Reordering primitive_priority must change pass/fail at tight budget.

    Source has only a `<` and only one constant; budget=1. With
    `swap_compare_strict` first the fix lands on attempt 1; with the
    primitive at the tail of priority it never runs.
    """
    td = _make_task(
        tmp_path,
        "task",
        """\
        def upto(n):
            i = 1
            out = []
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
    cmp_first = ["swap_compare_strict"] + [
        p for p in PRIMITIVE_NAMES if p != "swap_compare_strict"
    ]
    cmp_last = [p for p in PRIMITIVE_NAMES if p != "swap_compare_strict"] + [
        "swap_compare_strict"
    ]

    bp_first = Blueprint(
        name="first", primitive_priority=cmp_first, primitive_budget=1, early_stop_no_progress=1
    )
    bp_last = Blueprint(
        name="last", primitive_priority=cmp_last, primitive_budget=1, early_stop_no_progress=1
    )
    res_first = solve_task(td, bp_first, task_id="task")
    res_last = solve_task(td, bp_last, task_id="task")

    assert res_first.solved is True
    assert res_first.iterations == 1
    assert res_last.solved is False


def test_invalid_workflow_raises(tmp_path):
    td = _make_task(
        tmp_path,
        "task",
        "def f():\n    return 1\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 1\n",
    )
    bp = Blueprint(workflow=["run_tests", "do_magic", "apply_check"])
    with pytest.raises(ValueError):
        solve_task(td, bp, task_id="task")


def test_solve_records_effective_budget(tmp_path):
    td = _make_task(
        tmp_path,
        "task",
        "def f():\n    return 1\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 2\n",
    )
    bp = Blueprint(
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=5,
        early_stop_no_progress=5,
    )
    res = solve_task(td, bp, task_id="task")
    assert res.effective_budget == 5
