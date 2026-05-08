import textwrap
from pathlib import Path

import pytest

from stem_agent.agent import solve_task
from stem_agent.blueprint import PRIMITIVE_NAMES, Blueprint
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
    assert res.note == "already passing"


def test_workflow_change_alters_behavior(tmp_path):
    """Adding `localize` to the workflow changes pass/fail at a tight budget.

    Source has constants on two lines; the bug raises an IndexError on
    the line whose fix lives at the tail of the natural variant order.
    With the default workflow, the budget is consumed mutating the
    irrelevant line. With `localize` first, the soft prior bumps the
    bug line's variants ahead and the fix is found inside budget.
    """
    td = _make_task(
        tmp_path,
        "task",
        """\
        def f(x):
            items = [10, 20, 30]
            return items[x - 5]
        """,
        """\
        from solution import f


        def test_basic():
            assert f(8) == 30
        """,
    )

    bp_plain = Blueprint(
        name="plain",
        workflow=["run_tests", "propose", "apply_check"],
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=2,
        early_stop_no_progress=2,
    )
    bp_loc = Blueprint(
        name="loc",
        workflow=["run_tests", "localize", "propose", "apply_check"],
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=2,
        early_stop_no_progress=2,
    )

    res_plain = solve_task(td, bp_plain, task_id="task")
    res_loc = solve_task(td, bp_loc, task_id="task")

    assert res_plain.solved is False, "plain workflow should run out of budget"
    assert res_loc.solved is True, "localize workflow should fit the fix in budget"
    assert res_loc.iterations <= 2


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
