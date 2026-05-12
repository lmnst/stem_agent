"""Failure-mode tests for solve_task.

Verifies that termination notes correctly distinguish:

- `no variants`         when the source has a syntax error so
                        `generate_variants` returns empty;
- `variants exhausted`  when the variant queue is small and the budget
                        is large, and every variant still misses;

and that a malformed `test_solution.py` does not mutate the original
benchmark dir (the workspace copy is the only thing the agent ever
writes to).
"""
from pathlib import Path

from blueprint_repair.agent import solve_task
from blueprint_repair.evolve import stem_blueprint


def _write_task(td: Path, sol: str, test_src: str) -> Path:
    td.mkdir(parents=True, exist_ok=True)
    (td / "solution.py").write_text(sol, encoding="utf-8")
    (td / "test_solution.py").write_text(test_src, encoding="utf-8")
    return td


def test_syntax_error_in_source_yields_no_variants(tmp_path):
    td = _write_task(
        tmp_path / "task",
        "def f(:\n    return 1\n",  # malformed signature
        "from solution import f\n\ndef test_a():\n    assert f() == 1\n",
    )
    res = solve_task(td, stem_blueprint(), task_id="task")
    assert res.solved is False
    assert res.iterations == 0
    assert res.note == "no variants"


def test_small_variant_queue_exhausted_before_budget(tmp_path):
    """Source admits only one variant; budget is 8; that variant misses."""
    td = _write_task(
        tmp_path / "task",
        "def f():\n    return True\n",  # only swap_true_false applies
        "from solution import f\n\ndef test_a():\n    assert f() == 'banana'\n",
    )
    res = solve_task(td, stem_blueprint(), task_id="task")
    assert res.solved is False
    assert res.iterations == 1
    assert res.note == "variants exhausted"


def test_broken_test_does_not_mutate_task_dir(tmp_path):
    """A malformed `test_solution.py` must not leak edits onto the benchmark."""
    src = "def f():\n    return 1\n"
    test_src = "from solution import f\n\ndef test_a(:\n    pass\n"  # syntax err
    td = _write_task(tmp_path / "task", src, test_src)
    solve_task(td, stem_blueprint(), task_id="task")
    assert (td / "solution.py").read_text(encoding="utf-8") == src
    assert (td / "test_solution.py").read_text(encoding="utf-8") == test_src


def test_already_passing_note(tmp_path):
    td = _write_task(
        tmp_path / "task",
        "def f():\n    return 1\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 1\n",
    )
    res = solve_task(td, stem_blueprint(), task_id="task")
    assert res.solved is True
    assert res.iterations == 0
    assert res.note == "already passing"


def test_budget_exhausted_note(tmp_path):
    """Source admits many variants; tight budget runs out without a fix."""
    src = (
        "def f(x):\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a + b + c + x\n"
    )
    test_src = (
        "from solution import f\n\n"
        "def test_a():\n"
        "    assert f(0) == 'never'\n"  # impossible to satisfy via primitive
    )
    td = _write_task(tmp_path / "task", src, test_src)
    from blueprint_repair.blueprint import Blueprint, PRIMITIVE_NAMES
    bp = Blueprint(
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=2,
        early_stop_no_progress=999,  # high so it does not fire first
    )
    res = solve_task(td, bp, task_id="task")
    assert res.solved is False
    assert res.iterations == 2
    assert res.note == "budget exhausted"
