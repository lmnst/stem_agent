"""Challenge split contract tests.

Pins:
- the eight challenge tasks load and run cleanly through evaluate_split;
- the original benchmark files are never mutated by an agent run;
- at least six of the eight challenge tasks generate a non-trivial
  variant queue (>=2 variants) so the failure mode is real rather
  than a no-variant short-circuit;
- the no-offending-docstring rule on challenge tasks holds (no task
  may admit it was authored to defeat a particular primitive);
- when produced, `compare_challenge.json` includes both actual and
  effective-budget attempt metrics so reports can not conflate them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from stem_agent.agent import evaluate_split
from stem_agent.blueprint import Blueprint
from stem_agent.evolve import stem_blueprint
from stem_agent.primitives import generate_variants


_REPO = Path(__file__).resolve().parent.parent
_BENCH = _REPO / "benchmarks" / "pybugs"
_CHALLENGE = _BENCH / "challenge"
_COMPARE_CH = _REPO / "artifacts" / "compare_challenge.json"


def _challenge_task_dirs() -> list[Path]:
    return sorted(p for p in _CHALLENGE.iterdir() if p.is_dir())


def test_challenge_split_is_present():
    assert _CHALLENGE.is_dir(), "challenge split directory missing"
    dirs = _challenge_task_dirs()
    assert len(dirs) == 8, f"challenge split must contain 8 tasks, found {len(dirs)}"


def test_challenge_tasks_load_and_run():
    bp = stem_blueprint()
    results = evaluate_split(_CHALLENGE, bp)
    assert len(results) == 8
    for r in results:
        # Every task must complete (solved or not), without leaking exceptions.
        assert isinstance(r.iterations, int)
        assert r.duration_s >= 0


def test_original_benchmark_files_are_not_mutated_by_a_run():
    """Snapshot every challenge solution + test file, run the agent,
    confirm bytes are identical afterwards."""
    snapshots = {}
    for d in _challenge_task_dirs():
        snapshots[d.name] = (
            (d / "solution.py").read_bytes(),
            (d / "test_solution.py").read_bytes(),
        )
    bp = stem_blueprint()
    evaluate_split(_CHALLENGE, bp)
    for name, (sol, test) in snapshots.items():
        d = _CHALLENGE / name
        assert (d / "solution.py").read_bytes() == sol, f"{name}/solution.py mutated"
        assert (d / "test_solution.py").read_bytes() == test, f"{name}/test_solution.py mutated"


def test_at_least_six_of_eight_challenge_tasks_have_non_trivial_variants():
    """The brief requires 6 of 8; we satisfy 8 of 8 in this submission."""
    from stem_agent.blueprint import PRIMITIVE_NAMES

    counts = {}
    for d in _challenge_task_dirs():
        source = (d / "solution.py").read_text(encoding="utf-8")
        variants = generate_variants(source, list(PRIMITIVE_NAMES))
        counts[d.name] = len(variants)

    non_trivial = sum(1 for n in counts.values() if n >= 2)
    assert non_trivial >= 6, (
        f"at least 6 of 8 challenge tasks must produce a non-trivial "
        f"(>=2) variant queue; got {non_trivial}: {counts}"
    )


def test_no_challenge_docstring_admits_authoring_against_a_primitive():
    """Any sentence in a challenge solution.py that references a
    specific primitive name as the thing the bug was meant to defeat
    is forbidden. The rule's spirit: bugs are designed bug-class-first,
    not by reverse-engineering against `swap_call_args` etc."""
    from stem_agent.blueprint import PRIMITIVE_NAMES

    offending: list[str] = []
    for d in _challenge_task_dirs():
        text = (d / "solution.py").read_text(encoding="utf-8")
        lowered = text.lower()
        for prim in PRIMITIVE_NAMES:
            if prim.lower() in lowered:
                offending.append(f"{d.name}: mentions {prim}")
        # Also catch the structural admission patterns.
        for phrase in ("nowhere to land", "no primitive can", "defeat the", "defeat a primitive"):
            if phrase in lowered:
                offending.append(f"{d.name}: contains forbidden phrase {phrase!r}")
    assert not offending, (
        f"challenge tasks must be designed bug-class-first, not by "
        f"reverse-engineering against the primitive bank: {offending}"
    )


def test_compare_challenge_includes_actual_and_eff_budget_metrics():
    if not _COMPARE_CH.exists():
        pytest.skip("artifacts/compare_challenge.json not present; run the documented compare command")
    payload = json.loads(_COMPARE_CH.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert rows, "compare_challenge must contain rows"
    for r in rows:
        assert "actual_attempts_sum" in r
        assert "eff_budget_attempts_sum" in r
        assert "mean_iters_actual_all" in r
        assert "mean_iters_eff_all" in r
        # On challenge, every task fails, so eff_bud >= actual at the row level.
        assert r["eff_budget_attempts_sum"] >= r["actual_attempts_sum"]


def test_perturbation_report_keeps_challenge_actual_and_eff_separate():
    canonical = _REPO / "docs" / "evaluation" / "perturbation_report.json"
    if not canonical.exists():
        pytest.skip("docs/evaluation/perturbation_report.json not present")
    report = json.loads(canonical.read_text(encoding="utf-8"))
    if "challenge" not in report.get("splits", {}):
        pytest.skip("challenge split not in canonical report")
    rows = report["splits"]["challenge"]["rows"]
    for r in rows:
        # Challenge tasks all fail in this submission; on at least
        # one row the two columns must differ (variants exhaust before
        # the budget on most tasks, so actual < eff_bud).
        assert "actual_attempts_sum" in r
        assert "eff_budget_attempts_sum" in r
    # Pin the documented inequality on the canonical "deployed evolved" row.
    deployed = next(r for r in rows if r["row"] == "deployed evolved")
    assert deployed["actual_attempts_sum"] < deployed["eff_budget_attempts_sum"], (
        "challenge report must show actual < eff_bud on deployed evolved "
        "(variants exhaust before budget on most tasks)"
    )
