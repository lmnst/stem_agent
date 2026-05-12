"""Tests for the learned per-task primitive policy.

These pin the contract:
- features extracted from a known source match expectations;
- fitting on synthetic observations produces non-trivial weights only
  for primitives that actually fixed something;
- the per-task ranking changes when features change;
- `policy_priority` is deterministic under tied scores;
- the agent's runtime behaviour depends on `policy_weights` (when
  empty: global priority; when non-empty: per-task ordering kicks in
  and a low-confidence task hits `policy_fallback_budget`).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from blueprint_repair.agent import solve_task
from blueprint_repair.blueprint import PRIMITIVE_NAMES, Blueprint
from blueprint_repair.policy import (
    FEATURE_NAMES,
    extract_features,
    fit_fallback_budget_from_iters,
    fit_policy,
    fit_threshold,
    policy_priority,
    score_primitive,
    should_use_fallback_budget,
)


# ------------------------------ extract_features ----------------------------


def test_extract_features_counts_basic_nodes():
    src = textwrap.dedent(
        """
        def f(x):
            return x < 5 and x != 0
        """
    )
    feats = extract_features(src)
    assert feats["n_compare_strict_op"] == 1
    assert feats["n_compare_eq_op"] == 1
    assert feats["n_bool_op"] == 1
    assert feats["n_int_const"] == 2  # 5 and 0
    assert feats["n_int_const_large"] == 0
    assert feats["n_return"] == 1


def test_extract_features_marks_large_int_constants():
    feats = extract_features("def f():\n    return 100\n")
    assert feats["n_int_const"] == 1
    assert feats["n_int_const_large"] == 1


def test_extract_features_handles_syntax_error_safely():
    feats = extract_features("def f(:\n")
    assert feats == {f: 0 for f in FEATURE_NAMES}


def test_extract_features_counts_subscript_and_attribute():
    feats = extract_features("def f(xs):\n    return xs[0] + xs.size\n")
    assert feats["n_subscript"] == 1
    assert feats["n_attribute"] == 1


# ------------------------------ fit_policy ----------------------------------


def test_fit_policy_zero_observations_returns_zero_weights():
    weights = fit_policy([])
    for p in PRIMITIVE_NAMES:
        assert all(w == 0.0 for w in weights[p].values())


def test_fit_policy_assigns_positive_lift_to_characteristic_feature():
    """A primitive whose solves consistently have feature > 0 should
    get a positive weight on that feature (positive lift over the
    overall mean)."""
    obs = [
        ({"n_compare_strict_op": 2, "n_int_const": 0, "func_body_stmts": 1}, "swap_compare_strict"),
        ({"n_compare_strict_op": 1, "n_int_const": 0, "func_body_stmts": 1}, "swap_compare_strict"),
        ({"n_compare_strict_op": 0, "n_int_const": 1, "func_body_stmts": 1}, "shift_const_pm1"),
        ({"n_compare_strict_op": 0, "n_int_const": 1, "func_body_stmts": 1}, "shift_const_pm1"),
    ]
    weights = fit_policy(obs)
    assert weights["swap_compare_strict"]["n_compare_strict_op"] > 0
    assert weights["shift_const_pm1"]["n_int_const"] > 0
    # And cross-primitive: swap_compare_strict shouldn't claim n_int_const.
    assert weights["swap_compare_strict"]["n_int_const"] < 0


def test_fit_policy_unobserved_primitive_gets_zero_weights():
    """A primitive with no observed solves keeps an all-zero row.

    The other primitives' weights are non-trivial because each row's
    mean is differenced against the overall mean across all solves;
    here `shift_const_pm1` has high `n_int_const` while
    `swap_compare_strict` has high `n_compare_strict_op`, so each
    primitive picks up positive lift on its own characteristic
    feature.
    """
    obs = [
        ({"n_int_const": 1, "n_compare_strict_op": 0}, "shift_const_pm1"),
        ({"n_int_const": 0, "n_compare_strict_op": 1}, "swap_compare_strict"),
    ]
    weights = fit_policy(obs)
    assert all(v == 0.0 for v in weights["swap_call_args"].values())
    assert weights["shift_const_pm1"]["n_int_const"] > 0
    assert weights["swap_compare_strict"]["n_compare_strict_op"] > 0


# ------------------------------ score / priority -----------------------------


def test_score_primitive_is_dot_product_over_features():
    feats = {"a": 2, "b": 3, "c": 0}
    weights = {"a": 1.5, "b": -0.5, "c": 99.0, "d": 100.0}  # missing 'd' in features
    assert score_primitive(feats, weights) == pytest.approx(2 * 1.5 + 3 * -0.5 + 0 * 99.0 + 0 * 100.0)


def test_policy_priority_changes_with_features():
    """Two tasks with different features should yield different orderings."""
    weights = {
        p: {f: 0.0 for f in FEATURE_NAMES} for p in PRIMITIVE_NAMES
    }
    # swap_compare_strict prefers comparisons; swap_arith_pair prefers arith.
    weights["swap_compare_strict"]["n_compare_strict_op"] = 1.0
    weights["swap_arith_pair"]["n_arith_op"] = 1.0

    cmp_task = {f: 0 for f in FEATURE_NAMES}
    cmp_task["n_compare_strict_op"] = 3
    arith_task = {f: 0 for f in FEATURE_NAMES}
    arith_task["n_arith_op"] = 3

    pri_cmp = policy_priority(cmp_task, weights, fallback_priority=PRIMITIVE_NAMES)
    pri_arith = policy_priority(arith_task, weights, fallback_priority=PRIMITIVE_NAMES)
    assert pri_cmp[0] == "swap_compare_strict"
    assert pri_arith[0] == "swap_arith_pair"
    assert pri_cmp != pri_arith


def test_policy_priority_tiebreak_is_stable_to_fallback():
    """All scores tied at zero -> ordering equals the fallback priority."""
    weights = {p: {f: 0.0 for f in FEATURE_NAMES} for p in PRIMITIVE_NAMES}
    feats = {f: 0 for f in FEATURE_NAMES}
    fb = list(PRIMITIVE_NAMES)
    pri = policy_priority(feats, weights, fallback_priority=fb)
    assert pri == fb


# ------------------------------ threshold + fallback -------------------------


def test_should_use_fallback_with_empty_weights_is_inert():
    feats = {f: 0 for f in FEATURE_NAMES}
    assert should_use_fallback_budget(feats, {}, threshold=999.0) is False


def test_fit_threshold_returns_25th_percentile_score():
    weights = {p: {"a": 1.0, "b": 0.0} for p in PRIMITIVE_NAMES}
    obs = [
        ({"a": 1, "b": 0}, PRIMITIVE_NAMES[0]),
        ({"a": 2, "b": 0}, PRIMITIVE_NAMES[0]),
        ({"a": 3, "b": 0}, PRIMITIVE_NAMES[0]),
        ({"a": 4, "b": 0}, PRIMITIVE_NAMES[0]),
    ]
    th = fit_threshold(obs, weights)
    # Sorted scores: [1, 2, 3, 4]; 25th percentile by index len//4 = 1 -> 2.
    assert th == 2.0


def test_fit_fallback_budget_floors_at_two():
    assert fit_fallback_budget_from_iters([1]) == 2
    assert fit_fallback_budget_from_iters([]) == 2


def test_fit_fallback_budget_uses_median():
    assert fit_fallback_budget_from_iters([1, 2, 3, 5, 10]) == 3


# ------------------------------ agent integration ----------------------------


def _make_task(parent: Path, name: str, sol: str, test: str) -> Path:
    td = parent / name
    td.mkdir()
    (td / "solution.py").write_text(textwrap.dedent(sol), encoding="utf-8")
    (td / "test_solution.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return td


def test_agent_uses_per_task_ordering_when_policy_present(tmp_path):
    """With policy_weights active, the variant queue is reordered."""
    td = _make_task(
        tmp_path,
        "task",
        # Source has only a strict comparator and one int constant.
        # Without policy, alphabetical priority hits flip_compare first.
        # With policy weighted toward swap_compare_strict, swap_compare_strict
        # wins and the variant `x <= 1` lands first.
        """
        def f(x):
            return x < 1
        """,
        """
        from solution import f


        def test_basic():
            assert f(1) is True
        """,
    )

    weights = {p: {f: 0.0 for f in FEATURE_NAMES} for p in PRIMITIVE_NAMES}
    # Strongly favour swap_compare_strict on this task's feature shape.
    weights["swap_compare_strict"]["n_compare_strict_op"] = 10.0

    bp_with_policy = Blueprint(
        name="policy",
        primitive_priority=sorted(PRIMITIVE_NAMES),  # alpha order (flip_compare first)
        primitive_budget=1,
        early_stop_no_progress=1,
        policy_weights=weights,
        policy_confidence_threshold=0.0,  # never falls back
        policy_fallback_budget=0,
    )
    res_policy = solve_task(td, bp_with_policy, task_id="task")
    # Policy bumps swap_compare_strict to the front, so iter 1 is the fix.
    assert res_policy.solved is True
    assert res_policy.fixing_primitive == "swap_compare_strict"


def test_agent_uses_fallback_budget_on_low_confidence_task(tmp_path):
    """When the top primitive score is below threshold, the agent
    runs `policy_fallback_budget` instead of `primitive_budget`."""
    td = _make_task(
        tmp_path,
        "task",
        # Source has features whose top score will be 0 under the
        # policy below; the threshold is 0.5, so this task is
        # low-confidence and the budget is clamped to 1.
        "def f():\n    return 1\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 'unreachable'\n",
    )
    # Empty-ish weights: every score is 0.
    weights = {p: {f: 0.0 for f in FEATURE_NAMES} for p in PRIMITIVE_NAMES}
    bp = Blueprint(
        name="lowconf",
        primitive_priority=sorted(PRIMITIVE_NAMES),
        primitive_budget=8,
        early_stop_no_progress=999,
        policy_weights=weights,
        policy_confidence_threshold=0.5,  # 0 < 0.5 -> low confidence
        policy_fallback_budget=1,
    )
    res = solve_task(td, bp, task_id="task")
    assert res.solved is False
    assert res.policy_low_confidence is True
    assert res.effective_budget == 1
    assert "policy: low confidence" in res.note


def test_agent_no_policy_path_when_weights_empty(tmp_path):
    """An empty `policy_weights` dict must keep the runtime on the
    no-policy path (otherwise stem performance regresses)."""
    td = _make_task(
        tmp_path,
        "task",
        "def f(x):\n    return x < 1\n",
        "from solution import f\n\ndef test_a():\n    assert f(1) is True\n",
    )
    bp = Blueprint(
        name="nopolicy",
        primitive_priority=sorted(PRIMITIVE_NAMES),
        primitive_budget=8,
        early_stop_no_progress=8,
        policy_weights={},  # empty
        policy_confidence_threshold=999.0,  # would fire if weights existed
        policy_fallback_budget=1,
    )
    res = solve_task(td, bp, task_id="task")
    assert res.policy_low_confidence is False
    assert res.policy_top_score is None
    assert res.effective_budget == 8
    assert res.solved is True
