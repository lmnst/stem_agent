"""Tests for the perturbation report.

Pins:
- the report contains all seven required rows in the documented order;
- the random-policy row uses a fixed seed and is reproducible across runs;
- the zero-policy row is reproducible (the deployed strategy is itself
  reproducible, and zero-policy mirrors it on path B);
- the reverse-only row is represented explicitly;
- actual attempts and effective-budget attempts are computed separately;
- the deployed blueprint is path-B compliant: it carries no policy fields;
- the canonical committed report is regenerable from the documented command;
- strategy selection in evolve is deterministic across two consecutive runs.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from blueprint_repair.analysis import analyze_domain
from blueprint_repair.blueprint import PRIMITIVE_NAMES, Blueprint
from blueprint_repair.cli import main as cli_main
from blueprint_repair.evolve import evolve, stem_blueprint
from blueprint_repair.perturb import (
    PerturbConfig,
    build_report,
    random_policy_weights,
)


_REQUIRED_ROW_ORDER = [
    "stem default",
    "stem evolved budget",
    "deployed evolved",
    "zero policy",
    "random policy",
    "reverse only",
    "policy only",
]


def _make_task(parent: Path, name: str, sol: str, test: str) -> Path:
    td = parent / name
    td.mkdir(parents=True, exist_ok=True)
    (td / "solution.py").write_text(textwrap.dedent(sol), encoding="utf-8")
    (td / "test_solution.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return td


def _build_mini_bench(root: Path) -> None:
    """Tiny bench: 2 train, 2 dev, 2 test, 2 challenge tasks."""
    _make_task(
        root / "train", "task_a",
        "def f():\n    return 5\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 6\n",
    )
    _make_task(
        root / "train", "task_b",
        "def f(x):\n    return x < 1\n",
        "from solution import f\n\ndef test_a():\n    assert f(2) is True\n",
    )
    _make_task(
        root / "dev", "task_c",
        "def f():\n    return 9\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 10\n",
    )
    _make_task(
        root / "dev", "task_d",
        "def f(x):\n    return x == 0\n",
        "from solution import f\n\ndef test_a():\n    assert f(1) is True\n",
    )
    _make_task(
        root / "test", "task_e",
        "def f():\n    return 1\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 2\n",
    )
    _make_task(
        root / "test", "task_f",
        "def f(x):\n    return x > 1\n",
        "from solution import f\n\ndef test_a():\n    assert f(0) is True\n",
    )
    _make_task(
        root / "challenge", "ch_a",
        "def f():\n    return 'cant fix this'\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 'right'\n",
    )
    _make_task(
        root / "challenge", "ch_b",
        "def f(xs):\n    return xs[:]\n",
        "from solution import f\n\ndef test_a():\n    assert f([1,2,3]) == [3,2,1]\n",
    )


def _build_blueprints(tmp_path: Path) -> tuple[Path, Path, Path]:
    bench = tmp_path / "bench"
    _build_mini_bench(bench)
    out = tmp_path / "art"
    out.mkdir()
    profile, _ = analyze_domain(bench / "train")
    best, _ = evolve(profile, bench / "dev", max_generations=2, patience=2)
    stem_path = out / "stem.json"
    evolved_path = out / "evolved.json"
    stem_blueprint().to_json(stem_path)
    best.to_json(evolved_path)
    return bench, stem_path, evolved_path


# ------------------------- structural pins -----------------------------


def test_report_contains_all_required_rows_in_order(tmp_path):
    bench, stem_path, evolved_path = _build_blueprints(tmp_path)
    config = PerturbConfig(
        bench=bench,
        stem_blueprint=Blueprint.from_json(stem_path),
        deployed_blueprint=Blueprint.from_json(evolved_path),
        splits=("test",),
        random_seed=1234,
    )
    report = build_report(config)
    assert report["row_order"] == _REQUIRED_ROW_ORDER
    rows = report["splits"]["test"]["rows"]
    assert [r["row"] for r in rows] == _REQUIRED_ROW_ORDER


def test_report_runs_on_challenge_split(tmp_path):
    bench, stem_path, evolved_path = _build_blueprints(tmp_path)
    config = PerturbConfig(
        bench=bench,
        stem_blueprint=Blueprint.from_json(stem_path),
        deployed_blueprint=Blueprint.from_json(evolved_path),
        splits=("test", "challenge"),
        random_seed=1234,
    )
    report = build_report(config)
    assert "challenge" in report["splits"]
    rows = report["splits"]["challenge"]["rows"]
    assert [r["row"] for r in rows] == _REQUIRED_ROW_ORDER


def test_report_columns_separate_actual_from_eff_budget(tmp_path):
    bench, stem_path, evolved_path = _build_blueprints(tmp_path)
    config = PerturbConfig(
        bench=bench,
        stem_blueprint=Blueprint.from_json(stem_path),
        deployed_blueprint=Blueprint.from_json(evolved_path),
        splits=("test",),
        random_seed=1234,
    )
    report = build_report(config)
    for r in report["splits"]["test"]["rows"]:
        assert "actual_attempts_sum" in r
        assert "eff_budget_attempts_sum" in r
        assert "mean_iters_actual_all" in r
        assert "mean_iters_eff_all" in r
        assert "mean_iters_solved" in r
        # Actual attempts cannot exceed effective-budget attempts since
        # failures are charged at iters they ran (<= effective budget).
        assert r["actual_attempts_sum"] <= r["eff_budget_attempts_sum"]


# ------------------------- reproducibility pins ------------------------


def test_random_policy_weights_are_reproducible_for_fixed_seed():
    a = random_policy_weights(1234)
    b = random_policy_weights(1234)
    assert a == b
    c = random_policy_weights(5678)
    assert a != c
    assert set(a.keys()) == set(PRIMITIVE_NAMES)


def test_two_consecutive_perturb_runs_produce_byte_identical_report(tmp_path):
    bench, stem_path, evolved_path = _build_blueprints(tmp_path)
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    rc_a = cli_main([
        "perturb",
        "--stem", str(stem_path),
        "--evolved", str(evolved_path),
        "--bench", str(bench),
        "--splits", "test", "challenge",
        "--seed", "1234",
        "--out", str(out_a),
    ])
    rc_b = cli_main([
        "perturb",
        "--stem", str(stem_path),
        "--evolved", str(evolved_path),
        "--bench", str(bench),
        "--splits", "test", "challenge",
        "--seed", "1234",
        "--out", str(out_b),
    ])
    assert rc_a == 0 and rc_b == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def test_zero_policy_row_matches_deployed_when_path_b(tmp_path):
    """On path B the deployed blueprint has no policy, so `zero policy`
    is the same strategy as `deployed evolved`. Their pass and attempt
    columns must agree exactly."""
    bench, stem_path, evolved_path = _build_blueprints(tmp_path)
    config = PerturbConfig(
        bench=bench,
        stem_blueprint=Blueprint.from_json(stem_path),
        deployed_blueprint=Blueprint.from_json(evolved_path),
        splits=("test",),
        random_seed=1234,
    )
    report = build_report(config)
    rows = {r["row"]: r for r in report["splits"]["test"]["rows"]}
    for col in ("n_solved", "actual_attempts_sum", "eff_budget_attempts_sum"):
        assert rows["deployed evolved"][col] == rows["zero policy"][col]


def test_reverse_only_row_uses_reverse_priority(tmp_path):
    bench, stem_path, evolved_path = _build_blueprints(tmp_path)
    config = PerturbConfig(
        bench=bench,
        stem_blueprint=Blueprint.from_json(stem_path),
        deployed_blueprint=Blueprint.from_json(evolved_path),
        splits=("test",),
        random_seed=1234,
    )
    report = build_report(config)
    rows = {r["row"]: r for r in report["splits"]["test"]["rows"]}
    assert rows["reverse only"]["priority"] == sorted(PRIMITIVE_NAMES, reverse=True)
    assert rows["policy only"]["priority"] == sorted(PRIMITIVE_NAMES)
    assert rows["policy only"]["policy_active"] is True
    assert rows["zero policy"]["policy_active"] is False


# ------------------------- determinism pin -----------------------------


def test_strategy_selection_is_deterministic(tmp_path):
    """Two evolve runs from the same inputs must pick the same blueprint."""
    bench = tmp_path / "bench"
    _build_mini_bench(bench)

    profile_a, results_a = analyze_domain(bench / "train")
    bp_a, _ = evolve(profile_a, bench / "dev", max_generations=2, patience=2)
    profile_b, results_b = analyze_domain(bench / "train")
    bp_b, _ = evolve(profile_b, bench / "dev", max_generations=2, patience=2)
    assert bp_a.to_dict() == bp_b.to_dict()


# ------------------------- canonical report pin ------------------------


_CANONICAL_REPORT = Path(__file__).resolve().parent.parent / "docs" / "evaluation" / "perturbation_report.json"
_REPO_BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "pybugs"


@pytest.mark.skipif(
    not _CANONICAL_REPORT.exists() or not _REPO_BENCH.exists(),
    reason="canonical report or bench missing (regenerable via the documented command)",
)
def test_canonical_perturbation_report_is_regenerable(tmp_path):
    """Rerun the documented command and check the report matches what
    is committed under `docs/evaluation/perturbation_report.json`."""
    stem_path = tmp_path / "stem.json"
    evolved_path = tmp_path / "evolved.json"
    out_dir = tmp_path / "art"
    out_dir.mkdir()
    rc = cli_main([
        "evolve",
        "--bench", str(_REPO_BENCH),
        "--out", str(out_dir),
    ])
    assert rc == 0
    # the evolve command writes both stem and evolved blueprints
    stem_path.write_bytes((out_dir / "stem_blueprint.json").read_bytes())
    evolved_path.write_bytes((out_dir / "evolved_blueprint.json").read_bytes())

    out = tmp_path / "report.json"
    rc = cli_main([
        "perturb",
        "--stem", str(stem_path),
        "--evolved", str(evolved_path),
        "--bench", str(_REPO_BENCH),
        "--splits", "test", "challenge",
        "--seed", "1234",
        "--out", str(out),
    ])
    assert rc == 0
    assert out.read_bytes() == _CANONICAL_REPORT.read_bytes(), (
        "regenerated perturbation report differs from the committed copy "
        "under docs/evaluation/perturbation_report.json"
    )
