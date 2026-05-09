"""End-to-end tests over a small inline benchmark.

These tests assert the pipeline's invariants without depending on the
real `benchmarks/pybugs` corpus:

- Both `stem` and `evolved` blueprints are produced and named correctly.
- The held-out `test/` directory is never opened during analysis or
  evolution (pinned by monkeypatching `agent.task_workspace`).
- Two evaluations of the same blueprint produce identical per-task
  records (excluding wall-clock duration).
- The evolved blueprint differs from the stem on at least one field
  that `solve_task` actually consumes.
"""
from __future__ import annotations

import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import List

import pytest

from stem_agent import agent as agent_module
from stem_agent.agent import evaluate_split
from stem_agent.analysis import analyze_domain
from stem_agent.blueprint import Blueprint
from stem_agent.evolve import evolve, stem_blueprint


def _make_task(parent: Path, name: str, sol: str, test: str) -> Path:
    td = parent / name
    td.mkdir(parents=True, exist_ok=True)
    (td / "solution.py").write_text(textwrap.dedent(sol), encoding="utf-8")
    (td / "test_solution.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return td


def _build_bench(root: Path) -> None:
    """Build a tiny inline bench: 3 train, 2 dev, 2 test tasks."""
    _make_task(
        root / "train", "task_a",
        "def f():\n    return 5\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 6\n",
    )
    _make_task(
        root / "train", "task_b",
        "def f():\n    return 3\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 4\n",
    )
    _make_task(
        root / "train", "task_c",
        "def f(x):\n    return x < 1\n",
        "from solution import f\n\ndef test_a():\n    assert f(2) is True\n",
    )
    _make_task(
        root / "dev", "task_d",
        "def f():\n    return 9\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 10\n",
    )
    _make_task(
        root / "dev", "task_e",
        "def f(x):\n    return x == 0\n",
        "from solution import f\n\ndef test_a():\n    assert f(1) is True\n",
    )
    _make_task(
        root / "test", "task_f",
        "def f():\n    return 1\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 2\n",
    )
    _make_task(
        root / "test", "task_g",
        "def f(x):\n    return x > 1\n",
        "from solution import f\n\ndef test_a():\n    assert f(0) is True\n",
    )


def test_pipeline_produces_stem_and_evolved_blueprints(tmp_path):
    bench = tmp_path / "bench"
    _build_bench(bench)
    profile, _ = analyze_domain(bench / "train")
    best, _ = evolve(profile, bench / "dev", max_generations=1, patience=1)

    stem = stem_blueprint()
    assert stem.name == "stem"
    assert best.name == "evolved"
    # Lineage should record at least the candidate that produced the evolved bp.
    assert isinstance(best.lineage, list) and best.lineage


def test_test_split_is_never_opened_during_analysis_or_evolution(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    _build_bench(bench)
    test_dir = (bench / "test").resolve()

    seen: List[Path] = []
    real = agent_module.task_workspace

    @contextmanager
    def watching_task_workspace(task_dir):
        seen.append(Path(task_dir).resolve())
        with real(task_dir) as ws:
            yield ws

    monkeypatch.setattr(agent_module, "task_workspace", watching_task_workspace)

    profile, _ = analyze_domain(bench / "train")
    evolve(profile, bench / "dev", max_generations=2, patience=2)

    # Every workspace opened during analysis or evolution must live under
    # train/ or dev/ -- never under test/.
    test_str = str(test_dir)
    leaks = [p for p in seen if str(p).startswith(test_str)]
    assert not leaks, f"test split was opened during analysis/evolution: {leaks}"


def test_two_consecutive_runs_produce_identical_task_records(tmp_path):
    bench = tmp_path / "bench"
    _build_bench(bench)
    bp = stem_blueprint()
    a = evaluate_split(bench / "test", bp)
    b = evaluate_split(bench / "test", bp)
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.task_id == rb.task_id
        assert ra.solved == rb.solved
        assert ra.iterations == rb.iterations
        assert ra.fixing_primitive == rb.fixing_primitive
        assert ra.note == rb.note
        # duration_s deliberately not asserted (wall-clock varies)


def test_evolved_blueprint_differs_from_stem_on_agent_read_field(tmp_path):
    bench = tmp_path / "bench"
    _build_bench(bench)
    profile, train_results = analyze_domain(bench / "train")
    best, _ = evolve(
        profile,
        bench / "dev",
        train_dir=bench / "train",
        train_results=train_results,
        max_generations=2,
        patience=2,
    )
    stem = stem_blueprint()

    # Fields that solve_task actually reads. `name`, `description`, and
    # `lineage` are metadata; differing on those alone would be cosmetic.
    agent_fields = (
        "workflow",
        "primitive_priority",
        "primitive_budget",
        "early_stop_no_progress",
        "policy_weights",
        "policy_confidence_threshold",
        "policy_fallback_budget",
    )
    differences = [f for f in agent_fields if getattr(stem, f) != getattr(best, f)]
    assert differences, (
        "evolved must differ from stem on at least one agent-read field; "
        f"both had identical {agent_fields}"
    )


def test_evolved_carries_a_learned_artifact_absent_from_stem(tmp_path):
    """Acceptance pin: the evolved blueprint must contain a non-empty
    `policy_weights` dict, derived from train+dev observations,
    serialized in the JSON, and absent from the stem."""
    bench = tmp_path / "bench"
    _build_bench(bench)
    profile, train_results = analyze_domain(bench / "train")
    best, _ = evolve(
        profile,
        bench / "dev",
        train_dir=bench / "train",
        train_results=train_results,
        max_generations=1,
        patience=1,
    )
    stem = stem_blueprint()
    assert stem.policy_weights == {}
    assert best.policy_weights, "evolved must have learned policy weights"
    # Round-trip through JSON to confirm serialization works.
    p = tmp_path / "evolved.json"
    best.to_json(p)
    reloaded = Blueprint.from_json(p)
    assert reloaded.policy_weights == best.policy_weights
    assert reloaded.policy_confidence_threshold == best.policy_confidence_threshold
    assert reloaded.policy_fallback_budget == best.policy_fallback_budget


def test_two_consecutive_evolves_produce_byte_identical_blueprint(tmp_path):
    """L3: deterministic evolution. Same inputs => byte-identical output."""
    bench = tmp_path / "bench"
    _build_bench(bench)

    profile_a, results_a = analyze_domain(bench / "train")
    bp_a, _ = evolve(
        profile_a,
        bench / "dev",
        train_dir=bench / "train",
        train_results=results_a,
        max_generations=2,
        patience=2,
    )
    out_a = tmp_path / "a.json"
    bp_a.to_json(out_a)

    profile_b, results_b = analyze_domain(bench / "train")
    bp_b, _ = evolve(
        profile_b,
        bench / "dev",
        train_dir=bench / "train",
        train_results=results_b,
        max_generations=2,
        patience=2,
    )
    out_b = tmp_path / "b.json"
    bp_b.to_json(out_b)

    assert out_a.read_bytes() == out_b.read_bytes()
