"""End-to-end evolve test on an inline mini-benchmark.

Tests must run without network and without an API key.
"""
import textwrap
from pathlib import Path

from stem_agent.analysis import analyze_domain
from stem_agent.blueprint import PRIMITIVE_NAMES
from stem_agent.evolve import evolve, stem_blueprint, score_blueprint


def _make_task(parent: Path, name: str, sol: str, test: str) -> Path:
    td = parent / name
    td.mkdir(parents=True, exist_ok=True)
    (td / "solution.py").write_text(textwrap.dedent(sol), encoding="utf-8")
    (td / "test_solution.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return td


def _build_mini_bench(root: Path):
    # 2 train tasks, both fixable by shift_const_pm1
    _make_task(
        root / "train",
        "task_a",
        "def f():\n    return 5\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 6\n",
    )
    _make_task(
        root / "train",
        "task_b",
        "def f():\n    return 3\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 4\n",
    )
    # 1 dev task, also shift_const_pm1
    _make_task(
        root / "dev",
        "task_c",
        "def f():\n    return 9\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 10\n",
    )


def test_analyze_domain_picks_up_dominant_primitive(tmp_path):
    bench = tmp_path / "bench"
    _build_mini_bench(bench)
    profile, results = analyze_domain(bench / "train")
    assert profile.sample_size == 2
    assert all(r.solved for r in results)
    assert profile.primitive_frequencies.get("shift_const_pm1", 0) >= 0.5


def test_score_blueprint_offline(tmp_path):
    bench = tmp_path / "bench"
    _build_mini_bench(bench)
    cs = score_blueprint(stem_blueprint(), bench / "dev")
    assert cs.n_total == 1
    assert cs.n_solved == 1
    assert cs.pass_rate == 1.0


def test_evolve_returns_a_valid_blueprint(tmp_path):
    bench = tmp_path / "bench"
    _build_mini_bench(bench)
    profile, _ = analyze_domain(bench / "train")
    best, history = evolve(profile, bench / "dev", max_generations=1, patience=1)
    assert set(best.primitive_priority) == set(PRIMITIVE_NAMES)
    assert best.use_llm_proposal is False  # offline
    assert any(s.pass_rate == 1.0 for s in history)
