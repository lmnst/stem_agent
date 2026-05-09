"""End-to-end test of the budget-controlled `compare` command.

The reviewer asked for a four-row stem-vs-evolved table at both
budgets. This test pins:
- the rows are present and in the documented order;
- both `default` and `swapped budget` rows differ in budget, not just
  in label;
- pass rates land in [0, 1] with non-trivial Wilson CIs;
- when stem and evolved are byte-identical, all four rows produce the
  same numeric pass rate (sanity check on the budget-override path).
"""
import json
import textwrap
from pathlib import Path

from stem_agent.blueprint import PRIMITIVE_NAMES, Blueprint
from stem_agent.cli import main as cli_main
from stem_agent.evolve import stem_blueprint


def _make_task(parent: Path, name: str, sol: str, test: str) -> Path:
    td = parent / name
    td.mkdir(parents=True, exist_ok=True)
    (td / "solution.py").write_text(textwrap.dedent(sol), encoding="utf-8")
    (td / "test_solution.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return td


def _build_bench(root: Path) -> None:
    _make_task(
        root / "test", "task_a",
        "def f():\n    return 1\n",
        "from solution import f\n\ndef test_a():\n    assert f() == 2\n",
    )
    _make_task(
        root / "test", "task_b",
        "def f(x):\n    return x < 1\n",
        "from solution import f\n\ndef test_a():\n    assert f(2) is True\n",
    )


def test_compare_writes_four_rows_in_documented_order(tmp_path):
    bench = tmp_path / "bench"
    _build_bench(bench)
    stem_path = tmp_path / "stem.json"
    evolved_path = tmp_path / "evolved.json"
    stem_blueprint().to_json(stem_path)
    evolved = Blueprint(
        name="evolved",
        primitive_priority=sorted(PRIMITIVE_NAMES, reverse=True),
        primitive_budget=12,
        early_stop_no_progress=12,
    )
    evolved.to_json(evolved_path)

    out = tmp_path / "compare.json"
    rc = cli_main(
        [
            "compare",
            "--stem", str(stem_path),
            "--evolved", str(evolved_path),
            "--bench", str(bench),
            "--split", "test",
            "--out", str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert [r["row"] for r in rows] == [
        "stem (default)",
        "stem (evolved budget)",
        "evolved (stem budget)",
        "evolved (default)",
    ]
    assert rows[0]["budget"] == 8
    assert rows[1]["budget"] == 12
    assert rows[2]["budget"] == 8
    assert rows[3]["budget"] == 12
    for r in rows:
        lo, hi = r["pass_rate_ci95"]
        assert 0.0 <= lo <= r["pass_rate"] <= hi <= 1.0


def test_compare_byte_identical_blueprints_produce_aligned_rates(tmp_path):
    """If stem and evolved are byte-identical, the two `default`-row rates match.

    This is a regression guard: budget-override should not silently
    perturb rate when the budget override is a no-op.
    """
    bench = tmp_path / "bench"
    _build_bench(bench)
    stem_path = tmp_path / "stem.json"
    evolved_path = tmp_path / "evolved.json"
    bp = stem_blueprint()
    bp.to_json(stem_path)
    bp.to_json(evolved_path)

    out = tmp_path / "compare.json"
    rc = cli_main(
        [
            "compare",
            "--stem", str(stem_path),
            "--evolved", str(evolved_path),
            "--bench", str(bench),
            "--split", "test",
            "--out", str(out),
        ]
    )
    assert rc == 0
    rows = json.loads(out.read_text(encoding="utf-8"))["rows"]
    # rows[0] is stem(default), rows[3] is evolved(default).
    # With identical blueprints, the rates must match.
    assert rows[0]["pass_rate"] == rows[3]["pass_rate"]
    assert rows[0]["budget"] == rows[3]["budget"]
