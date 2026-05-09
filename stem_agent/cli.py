"""Command-line entry point: `stem evolve | eval | solve | compare`."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .agent import SolveResult, evaluate_split, solve_task
from .analysis import analyze_domain
from .blueprint import Blueprint
from .evolve import evolve, stem_blueprint
from .stats import wilson_interval


def _summarize(results: List[SolveResult], *, budget: Optional[int] = None) -> dict:
    """Build a per-blueprint summary with both iter-mean denominators.

    `mean_iters_solved` is the historical metric: averaged only over
    tasks the blueprint solved. `mean_iters_all_with_failed_at_budget`
    counts unsolved tasks at the effective per-task budget instead, so
    a blueprint that solves more tasks isn't unfairly penalized for
    having to count its harder solves while a worse blueprint averages
    only its easy ones. We report both.
    """
    n = len(results)
    n_solved = sum(1 for r in results if r.solved)
    iters_solved = [r.iterations for r in results if r.solved]
    iters_all = [
        r.iterations if r.solved else (r.effective_budget or budget or 0)
        for r in results
    ]
    durations = [r.duration_s for r in results]
    lo, hi = wilson_interval(n_solved, n)
    return {
        "n_tasks": n,
        "n_solved": n_solved,
        "pass_rate": (n_solved / n) if n else 0.0,
        "pass_rate_ci95": [lo, hi],
        "mean_iters_solved": (sum(iters_solved) / len(iters_solved)) if iters_solved else 0.0,
        "mean_iters_all_with_failed_at_budget": (sum(iters_all) / n) if n else 0.0,
        "mean_wall_s": (sum(durations) / n) if n else 0.0,
        "tasks": [
            {
                "id": r.task_id,
                "solved": r.solved,
                "iters": r.iterations,
                "effective_budget": r.effective_budget,
                "duration_s": round(r.duration_s, 3),
                "fix": r.fixing_primitive,
                "note": r.note,
            }
            for r in results
        ],
    }


def cmd_evolve(args: argparse.Namespace) -> int:
    bench = Path(args.bench)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] domain analysis (probe over train)", file=sys.stderr)
    profile, _ = analyze_domain(bench / "train")
    profile.to_json(out_dir / "profile.json")
    print(f"  primitive_frequencies: {profile.primitive_frequencies}", file=sys.stderr)
    print(f"  recommended_budget:    {profile.recommended_budget}", file=sys.stderr)

    print("[2/3] evolving on dev split", file=sys.stderr)
    best_bp, history = evolve(
        profile,
        bench / "dev",
        max_generations=args.max_generations,
        patience=args.patience,
    )
    best_bp.to_json(out_dir / "evolved_blueprint.json")

    history_dump = [
        {
            "generation": s.generation,
            "name": s.blueprint.name,
            "lineage": list(s.blueprint.lineage),
            "blueprint": s.blueprint.to_dict(),
            "pass_rate": s.pass_rate,
            # math.inf marks "no solves"; keep the JSON valid by writing null.
            "mean_iters": None if math.isinf(s.mean_iters) else s.mean_iters,
            "n_solved": s.n_solved,
            "n_total": s.n_total,
        }
        for s in history
    ]
    (out_dir / "evolution_log.json").write_text(
        json.dumps(history_dump, indent=2), encoding="utf-8"
    )

    print("[3/3] saving stem (baseline) blueprint", file=sys.stderr)
    stem_blueprint().to_json(out_dir / "stem_blueprint.json")

    print(f"evolved blueprint -> {out_dir / 'evolved_blueprint.json'}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    bp = Blueprint.from_json(Path(args.blueprint))
    bench = Path(args.bench)
    split_dir = bench / args.split
    results = evaluate_split(split_dir, bp)
    summary = _summarize(results)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=2))
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    bp = Blueprint.from_json(Path(args.blueprint))
    res = solve_task(Path(args.task), bp)
    print(
        json.dumps(
            {
                "task": res.task_id,
                "solved": res.solved,
                "iters": res.iterations,
                "effective_budget": res.effective_budget,
                "duration_s": round(res.duration_s, 3),
                "fix": res.fixing_primitive,
                "note": res.note,
            },
            indent=2,
        )
    )
    return 0 if res.solved else 1


def _row(label: str, summary: dict, budget: int) -> Dict[str, object]:
    lo, hi = summary["pass_rate_ci95"]
    return {
        "row": label,
        "budget": budget,
        "n_solved": summary["n_solved"],
        "n_tasks": summary["n_tasks"],
        "pass_rate": summary["pass_rate"],
        "pass_rate_ci95": [lo, hi],
        "mean_iters_solved": summary["mean_iters_solved"],
        "mean_iters_all_with_failed_at_budget": summary["mean_iters_all_with_failed_at_budget"],
        "mean_wall_s": summary["mean_wall_s"],
    }


def _ascii_table(rows: List[Dict[str, object]]) -> str:
    """Render the budget-controlled comparison as a fixed-width table."""
    header = (
        f"{'row':<24} {'budget':>6} {'pass':<14} {'CI95':<22} "
        f"{'iters/solved':>13} {'iters/all*':>11} {'wall_s':>8}"
    )
    out = [header, "-" * len(header)]
    for r in rows:
        lo, hi = r["pass_rate_ci95"]
        pass_str = f"{r['n_solved']}/{r['n_tasks']} ({r['pass_rate'] * 100:.1f}%)"
        ci_str = f"[{lo * 100:.1f}, {hi * 100:.1f}]"
        out.append(
            f"{r['row']:<24} {r['budget']:>6d} {pass_str:<14} {ci_str:<22} "
            f"{r['mean_iters_solved']:>13.2f} "
            f"{r['mean_iters_all_with_failed_at_budget']:>11.2f} "
            f"{r['mean_wall_s']:>8.2f}"
        )
    out.append("* iters/all counts unsolved tasks at their effective budget.")
    return "\n".join(out)


def _override_budget(bp: Blueprint, budget: int) -> Blueprint:
    """Clone a blueprint with the budget (and matching early-stop) overridden."""
    return Blueprint.from_dict(
        {**bp.to_dict(), "primitive_budget": budget, "early_stop_no_progress": budget}
    )


def cmd_compare(args: argparse.Namespace) -> int:
    """Four-row budget-controlled comparison of stem vs evolved."""
    stem_bp = Blueprint.from_json(Path(args.stem))
    evolved_bp = Blueprint.from_json(Path(args.evolved))
    bench = Path(args.bench)
    split_dir = bench / args.split

    stem_budget = stem_bp.primitive_budget
    evolved_budget = evolved_bp.primitive_budget

    pairs: List[Tuple[str, Blueprint, int]] = [
        ("stem (default)", stem_bp, stem_budget),
        ("stem (evolved budget)", _override_budget(stem_bp, evolved_budget), evolved_budget),
        ("evolved (stem budget)", _override_budget(evolved_bp, stem_budget), stem_budget),
        ("evolved (default)", evolved_bp, evolved_budget),
    ]

    rows: List[Dict[str, object]] = []
    per_blueprint_tasks: Dict[str, List[dict]] = {}
    for label, bp, budget in pairs:
        results = evaluate_split(split_dir, bp)
        summ = _summarize(results, budget=budget)
        rows.append(_row(label, summ, budget))
        per_blueprint_tasks[label] = summ["tasks"]

    table_str = _ascii_table(rows)
    print(table_str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": args.split,
            "rows": rows,
            "per_blueprint_tasks": per_blueprint_tasks,
            "table": table_str,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="stem")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ev = sub.add_parser("evolve", help="domain analysis + evolution against a benchmark")
    p_ev.add_argument("--bench", default="benchmarks/pybugs")
    p_ev.add_argument("--out", default="artifacts")
    p_ev.add_argument("--max-generations", type=int, default=3)
    p_ev.add_argument("--patience", type=int, default=2)
    p_ev.set_defaults(func=cmd_evolve)

    p_eval = sub.add_parser("eval", help="evaluate a blueprint on a benchmark split")
    p_eval.add_argument("--blueprint", required=True)
    p_eval.add_argument("--bench", default="benchmarks/pybugs")
    p_eval.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_solve = sub.add_parser("solve", help="solve a single task with a blueprint")
    p_solve.add_argument("--blueprint", required=True)
    p_solve.add_argument("--task", required=True)
    p_solve.set_defaults(func=cmd_solve)

    p_cmp = sub.add_parser(
        "compare",
        help="budget-controlled 4-row comparison of stem vs evolved on a split",
    )
    p_cmp.add_argument("--stem", required=True)
    p_cmp.add_argument("--evolved", required=True)
    p_cmp.add_argument("--bench", default="benchmarks/pybugs")
    p_cmp.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p_cmp.add_argument("--out", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
