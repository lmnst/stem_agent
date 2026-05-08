"""Command-line entry point: `stem evolve | eval | solve`."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

from .agent import SolveResult, evaluate_split, solve_task
from .analysis import analyze_domain
from .blueprint import Blueprint
from .evolve import evolve, stem_blueprint
from .llm import LLMClient


def _summarize(results: List[SolveResult]) -> dict:
    n = len(results)
    n_solved = sum(1 for r in results if r.solved)
    iters = [r.iterations for r in results if r.solved]
    durations = [r.duration_s for r in results]
    return {
        "n_tasks": n,
        "n_solved": n_solved,
        "pass_rate": (n_solved / n) if n else 0.0,
        "mean_iters_solved": (sum(iters) / len(iters)) if iters else 0.0,
        "mean_wall_s": (sum(durations) / n) if n else 0.0,
        "tasks": [
            {
                "id": r.task_id,
                "solved": r.solved,
                "iters": r.iterations,
                "duration_s": round(r.duration_s, 3),
                "fix": r.fixing_primitive,
                "note": r.note,
            }
            for r in results
        ],
    }


def _print_llm_summary(llm: Optional[LLMClient], use_llm: bool) -> None:
    if not use_llm or llm is None:
        return
    print(f"[llm] {llm.summary()}", file=sys.stderr)


def cmd_evolve(args: argparse.Namespace) -> int:
    bench = Path(args.bench)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient() if args.use_llm else None
    if llm is not None:
        print(f"[llm] {llm.status()}", file=sys.stderr)

    print("[1/3] domain analysis (probe over train)", file=sys.stderr)
    profile, _ = analyze_domain(bench / "train", llm=llm)
    profile.to_json(out_dir / "profile.json")
    print(f"  primitive_frequencies: {profile.primitive_frequencies}", file=sys.stderr)
    print(f"  localization_useful:   {profile.localization_useful}", file=sys.stderr)

    print("[2/3] evolving on dev split", file=sys.stderr)
    best_bp, history = evolve(
        profile,
        bench / "dev",
        llm=llm,
        max_generations=args.max_generations,
        patience=args.patience,
    )
    best_bp.to_json(out_dir / "evolved_blueprint.json")

    history_dump = [
        {
            "generation": s.generation,
            "name": s.blueprint.name,
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
    _print_llm_summary(llm, args.use_llm)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    bp = Blueprint.from_json(Path(args.blueprint))
    bench = Path(args.bench)
    split_dir = bench / args.split
    llm = LLMClient() if args.use_llm else None
    results = evaluate_split(split_dir, bp, llm=llm)
    summary = _summarize(results)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "tasks"}, indent=2))
    _print_llm_summary(llm, args.use_llm)
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    bp = Blueprint.from_json(Path(args.blueprint))
    llm = LLMClient() if args.use_llm else None
    res = solve_task(Path(args.task), bp, llm=llm)
    print(
        json.dumps(
            {
                "task": res.task_id,
                "solved": res.solved,
                "iters": res.iterations,
                "duration_s": round(res.duration_s, 3),
                "fix": res.fixing_primitive,
                "note": res.note,
            },
            indent=2,
        )
    )
    _print_llm_summary(llm, args.use_llm)
    return 0 if res.solved else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="stem")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ev = sub.add_parser("evolve", help="domain analysis + evolution against a benchmark")
    p_ev.add_argument("--bench", default="benchmarks/pybugs")
    p_ev.add_argument("--out", default="artifacts")
    p_ev.add_argument("--use-llm", action="store_true",
                      help="Permit LLM calls if OPENAI_API_KEY is set; required for LLM-driven candidates.")
    p_ev.add_argument("--max-generations", type=int, default=3)
    p_ev.add_argument("--patience", type=int, default=2)
    p_ev.set_defaults(func=cmd_evolve)

    p_eval = sub.add_parser("eval", help="evaluate a blueprint on a benchmark split")
    p_eval.add_argument("--blueprint", required=True)
    p_eval.add_argument("--bench", default="benchmarks/pybugs")
    p_eval.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p_eval.add_argument("--out", default=None)
    p_eval.add_argument("--use-llm", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_solve = sub.add_parser("solve", help="solve a single task with a blueprint")
    p_solve.add_argument("--blueprint", required=True)
    p_solve.add_argument("--task", required=True)
    p_solve.add_argument("--use-llm", action="store_true")
    p_solve.set_defaults(func=cmd_solve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
