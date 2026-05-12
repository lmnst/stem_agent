"""First-class perturbation report.

Reproducible ablation table for the deployed evolved blueprint. The
report builds seven rows on every requested split:

1. `stem default`         - stem priority + stem budget, no policy
2. `stem evolved budget`  - stem priority + evolved budget, no policy
3. `deployed evolved`     - the deployed evolved blueprint as written
4. `zero policy`          - deployed priority + budget, policy fields zeroed
5. `random policy`        - deployed priority + budget + Gaussian policy
                            weights drawn from a fixed seed
6. `reverse only`         - reverse alphabetical priority + evolved
                            budget, no policy
7. `policy only`          - alphabetical priority + evolved budget +
                            learned policy fit on train + dev

Rows 4 and 6 are explicitly identical to the deployed evolved row
when the deployed blueprint carries no policy. The redundancy is the
point: it shows the reader that the deployed strategy is just budget
plus ordering.

The learned-policy ablation (rows 5 and 7, plus a phantom variant of
row 3) is reconstructed from train + dev observations on every report
run; it is *not* read from the deployed blueprint, because the
deployed blueprint does not carry policy fields. This makes the
report stand on its own.

Each row carries:
- pass rate and Wilson 95% CI;
- actual attempts spent (sum and mean);
- effective-budget attempts for failed tasks (sum and mean);
- mean attempts on solved tasks;
- per-task records, including effective budget and policy state;
- whether the policy fallback budget fired on any task;
- the exact blueprint fields the row's strategy actually consumes.

All numbers are derived from the deterministic primitive-search path.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .agent import SolveResult, evaluate_split
from .analysis import analyze_domain
from .blueprint import PRIMITIVE_NAMES, WORKFLOW_DEFAULT, Blueprint
from .policy import (
    FEATURE_NAMES,
    Observation,
    extract_features,
    fit_fallback_budget_from_iters,
    fit_policy,
    fit_threshold,
)
from .stats import wilson_interval


_DEPLOYED_FIELDS = (
    "workflow",
    "primitive_priority",
    "primitive_budget",
    "early_stop_no_progress",
)
_POLICY_FIELDS = (
    "policy_weights",
    "policy_confidence_threshold",
    "policy_fallback_budget",
)


@dataclass
class PerturbConfig:
    """Inputs for one perturbation report run.

    `random_seed` pins the Gaussian draw used for the random-policy
    row. Two reports built from the same blueprints, bench, and seed
    are byte-identical.
    """

    bench: Path
    stem_blueprint: Blueprint
    deployed_blueprint: Blueprint
    splits: Tuple[str, ...] = ("test",)
    random_seed: int = 1234


def gather_observations(
    train_dir: Path,
    train_results: List[SolveResult],
    dev_dir: Path,
    dev_results: List[SolveResult],
) -> Tuple[List[Observation], List[int]]:
    """Build (features, fixing_primitive) observations across train + dev.

    Features come from the *original* (buggy) source of each task.
    The accompanying iters list (one per solved task) feeds
    `fit_fallback_budget_from_iters` for the policy fallback budget.
    """
    obs: List[Observation] = []
    iters: List[int] = []
    for split_dir, results in [(train_dir, train_results), (dev_dir, dev_results)]:
        for r in results:
            task_path = Path(split_dir) / r.task_id / "solution.py"
            if not task_path.exists():
                continue
            feats = extract_features(task_path.read_text(encoding="utf-8"))
            obs.append((feats, r.fixing_primitive if r.solved else None))
            if r.solved:
                iters.append(r.iterations)
    return obs, iters


def fit_experimental_policy(
    bench: Path, base_for_dev_eval: Blueprint
) -> Tuple[Dict[str, Dict[str, float]], float, int]:
    """Refit the rejected per-task policy from train + dev for ablation rows.

    The deployed blueprint does not carry policy fields. The
    perturbation report needs a learned policy to score the
    `random policy` and `policy only` ablations against, so we
    rebuild the same artefact the rejected experiment produced:
    weights from `fit_policy`, threshold from `fit_threshold`,
    fallback budget from observed iters-to-solve.
    """
    profile, train_results = analyze_domain(bench / "train")
    del profile
    dev_results = evaluate_split(bench / "dev", base_for_dev_eval)
    obs, iters = gather_observations(
        bench / "train", train_results, bench / "dev", dev_results
    )
    weights = fit_policy(obs)
    threshold = fit_threshold(obs, weights)
    fallback = fit_fallback_budget_from_iters(iters)
    return weights, threshold, fallback


def random_policy_weights(seed: int) -> Dict[str, Dict[str, float]]:
    """Gaussian(0, 1) weights, primitive x feature, drawn from `seed`.

    Deterministic across platforms because `random.Random` with a
    fixed seed is part of the Python contract.
    """
    rng = random.Random(seed)
    return {
        p: {f: rng.gauss(0.0, 1.0) for f in FEATURE_NAMES}
        for p in PRIMITIVE_NAMES
    }


def _make_blueprint(
    *,
    name: str,
    description: str,
    priority: List[str],
    budget: int,
    policy_weights: Optional[Dict[str, Dict[str, float]]] = None,
    policy_threshold: float = 0.0,
    policy_fallback: int = 0,
) -> Blueprint:
    return Blueprint(
        name=name,
        description=description,
        workflow=list(WORKFLOW_DEFAULT),
        primitive_priority=list(priority),
        primitive_budget=budget,
        early_stop_no_progress=budget,
        policy_weights=policy_weights or {},
        policy_confidence_threshold=policy_threshold,
        policy_fallback_budget=policy_fallback,
    )


def _summarize_results(
    bp: Blueprint, results: List[SolveResult]
) -> Dict[str, object]:
    n = len(results)
    n_solved = sum(1 for r in results if r.solved)
    actual_sum = sum(r.iterations for r in results)
    eff_sum = sum(
        r.iterations if r.solved else (r.effective_budget or bp.primitive_budget)
        for r in results
    )
    iters_solved = [r.iterations for r in results if r.solved]
    lo, hi = wilson_interval(n_solved, n)
    fallback_fired = any(r.policy_low_confidence for r in results)
    fields_used = list(_DEPLOYED_FIELDS)
    if bp.policy_weights:
        fields_used.extend(_POLICY_FIELDS)
    per_task = [
        {
            "id": r.task_id,
            "solved": r.solved,
            "actual_attempts": r.iterations,
            "effective_budget": r.effective_budget,
            "fix": r.fixing_primitive,
            "note": r.note,
            "policy_low_confidence": r.policy_low_confidence,
            "policy_top_primitive": r.policy_top_primitive,
            "policy_top_score": r.policy_top_score,
            "primitive_order": list(bp.primitive_priority),
        }
        for r in results
    ]
    return {
        "n_tasks": n,
        "n_solved": n_solved,
        "pass_rate": (n_solved / n) if n else 0.0,
        "pass_rate_ci95": [lo, hi],
        "actual_attempts_sum": actual_sum,
        "eff_budget_attempts_sum": eff_sum,
        "mean_iters_solved": (sum(iters_solved) / len(iters_solved)) if iters_solved else 0.0,
        "mean_iters_actual_all": (actual_sum / n) if n else 0.0,
        "mean_iters_eff_all": (eff_sum / n) if n else 0.0,
        "fallback_budget_fired": fallback_fired,
        "blueprint_fields_used": fields_used,
        "tasks": per_task,
    }


def _row_blueprints(
    config: PerturbConfig,
    learned_weights: Dict[str, Dict[str, float]],
    learned_threshold: float,
    learned_fallback: int,
) -> List[Tuple[str, Blueprint]]:
    """Build the 7 ablation rows in the documented order.

    The deployed blueprint defines the priority and budget that the
    `zero policy`, `random policy`, and `policy only` rows pivot
    around. The deployed blueprint may itself carry policy fields
    (path-A submission); under path B it does not, in which case
    rows 3, 4, and 6 are byte-identical strategies.
    """
    stem = config.stem_blueprint
    deployed = config.deployed_blueprint
    alpha = sorted(PRIMITIVE_NAMES)
    reverse = sorted(PRIMITIVE_NAMES, reverse=True)

    return [
        (
            "stem default",
            _make_blueprint(
                name="row_stem_default",
                description="Stem blueprint as shipped.",
                priority=stem.primitive_priority,
                budget=stem.primitive_budget,
            ),
        ),
        (
            "stem evolved budget",
            _make_blueprint(
                name="row_stem_evolved_budget",
                description="Stem priority with the evolved budget.",
                priority=stem.primitive_priority,
                budget=deployed.primitive_budget,
            ),
        ),
        (
            "deployed evolved",
            deployed,
        ),
        (
            "zero policy",
            _make_blueprint(
                name="row_zero_policy",
                description="Deployed priority and budget, policy fields zeroed.",
                priority=deployed.primitive_priority,
                budget=deployed.primitive_budget,
            ),
        ),
        (
            "random policy",
            _make_blueprint(
                name="row_random_policy",
                description=(
                    f"Deployed priority and budget plus Gaussian(0,1) weights "
                    f"drawn from seed={config.random_seed}."
                ),
                priority=deployed.primitive_priority,
                budget=deployed.primitive_budget,
                policy_weights=random_policy_weights(config.random_seed),
                policy_threshold=learned_threshold,
                policy_fallback=learned_fallback,
            ),
        ),
        (
            "reverse only",
            _make_blueprint(
                name="row_reverse_only",
                description="Reverse-alphabetical priority with the evolved budget.",
                priority=reverse,
                budget=deployed.primitive_budget,
            ),
        ),
        (
            "policy only",
            _make_blueprint(
                name="row_policy_only",
                description=(
                    "Alphabetical priority with the rejected learned policy "
                    "attached. Isolates the policy from the priority change."
                ),
                priority=alpha,
                budget=deployed.primitive_budget,
                policy_weights=learned_weights,
                policy_threshold=learned_threshold,
                policy_fallback=learned_fallback,
            ),
        ),
    ]


def build_report(config: PerturbConfig) -> Dict[str, object]:
    """Run the perturbation experiments and return a JSON-ready report."""
    base_for_fit = _make_blueprint(
        name="fit_eval",
        description="Used only to gather dev observations for policy fit.",
        priority=config.deployed_blueprint.primitive_priority,
        budget=config.deployed_blueprint.primitive_budget,
    )
    weights, threshold, fallback = fit_experimental_policy(
        config.bench, base_for_fit
    )
    rows_blueprints = _row_blueprints(config, weights, threshold, fallback)

    splits_payload: Dict[str, Dict[str, object]] = {}
    for split in config.splits:
        split_dir = config.bench / split
        if not split_dir.is_dir():
            continue
        rows: List[Dict[str, object]] = []
        for label, bp in rows_blueprints:
            results = evaluate_split(split_dir, bp)
            summary = _summarize_results(bp, results)
            rows.append(
                {
                    "row": label,
                    "budget": bp.primitive_budget,
                    "priority": list(bp.primitive_priority),
                    "policy_active": bool(bp.policy_weights),
                    **summary,
                }
            )
        splits_payload[split] = {"rows": rows}

    return {
        "schema_version": 1,
        "random_seed": config.random_seed,
        "row_order": [label for label, _ in rows_blueprints],
        "experimental_policy": {
            "note": (
                "Refit from train + dev observations on every report run. "
                "Not deployed: ablation rejected this mechanism on the "
                "held-out test split."
            ),
            "policy_confidence_threshold": threshold,
            "policy_fallback_budget": fallback,
            "n_primitives": len(weights),
            "n_features": len(next(iter(weights.values()))) if weights else 0,
            "n_positive_weights": sum(
                1 for fw in weights.values() for w in fw.values() if w > 0
            ),
        },
        "splits": splits_payload,
    }


def render_table(report: Dict[str, object]) -> str:
    """Render a fixed-width ASCII table for stdout."""
    lines: List[str] = []
    splits = report.get("splits", {})
    if not isinstance(splits, dict):
        return ""
    for split, payload in splits.items():
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        header = (
            f"{'row':<22} {'budget':>6} {'pass':<14} {'CI95':<22} "
            f"{'actual':>7} {'eff_bud':>7} {'fallback':>9}"
        )
        lines.append(f"=== {split} ===")
        lines.append(header)
        lines.append("-" * len(header))
        for r in rows:
            lo, hi = r["pass_rate_ci95"]
            pass_str = f"{r['n_solved']}/{r['n_tasks']} ({r['pass_rate'] * 100:.1f}%)"
            ci_str = f"[{lo * 100:.1f}, {hi * 100:.1f}]"
            fb = "yes" if r["fallback_budget_fired"] else "no"
            lines.append(
                f"{r['row']:<22} {r['budget']:>6d} {pass_str:<14} {ci_str:<22} "
                f"{r['actual_attempts_sum']:>7d} {r['eff_budget_attempts_sum']:>7d} {fb:>9}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def write_report(report: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
