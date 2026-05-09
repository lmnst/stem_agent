"""Candidate generation, dev-set selection, and stopping rules.

The candidate space is an explicit, finite grid: priority orderings
crossed with budgets. Each generation scores its candidates on the dev
set, takes the best (ties broken by mean iterations-to-solve, then by
budget closeness to the profile-recommended budget), then perturbs the
top survivors to seed the next generation.

Stopping rules:
- Stop when no improvement for `patience` consecutive generations.
- Hard cap at `max_generations`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .agent import evaluate_split
from .blueprint import (
    PRIMITIVE_NAMES,
    WORKFLOW_DEFAULT,
    Blueprint,
    DomainProfile,
)


@dataclass
class CandidateScore:
    blueprint: Blueprint
    pass_rate: float
    mean_iters: float
    n_solved: int
    n_total: int
    target_budget: int
    generation: int = 0

    def key(self) -> Tuple[float, float, int, int]:
        # Sort by descending pass_rate, then ascending mean_iters, then by
        # closeness to the profile-recommended budget (so saturated dev
        # runs do not reward wasteful budget growth), then by smaller
        # raw budget to break the remaining ties deterministically.
        return (
            -self.pass_rate,
            self.mean_iters,
            abs(self.blueprint.primitive_budget - self.target_budget),
            self.blueprint.primitive_budget,
        )


def stem_blueprint() -> Blueprint:
    """The initial domain-agnostic stem agent: what the baseline runs.

    Two deliberately unspecialized choices:
    - **Alphabetical priority** over primitives. The order is what
      `sorted(PRIMITIVE_NAMES)` produces. The stem has no reason to
      prefer one primitive class over another.
    - **Conservative budget = 8.** Without observing the domain, the
      stem doesn't know how hard tasks are. Eight attempts is enough
      to try the easy wins; harder tasks will run out.
    """
    return Blueprint(
        name="stem",
        description=(
            "Domain-agnostic stem. Alphabetical primitive priority "
            "(no curation), conservative budget."
        ),
        workflow=list(WORKFLOW_DEFAULT),
        primitive_priority=sorted(PRIMITIVE_NAMES),
        primitive_budget=8,
        early_stop_no_progress=8,
    )


def _initial_candidates(profile: DomainProfile) -> List[Blueprint]:
    """Candidate blueprints derived from the domain profile.

    Priority orderings: `ranked` from empirical fix-frequency, `alpha`
    as the no-domain-knowledge baseline, `reverse` as an intentionally
    adversarial control to verify the dev-set selection is doing real
    work. Budgets: 8, the profile-recommended budget, and twice that.
    """
    ranked = profile.ranked_primitives()
    alpha = sorted(PRIMITIVE_NAMES)
    reverse = sorted(PRIMITIVE_NAMES, reverse=True)
    priorities = [("ranked", ranked), ("alpha", alpha), ("reverse", reverse)]

    profile_budget = max(profile.recommended_budget, 8)
    budgets = sorted({8, profile_budget, profile_budget * 2})

    out: List[Blueprint] = []
    for plabel, pri in priorities:
        for budget in budgets:
            name = f"cand-{plabel}-b{budget}"
            out.append(
                Blueprint(
                    name=name,
                    description=f"Initial candidate ({plabel} priority, budget={budget}).",
                    workflow=list(WORKFLOW_DEFAULT),
                    primitive_priority=list(pri),
                    primitive_budget=budget,
                    early_stop_no_progress=budget,
                    lineage=[],
                )
            )
    return out


def _clone(bp: Blueprint, **overrides) -> Blueprint:
    data = bp.to_dict()
    data.update(overrides)
    return Blueprint.from_dict(data)


def _next_generation(
    survivors: List[CandidateScore],
    profile: DomainProfile,
) -> List[Blueprint]:
    """Perturb the top survivors to explore their neighborhood.

    Perturbations are *expansive* (raise budget, promote frequently
    fixing primitives) rather than contractive: the dev set may be
    easy but the test set isn't necessarily. Budget growth is capped
    at four times the profile-recommended budget so a saturated
    dev-set pass-rate cannot drive the persisted blueprint's budget
    arbitrarily high.
    """
    budget_cap = max(8, profile.recommended_budget * 4)
    out: List[Blueprint] = []
    for cs in survivors:
        bp = cs.blueprint
        next_budget = min(bp.primitive_budget * 2, budget_cap)
        if next_budget > bp.primitive_budget:
            child_name = bp.name + "-x2"
            out.append(
                _clone(
                    bp,
                    primitive_budget=next_budget,
                    early_stop_no_progress=next_budget,
                    name=child_name,
                    lineage=list(bp.lineage) + [bp.name],
                )
            )
        head = profile.ranked_primitives()[:5]
        tail = [p for p in bp.primitive_priority if p not in head]
        promoted = list(head) + tail
        if promoted != bp.primitive_priority:
            child_name = bp.name + "-promote"
            out.append(
                _clone(
                    bp,
                    primitive_priority=promoted,
                    name=child_name,
                    lineage=list(bp.lineage) + [bp.name],
                )
            )
    seen = set()
    unique: List[Blueprint] = []
    for b in out:
        key = (
            tuple(b.primitive_priority),
            b.primitive_budget,
            b.early_stop_no_progress,
            tuple(b.workflow),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(b)
    return unique


def score_blueprint(
    bp: Blueprint,
    dev_dir: Path,
    *,
    target_budget: int = 8,
    generation: int = 0,
) -> CandidateScore:
    results = evaluate_split(dev_dir, bp)
    n_total = len(results)
    n_solved = sum(1 for r in results if r.solved)
    pass_rate = (n_solved / n_total) if n_total else 0.0
    iters = [r.iterations for r in results if r.solved]
    if n_solved == 0:
        # No solves: declare mean iters as infinity so candidates that
        # fail every task never beat anything on the iters tiebreaker.
        mean_iters = math.inf
    else:
        mean_iters = sum(iters) / len(iters)
    return CandidateScore(
        blueprint=bp,
        pass_rate=pass_rate,
        mean_iters=mean_iters,
        n_solved=n_solved,
        n_total=n_total,
        target_budget=target_budget,
        generation=generation,
    )


def evolve(
    profile: DomainProfile,
    dev_dir: Path,
    *,
    max_generations: int = 4,
    patience: int = 2,
    survivors_per_gen: int = 2,
) -> Tuple[Blueprint, List[CandidateScore]]:
    """Run generational selection over candidate blueprints.

    Returns the best-scoring blueprint and the full per-candidate
    score history (for the artifacts/log). The returned blueprint is
    renamed to `evolved` and gets its full perturbation chain recorded
    in `lineage`.
    """
    target_budget = max(profile.recommended_budget, 8)
    pool = _initial_candidates(profile)
    history: List[CandidateScore] = []
    best: Optional[CandidateScore] = None
    no_improve = 0

    for gen in range(max_generations):
        scored = [
            score_blueprint(c, dev_dir, target_budget=target_budget, generation=gen)
            for c in pool
        ]
        scored.sort(key=lambda s: s.key())
        history.extend(scored)
        gen_best = scored[0]
        if best is None or gen_best.key() < best.key():
            best = gen_best
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
        pool = _next_generation(scored[:survivors_per_gen], profile)
        if not pool:
            break

    assert best is not None
    selected = best.blueprint
    chosen = _clone(
        selected,
        name="evolved",
        description=(
            "Evolved blueprint chosen by dev-set selection. Lineage names "
            "the candidate chain that produced it."
        ),
        lineage=list(selected.lineage) + [selected.name],
    )
    return chosen, history
