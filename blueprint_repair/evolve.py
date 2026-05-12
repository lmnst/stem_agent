"""Candidate generation, dev-set selection, and stopping rules.

The candidate space is an explicit, finite grid: priority orderings
crossed with budgets. Each generation scores its candidates on the
dev set, takes the best, then perturbs the top survivors to seed the
next generation.

Selection key (descending priority): (1) max dev pass rate; (2) min
total *actual* attempts spent across the dev split (failed tasks
counted at the iters they actually consumed); (3) closeness to the
profile-recommended budget; (4) smaller raw budget. The actual-attempt
metric replaced an earlier mean-iters-on-solved metric because the
latter let a worse blueprint flatter its mean by averaging only its
easy solves. The deployed strategy is the simplest blueprint that
ties on dev pass rate and minimizes actual attempts.

The deployed evolved blueprint stays purely a priority+budget
configuration. An earlier iteration attached a per-task primitive
policy fit on train + dev observations; controlled ablation on the
held-out test split rejected it (see
`docs/evaluation/perturbation_report.json`). The policy code path
remains in `agent.py` and `policy.py` so the perturbation report can
reconstruct the rejected configuration as a labelled ablation row,
but no policy fields are written into the deployed blueprint.

Stopping rules:
- Stop when no improvement for `patience` consecutive generations.
- Hard cap at `max_generations`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    actual_attempts_sum: int
    n_solved: int
    n_total: int
    target_budget: int
    generation: int = 0
    parent: Optional[str] = None
    mutation_reason: Optional[str] = None
    per_task: List[Dict[str, object]] = field(default_factory=list)
    stop_condition: Optional[str] = None

    def key(self) -> Tuple[float, int, int, int]:
        return (
            -self.pass_rate,
            self.actual_attempts_sum,
            abs(self.blueprint.primitive_budget - self.target_budget),
            self.blueprint.primitive_budget,
        )


# A pool entry pairs a candidate with its parent name and the reason it
# was generated. Parent is None for first-generation candidates.
PoolEntry = Tuple[Blueprint, Optional[str], str]


def stem_blueprint() -> Blueprint:
    """The initial domain-agnostic stem agent: what the baseline runs.

    Two deliberately unspecialized choices:
    - **Alphabetical priority** over primitives. The order is what
      `sorted(PRIMITIVE_NAMES)` produces. The stem has no reason to
      prefer one primitive class over another.
    - **Conservative budget = 8.** Without observing the domain, the
      stem doesn't know how hard tasks are. Eight attempts is enough
      to try the easy wins; harder tasks will run out.

    The policy fields are explicitly empty/zero. With an empty
    `policy_weights` the agent runtime takes the no-policy path and
    behaves exactly like a global-priority searcher.
    """
    return Blueprint(
        name="stem",
        description=(
            "Domain-agnostic stem. Alphabetical primitive priority "
            "(no curation), conservative budget, no learned policy."
        ),
        workflow=list(WORKFLOW_DEFAULT),
        primitive_priority=sorted(PRIMITIVE_NAMES),
        primitive_budget=8,
        early_stop_no_progress=8,
        policy_weights={},
        policy_confidence_threshold=0.0,
        policy_fallback_budget=0,
    )


def _initial_candidates(profile: DomainProfile) -> List[PoolEntry]:
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

    out: List[PoolEntry] = []
    for plabel, pri in priorities:
        for budget in budgets:
            name = f"cand-{plabel}-b{budget}"
            bp = Blueprint(
                name=name,
                description=f"Initial candidate ({plabel} priority, budget={budget}).",
                workflow=list(WORKFLOW_DEFAULT),
                primitive_priority=list(pri),
                primitive_budget=budget,
                early_stop_no_progress=budget,
                lineage=[],
            )
            out.append((bp, None, f"initial: priority={plabel}, budget={budget}"))
    return out


def _clone(bp: Blueprint, **overrides) -> Blueprint:
    data = bp.to_dict()
    data.update(overrides)
    return Blueprint.from_dict(data)


def _next_generation(
    survivors: List[CandidateScore],
    profile: DomainProfile,
) -> List[PoolEntry]:
    """Perturb the top survivors with parent/reason metadata.

    Perturbations are *expansive* (raise budget, promote frequently
    fixing primitives) rather than contractive: the dev set may be
    easy but the test set isn't necessarily. Budget growth is capped
    at four times the profile-recommended budget so a saturated
    dev-set pass-rate cannot drive the persisted blueprint's budget
    arbitrarily high.
    """
    budget_cap = max(8, profile.recommended_budget * 4)
    out: List[PoolEntry] = []
    for cs in survivors:
        bp = cs.blueprint
        next_budget = min(bp.primitive_budget * 2, budget_cap)
        if next_budget > bp.primitive_budget:
            child_name = bp.name + "-x2"
            child = _clone(
                bp,
                primitive_budget=next_budget,
                early_stop_no_progress=next_budget,
                name=child_name,
                lineage=list(bp.lineage) + [bp.name],
            )
            reason = f"double budget ({bp.primitive_budget} -> {next_budget})"
            out.append((child, bp.name, reason))
        head = profile.ranked_primitives()[:5]
        tail = [p for p in bp.primitive_priority if p not in head]
        promoted = list(head) + tail
        if promoted != bp.primitive_priority:
            child_name = bp.name + "-promote"
            child = _clone(
                bp,
                primitive_priority=promoted,
                name=child_name,
                lineage=list(bp.lineage) + [bp.name],
            )
            out.append((child, bp.name, "promote profile top-5 primitives"))

    seen = set()
    unique: List[PoolEntry] = []
    for entry in out:
        b = entry[0]
        key = (
            tuple(b.primitive_priority),
            b.primitive_budget,
            b.early_stop_no_progress,
            tuple(b.workflow),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def score_blueprint(
    bp: Blueprint,
    dev_dir: Path,
    *,
    target_budget: int = 8,
    generation: int = 0,
    parent: Optional[str] = None,
    mutation_reason: Optional[str] = None,
) -> CandidateScore:
    """Score `bp` on the dev split. Per-task records are kept on the
    score object so the evolution log can serialize them."""
    results = evaluate_split(dev_dir, bp)
    n_total = len(results)
    n_solved = sum(1 for r in results if r.solved)
    pass_rate = (n_solved / n_total) if n_total else 0.0
    actual_attempts_sum = sum(r.iterations for r in results)
    per_task = [
        {
            "id": r.task_id,
            "solved": r.solved,
            "iters": r.iterations,
            "fix": r.fixing_primitive,
            "note": r.note,
        }
        for r in results
    ]
    return CandidateScore(
        blueprint=bp,
        pass_rate=pass_rate,
        actual_attempts_sum=actual_attempts_sum,
        n_solved=n_solved,
        n_total=n_total,
        target_budget=target_budget,
        generation=generation,
        parent=parent,
        mutation_reason=mutation_reason,
        per_task=per_task,
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
    score history. The deployed blueprint is the dev winner with no
    policy attached: the per-task policy was attempted in an earlier
    iteration of this project, but controlled ablation on test
    rejected it. See `docs/evaluation/perturbation_report.json` for
    the rejection table; the perturbation report constructs the
    rejected configuration as a labelled ablation row.
    """
    target_budget = max(profile.recommended_budget, 8)
    pool: List[PoolEntry] = _initial_candidates(profile)
    history: List[CandidateScore] = []
    best: Optional[CandidateScore] = None
    no_improve = 0
    stop_condition = "completed_max_generations"

    for gen in range(max_generations):
        scored: List[CandidateScore] = []
        for bp, parent, reason in pool:
            scored.append(
                score_blueprint(
                    bp,
                    dev_dir,
                    target_budget=target_budget,
                    generation=gen,
                    parent=parent,
                    mutation_reason=reason,
                )
            )
        scored.sort(key=lambda s: s.key())
        history.extend(scored)
        gen_best = scored[0]
        if best is None or gen_best.key() < best.key():
            best = gen_best
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            stop_condition = f"no improvement for {patience} generation(s)"
            break
        pool = _next_generation(scored[:survivors_per_gen], profile)
        if not pool:
            stop_condition = "no perturbations available"
            break
    if history:
        history[-1].stop_condition = stop_condition

    assert best is not None
    selected = best.blueprint
    chosen = _clone(
        selected,
        name="evolved",
        description=(
            "Evolved blueprint: priority + budget chosen by dev-set "
            "selection. Per-task policy attempted but rejected by "
            "ablation; not deployed."
        ),
        lineage=list(selected.lineage) + [selected.name],
    )
    return chosen, history
