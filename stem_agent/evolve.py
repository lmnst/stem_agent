"""Candidate generation, dev-set selection, and stopping rules.

The candidate space is an explicit, finite grid: priority orderings ×
budgets × early-stop values × localization flag × LLM-proposal flag.
This is the "evolution" loop: each generation scores its candidates on
the dev set, takes the best (ties broken by mean iterations-to-solve),
and then perturbs the top survivors to seed the next generation.

Stopping rules:
- Stop when no improvement for `patience` consecutive generations.
- Hard cap at `max_generations`.

Both rules are needed: patience handles the common case (gain plateaus
after one or two rounds), the hard cap is a safety net.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from .agent import evaluate_split
from .blueprint import PRIMITIVE_NAMES, Blueprint, DomainProfile
from .llm import LLMClient


@dataclass
class CandidateScore:
    blueprint: Blueprint
    pass_rate: float
    mean_iters: float
    n_solved: int
    n_total: int
    generation: int = 0

    def key(self) -> Tuple[float, float, int]:
        # Sort by descending pass_rate, then ascending mean_iters,
        # then descending budget (larger = more headroom on hard tasks).
        return (-self.pass_rate, self.mean_iters, -self.blueprint.primitive_budget)


def stem_blueprint() -> Blueprint:
    """The initial domain-agnostic stem agent — what the baseline runs.

    Two deliberately unspecialized choices:
    - **Alphabetical priority** over primitives. The order isn't curated;
      it's what you get from `sorted(PRIMITIVE_NAMES)`. The stem has no
      reason to prefer one primitive class over another.
    - **Conservative budget = 8.** Without observing the domain, the
      stem doesn't know how hard tasks are. Eight attempts is enough to
      try the easy wins; harder tasks will run out.
    """
    return Blueprint(
        name="stem",
        description=(
            "Domain-agnostic stem. Alphabetical primitive priority "
            "(no curation), conservative budget."
        ),
        primitive_priority=sorted(PRIMITIVE_NAMES),
        primitive_budget=8,
        use_localization=False,
        use_llm_proposal=False,
        early_stop_no_progress=8,
        max_iterations=8,
    )


def _initial_candidates(
    profile: DomainProfile,
    *,
    use_llm: bool,
) -> List[Blueprint]:
    """Candidate blueprints derived from the domain profile.

    Three priority orderings × two budget tiers × localization options.
    The "ranked" priority comes from the profile's empirical
    primitive-fix frequencies; "alpha" is the no-domain-knowledge
    baseline; "reverse" is an intentionally adversarial ordering used to
    test that the dev-set selection is doing real work.
    """
    ranked = profile.ranked_primitives()
    alpha = sorted(PRIMITIVE_NAMES)
    reverse = sorted(PRIMITIVE_NAMES, reverse=True)
    priorities = [("ranked", ranked), ("alpha", alpha), ("reverse", reverse)]

    profile_budget = max(profile.recommended_budget, 8)
    budgets = sorted({8, profile_budget, profile_budget * 2})
    loc_options = (False, True) if profile.localization_useful else (False,)

    out: List[Blueprint] = []
    for plabel, pri in priorities:
        for budget in budgets:
            for use_loc in loc_options:
                name = f"cand-{plabel}-b{budget}-loc{int(use_loc)}-llm{int(use_llm)}"
                out.append(
                    Blueprint(
                        name=name,
                        description=f"Initial candidate ({plabel} priority, budget={budget}).",
                        primitive_priority=list(pri),
                        primitive_budget=budget,
                        use_localization=use_loc,
                        use_llm_proposal=use_llm,
                        llm_system_prompt=profile.llm_hint or "",
                        early_stop_no_progress=budget,
                        max_iterations=budget,
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

    Perturbations are *expansive* (raise budget, enable localization,
    promote frequently-fixing primitives) rather than contractive — the
    dev set may be easy, but the test set isn't necessarily.
    """
    out: List[Blueprint] = []
    for cs in survivors:
        bp = cs.blueprint
        # Expand budget
        out.append(
            _clone(
                bp,
                primitive_budget=bp.primitive_budget * 2,
                early_stop_no_progress=bp.primitive_budget * 2,
                max_iterations=bp.primitive_budget * 2,
                name=bp.name + "-x2",
            )
        )
        # Promote profile-top-5 to the head, keep tail order
        head = profile.ranked_primitives()[:5]
        tail = [p for p in bp.primitive_priority if p not in head]
        promoted = list(head) + tail
        if promoted != bp.primitive_priority:
            out.append(_clone(bp, primitive_priority=promoted, name=bp.name + "-promote"))
        # Localization if available and not yet on
        if profile.localization_useful and not bp.use_localization:
            out.append(_clone(bp, use_localization=True, name=bp.name + "-loc"))
    seen = set()
    unique: List[Blueprint] = []
    for b in out:
        key = (
            tuple(b.primitive_priority),
            b.primitive_budget,
            b.use_localization,
            b.early_stop_no_progress,
            b.use_llm_proposal,
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
    llm: Optional[LLMClient] = None,
    generation: int = 0,
) -> CandidateScore:
    results = evaluate_split(dev_dir, bp, llm=llm)
    n_total = len(results)
    n_solved = sum(1 for r in results if r.solved)
    pass_rate = (n_solved / n_total) if n_total else 0.0
    iters = [r.iterations for r in results if r.solved]
    mean_iters = (sum(iters) / len(iters)) if iters else float(bp.primitive_budget)
    return CandidateScore(
        blueprint=bp,
        pass_rate=pass_rate,
        mean_iters=mean_iters,
        n_solved=n_solved,
        n_total=n_total,
        generation=generation,
    )


def evolve(
    profile: DomainProfile,
    dev_dir: Path,
    *,
    llm: Optional[LLMClient] = None,
    max_generations: int = 4,
    patience: int = 2,
    survivors_per_gen: int = 2,
) -> Tuple[Blueprint, List[CandidateScore]]:
    """Run generational selection over candidate blueprints.

    Returns the best-scoring blueprint and the full per-candidate score
    history (for the artifacts/log).
    """
    use_llm = llm is not None and llm.available()
    pool = _initial_candidates(profile, use_llm=use_llm)
    history: List[CandidateScore] = []
    best: Optional[CandidateScore] = None
    no_improve = 0

    for gen in range(max_generations):
        scored = [
            score_blueprint(c, dev_dir, llm=llm, generation=gen) for c in pool
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
    return best.blueprint, history
