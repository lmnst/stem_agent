"""Domain analysis: produce a DomainProfile from a training pool.

Two signals contribute to the profile:

1. **Empirical primitive frequency.** Run a uniform-priority probe
   over the training tasks; for every task that gets solved, count
   which primitive produced the passing variant. The normalized counts
   are the prior the evolved blueprint uses to reorder its priorities.

2. **Iters-to-solve distribution.** The probe records how many
   attempts each train solve took. Recommended budget is derived from
   the maximum observed iter count plus headroom.

Nothing in this module hardcodes the specific tasks or their
primitives. The frequencies are observed from probing.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

from .agent import SolveResult, evaluate_split
from .blueprint import PRIMITIVE_NAMES, Blueprint, DomainProfile


def analyze_domain(
    train_dir: Path,
    *,
    probe_budget: int = 64,
) -> Tuple[DomainProfile, List[SolveResult]]:
    """Produce a DomainProfile by probing training tasks.

    Returns the profile and the per-task probe results. Callers
    consume `results` to fit the per-task primitive policy in
    `evolve` (see `policy.fit_policy`).
    """
    train_dir = Path(train_dir)

    probe = Blueprint(
        name="probe",
        description="Uniform-priority probe used during domain analysis.",
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=probe_budget,
        early_stop_no_progress=probe_budget,
    )
    results = evaluate_split(train_dir, probe)

    counter: Counter[str] = Counter()
    for r in results:
        if r.solved and r.fixing_primitive in PRIMITIVE_NAMES:
            counter[r.fixing_primitive] += 1
    total = sum(counter.values())
    freqs = {p: counter[p] / total for p in counter} if total > 0 else {}

    iters_solved = [r.iterations for r in results if r.solved]
    max_iters = max(iters_solved) if iters_solved else 0
    recommended_budget = max(8, int(round(max_iters * 1.5)) + 4) if max_iters else 8

    notes = [
        f"probed {len(results)} train tasks; {sum(1 for r in results if r.solved)} solved by uniform-priority probe",
        f"max iters observed = {max_iters}; recommended_budget = {recommended_budget}",
        f"primitive fix counts: {dict(counter)}",
    ]
    profile = DomainProfile(
        primitive_frequencies=freqs,
        sample_size=len(results),
        recommended_budget=recommended_budget,
        max_iters_observed=max_iters,
        notes=notes,
    )
    return profile, results
