"""Domain analysis: produce a DomainProfile from a training pool.

Three signals contribute to the profile:

1. **Empirical primitive frequency (heuristic).** Run a uniform-priority
   probe over the training tasks; for every task that gets solved, count
   which primitive produced the passing variant. The normalized counts
   are the prior the evolved blueprint uses to reorder its priorities.

2. **Localization usefulness (heuristic).** For each task, inspect the
   stdout/stderr the probe captured on its first (pre-mutation) run.
   If a majority of train tasks emit `solution.py:LINENO` references,
   `localization_useful` is set True. The probe already produced this
   output, so the analyzer reads it from `SolveResult.first_stdout/
   first_stderr` rather than re-running the tests.

3. **Bug-family hint (LLM-optional).** When an LLM is available, ask
   for a 3-bullet summary of recurring bug families. Used as the
   `llm_system_prompt` in candidate blueprints. Empty string when no
   LLM, and the rest of the pipeline is unaffected.

Nothing in this module hardcodes the specific tasks or their primitives.
The frequencies are observed from probing.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from .agent import SolveResult, evaluate_split
from .blueprint import PRIMITIVE_NAMES, Blueprint, DomainProfile
from .llm import LLMClient


_LINE_RE = re.compile(r"solution\.py[\"',\s:]+(?:line\s+)?(\d+)")


def _localization_signal(results: List[SolveResult]) -> Tuple[int, int]:
    """Return (count_with_line_refs, total) from the probe's captured first runs.

    Reads the stdout/stderr the probe already captured on each task's
    pre-mutation run, so we do not pay for a second pass over the train
    split.
    """
    found, total = 0, 0
    for r in results:
        blob = (r.first_stdout or "") + "\n" + (r.first_stderr or "")
        if _LINE_RE.search(blob):
            found += 1
        total += 1
    return found, total


def analyze_domain(
    train_dir: Path,
    *,
    llm: Optional[LLMClient] = None,
    probe_budget: int = 64,
) -> Tuple[DomainProfile, List[SolveResult]]:
    """Produce a DomainProfile by probing training tasks.

    Returns the profile and the per-task probe results (used by callers
    for logs / write-up artifacts).
    """
    train_dir = Path(train_dir)
    task_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())

    probe = Blueprint(
        name="probe",
        description="Uniform-priority probe used during domain analysis.",
        primitive_priority=list(PRIMITIVE_NAMES),
        primitive_budget=probe_budget,
        use_llm_proposal=False,
        early_stop_no_progress=probe_budget,
    )
    results = evaluate_split(train_dir, probe, llm=None)

    counter: Counter[str] = Counter()
    for r in results:
        if r.solved and r.fixing_primitive in PRIMITIVE_NAMES:
            counter[r.fixing_primitive] += 1
    total = sum(counter.values())
    freqs = {p: counter[p] / total for p in counter} if total > 0 else {}

    iters_solved = [r.iterations for r in results if r.solved]
    max_iters = max(iters_solved) if iters_solved else 0
    recommended_budget = max(8, int(round(max_iters * 1.5)) + 4) if max_iters else 8

    loc_found, loc_total = _localization_signal(results)
    localization_useful = loc_total > 0 and loc_found >= max(1, (loc_total + 1) // 2)

    hint = ""
    if llm is not None and llm.available():
        samples: List[Tuple[str, str]] = []
        for td in task_dirs[:8]:
            src_path = td / "solution.py"
            if src_path.exists():
                samples.append((td.name, src_path.read_text(encoding="utf-8")))
        hint = llm.summarize_bug_families(samples)

    notes = [
        f"probed {len(results)} train tasks; {sum(1 for r in results if r.solved)} solved by uniform-priority probe",
        f"max iters observed = {max_iters}; recommended_budget = {recommended_budget}",
        f"localization signal: {loc_found}/{loc_total} tasks emitted line refs",
        f"primitive fix counts: {dict(counter)}",
    ]
    if hint:
        notes.append("llm hint captured")
    profile = DomainProfile(
        primitive_frequencies=freqs,
        localization_useful=localization_useful,
        llm_hint=hint,
        sample_size=len(results),
        recommended_budget=recommended_budget,
        max_iters_observed=max_iters,
        notes=notes,
    )
    return profile, results
