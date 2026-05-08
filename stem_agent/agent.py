"""The blueprint-driven agent loop.

`solve_task` is used by both the baseline (with the stem blueprint) and
the evolved agent (with a tuned blueprint). The only thing that changes
between them is the blueprint and the optionally-injected LLM client.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from .blueprint import Blueprint
from .primitives import Variant, generate_variants
from .runner import read_solution, run_tests, task_workspace, write_solution


@dataclass
class AttemptLog:
    primitive: str
    site_index: int
    detail: str
    target_lineno: int
    passed: bool
    duration_s: float


@dataclass
class SolveResult:
    task_id: str
    solved: bool
    iterations: int
    duration_s: float
    fixing_primitive: Optional[str]
    attempts: List[AttemptLog] = field(default_factory=list)
    note: str = ""


_SOL_LINE_RE = re.compile(r"solution\.py[\"',\s:]+(?:line\s+)?(\d+)")


def _localized_lines(stdout: str, stderr: str) -> Set[int]:
    """Pull `solution.py:NN` line refs from a pytest run.

    Used as a coarse fault-localization signal. If pytest prints any
    references, we restrict mutations to those lines (with a safe
    fallback if filtering would empty the candidate list).
    """
    lines: Set[int] = set()
    for blob in (stdout, stderr):
        for m in _SOL_LINE_RE.finditer(blob):
            try:
                lines.add(int(m.group(1)))
            except ValueError:
                continue
    return lines


def solve_task(
    task_dir: Path,
    blueprint: Blueprint,
    *,
    llm=None,
    task_id: Optional[str] = None,
) -> SolveResult:
    """Try to fix the buggy solution.py in `task_dir` per `blueprint`.

    The task's files are never mutated; everything happens in a temp
    workspace that is cleaned up on exit. `llm` is an optional
    LLMClient-like object; if `blueprint.use_llm_proposal` is False or
    the client is unavailable, the agent runs deterministically.
    """
    tid = task_id or Path(task_dir).name
    attempts: List[AttemptLog] = []
    note = ""
    t_start = time.perf_counter()

    with task_workspace(Path(task_dir)) as ws:
        original = read_solution(ws)
        first = run_tests(ws)
        if first.passed:
            return SolveResult(
                task_id=tid,
                solved=True,
                iterations=0,
                duration_s=time.perf_counter() - t_start,
                fixing_primitive=None,
                attempts=attempts,
                note="already passing",
            )

        suspect_lines: Set[int] = set()
        if blueprint.use_localization:
            suspect_lines = _localized_lines(first.stdout, first.stderr)

        variants = generate_variants(original, blueprint.primitive_priority)

        if blueprint.use_llm_proposal and llm is not None and llm.available():
            try:
                proposed = llm.propose_fix(
                    blueprint.llm_system_prompt, original, first.stdout
                )
                if proposed and proposed != original:
                    variants.insert(
                        0,
                        Variant(
                            primitive="llm_proposal",
                            site_index=0,
                            detail="single-shot fix",
                            source=proposed,
                            target_lineno=0,
                        ),
                    )
            except Exception as e:  # LLM failures are best-effort
                note = f"llm proposal error: {type(e).__name__}"

        if blueprint.use_localization and suspect_lines:
            # Soft prior: variants whose target line is suspect try first.
            # Never filter — a misleading traceback shouldn't bury the fix.
            variants.sort(
                key=lambda v: (
                    0 if v.target_lineno == 0 or v.target_lineno in suspect_lines else 1
                )
            )

        budget = max(1, blueprint.primitive_budget)
        no_progress = 0
        for i, v in enumerate(variants[:budget]):
            write_solution(ws, v.source)
            res = run_tests(ws)
            attempts.append(
                AttemptLog(
                    primitive=v.primitive,
                    site_index=v.site_index,
                    detail=v.detail,
                    target_lineno=v.target_lineno,
                    passed=res.passed,
                    duration_s=res.duration_s,
                )
            )
            if res.passed:
                return SolveResult(
                    task_id=tid,
                    solved=True,
                    iterations=i + 1,
                    duration_s=time.perf_counter() - t_start,
                    fixing_primitive=v.primitive,
                    attempts=attempts,
                    note=note,
                )
            no_progress += 1
            if no_progress >= blueprint.early_stop_no_progress:
                note = note or "early stop"
                break

        return SolveResult(
            task_id=tid,
            solved=False,
            iterations=len(attempts),
            duration_s=time.perf_counter() - t_start,
            fixing_primitive=None,
            attempts=attempts,
            note=note or "budget exhausted",
        )


def evaluate_split(
    split_dir: Path,
    blueprint: Blueprint,
    *,
    llm=None,
) -> List[SolveResult]:
    """Run the agent on every task folder under split_dir, in name order."""
    split_dir = Path(split_dir)
    tasks = sorted(p for p in split_dir.iterdir() if p.is_dir())
    return [solve_task(t, blueprint, llm=llm, task_id=t.name) for t in tasks]
