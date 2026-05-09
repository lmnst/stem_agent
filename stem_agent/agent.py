"""The blueprint-driven agent loop.

`solve_task` is used by both the baseline (with the stem blueprint) and
the evolved agent (with a tuned blueprint). The only thing that changes
between them is the blueprint.

The workflow recorded on the blueprint drives which phases run. The
recognized phases are `run_tests`, `propose`, and `apply_check`; see
`blueprint.WORKFLOW_STEPS` and `validate_workflow`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .blueprint import Blueprint, validate_workflow
from .primitives import Variant, generate_variants
from .runner import TestResult, read_solution, run_tests, task_workspace, write_solution


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
    first_stdout: str = ""
    first_stderr: str = ""
    effective_budget: int = 0


def _append_note(existing: str, addition: str) -> str:
    if not addition:
        return existing
    if not existing:
        return addition
    return f"{existing}; {addition}"


def solve_task(
    task_dir: Path,
    blueprint: Blueprint,
    *,
    task_id: Optional[str] = None,
) -> SolveResult:
    """Try to fix the buggy solution.py in `task_dir` per `blueprint`.

    The task's files are never mutated; everything happens in a temp
    workspace that is cleaned up on exit.

    Termination notes (set on `SolveResult.note`):
    - `already passing`  the source already passed before any mutation.
    - `no variants`      the variant queue was empty (e.g. syntax error).
    - `variants exhausted` the queue ran dry before the budget did.
    - `early stop`       `early_stop_no_progress` consecutive misses.
    - `budget exhausted` ran the full budget without a fix.
    """
    tid = task_id or Path(task_dir).name
    attempts: List[AttemptLog] = []
    note = ""
    t_start = time.perf_counter()

    validate_workflow(blueprint.workflow)
    effective_budget = max(1, blueprint.primitive_budget)

    with task_workspace(Path(task_dir)) as ws:
        original = read_solution(ws)

        first: Optional[TestResult] = None
        variants: List[Variant] = []
        first_stdout = ""
        first_stderr = ""

        for step in blueprint.workflow:
            if step == "run_tests":
                first = run_tests(ws)
                first_stdout = first.stdout
                first_stderr = first.stderr
                if first.passed:
                    return SolveResult(
                        task_id=tid,
                        solved=True,
                        iterations=0,
                        duration_s=time.perf_counter() - t_start,
                        fixing_primitive=None,
                        attempts=attempts,
                        note="already passing",
                        first_stdout=first_stdout,
                        first_stderr=first_stderr,
                        effective_budget=effective_budget,
                    )

            elif step == "propose":
                variants = generate_variants(original, blueprint.primitive_priority)

            elif step == "apply_check":
                if not variants:
                    return SolveResult(
                        task_id=tid,
                        solved=False,
                        iterations=0,
                        duration_s=time.perf_counter() - t_start,
                        fixing_primitive=None,
                        attempts=attempts,
                        note=_append_note(note, "no variants"),
                        first_stdout=first_stdout,
                        first_stderr=first_stderr,
                        effective_budget=effective_budget,
                    )

                window = variants[:effective_budget]
                no_progress = 0
                stopped_early = False
                for i, v in enumerate(window):
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
                            first_stdout=first_stdout,
                            first_stderr=first_stderr,
                            effective_budget=effective_budget,
                        )
                    no_progress += 1
                    # Early-stop only fires while there are still untried
                    # variants in the window; otherwise let the loop end and
                    # report the actual cause (budget or variants exhausted).
                    if (
                        no_progress >= blueprint.early_stop_no_progress
                        and i + 1 < len(window)
                    ):
                        stopped_early = True
                        break

                if stopped_early:
                    final_note = _append_note(note, "early stop")
                elif len(window) < effective_budget:
                    final_note = _append_note(note, "variants exhausted")
                else:
                    final_note = _append_note(note, "budget exhausted")

                return SolveResult(
                    task_id=tid,
                    solved=False,
                    iterations=len(attempts),
                    duration_s=time.perf_counter() - t_start,
                    fixing_primitive=None,
                    attempts=attempts,
                    note=final_note,
                    first_stdout=first_stdout,
                    first_stderr=first_stderr,
                    effective_budget=effective_budget,
                )

        # Workflow finished without an apply_check phase. validate_workflow
        # rejects this case, but defend the path for future workflow shapes.
        return SolveResult(
            task_id=tid,
            solved=False,
            iterations=0,
            duration_s=time.perf_counter() - t_start,
            fixing_primitive=None,
            attempts=attempts,
            note=_append_note(note, "no apply_check step"),
            first_stdout=first_stdout,
            first_stderr=first_stderr,
            effective_budget=effective_budget,
        )


def evaluate_split(
    split_dir: Path,
    blueprint: Blueprint,
) -> List[SolveResult]:
    """Run the agent on every task folder under split_dir, in name order."""
    split_dir = Path(split_dir)
    tasks = sorted(p for p in split_dir.iterdir() if p.is_dir())
    return [solve_task(t, blueprint, task_id=t.name) for t in tasks]
