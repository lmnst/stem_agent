"""The blueprint-driven agent loop.

`solve_task` is used by both the baseline (with the stem blueprint) and
the evolved agent (with a tuned blueprint). The only thing that changes
between them is the blueprint and the optionally-injected LLM client.

The workflow recorded on the blueprint drives which phases run. The
recognized phases are `run_tests`, `localize`, `propose`, and
`apply_check`; see `blueprint.WORKFLOW_STEPS` and `validate_workflow`.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

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


_SOL_LINE_RE = re.compile(r"solution\.py[\"',\s:]+(?:line\s+)?(\d+)")


def _localized_lines(stdout: str, stderr: str) -> Set[int]:
    """Return the set of `solution.py:NN` line refs found in pytest output.

    This is the raw signal extracted by the `localize` workflow step.
    The agent applies it as a *soft prior*: variants whose target line
    is in this set are sorted to the front of the queue, but no variant
    is filtered out. A misleading traceback (for example, an assertion
    failure in the test that points at the test file rather than the
    solution) therefore costs ordering, not coverage.
    """
    lines: Set[int] = set()
    for blob in (stdout, stderr):
        for m in _SOL_LINE_RE.finditer(blob):
            try:
                lines.add(int(m.group(1)))
            except ValueError:
                continue
    return lines


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
    llm=None,
    task_id: Optional[str] = None,
) -> SolveResult:
    """Try to fix the buggy solution.py in `task_dir` per `blueprint`.

    The task's files are never mutated; everything happens in a temp
    workspace that is cleaned up on exit. `llm` is an optional
    LLMClient-like object; if `blueprint.use_llm_proposal` is False or
    the client is unavailable, the agent runs deterministically.

    Termination notes (set on `SolveResult.note`):
    - `already passing`  the source already passed before any mutation.
    - `no variants`      the variant queue was empty (e.g. syntax error).
    - `variants exhausted` the queue ran dry before the budget did.
    - `early stop`       `early_stop_no_progress` consecutive misses.
    - `budget exhausted` ran the full budget without a fix.
    LLM call errors append a `; llm: <ErrorClass>` suffix.
    """
    tid = task_id or Path(task_dir).name
    attempts: List[AttemptLog] = []
    note = ""
    t_start = time.perf_counter()

    validate_workflow(blueprint.workflow)

    with task_workspace(Path(task_dir)) as ws:
        original = read_solution(ws)

        first: Optional[TestResult] = None
        suspect_lines: Set[int] = set()
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
                    )

            elif step == "localize":
                if first is not None:
                    suspect_lines = _localized_lines(first.stdout, first.stderr)

            elif step == "propose":
                variants = generate_variants(original, blueprint.primitive_priority)
                if blueprint.use_llm_proposal and llm is not None and llm.available():
                    proposed = llm.propose_fix(
                        blueprint.llm_system_prompt, original, first_stdout
                    )
                    err_name = getattr(llm, "last_error_type", lambda: None)()
                    if err_name:
                        note = _append_note(note, f"llm: {err_name}")
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
                if suspect_lines:
                    variants.sort(
                        key=lambda v: (
                            0
                            if v.target_lineno == 0 or v.target_lineno in suspect_lines
                            else 1
                        )
                    )

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
                    )

                budget = max(1, blueprint.primitive_budget)
                window = variants[:budget]
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
                elif len(window) < budget:
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
