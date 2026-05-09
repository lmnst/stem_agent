"""Blueprint and DomainProfile data models with JSON serialization.

A Blueprint is the artifact that a stem agent produces (and an evolved
agent consumes). It encodes the choices the brief calls out:
configuration, tool/primitive selection, workflow, and stopping rules.
Every field is plain JSON so it can be diffed and version-controlled.

The `workflow` field is read by `agent.solve_task` and decides which
phases run. The recognized phases are listed in `WORKFLOW_STEPS`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List


log = logging.getLogger(__name__)


PRIMITIVE_NAMES: List[str] = [
    "swap_compare_strict",
    "flip_compare",
    "swap_eq_neq",
    "shift_const_pm1",
    "swap_arith_pair",
    "swap_and_or",
    "swap_true_false",
    "swap_call_args",
]


WORKFLOW_DEFAULT: List[str] = ["run_tests", "propose", "apply_check"]
WORKFLOW_STEPS = frozenset({"run_tests", "propose", "apply_check"})
_WORKFLOW_REQUIRED = ("run_tests", "propose", "apply_check")


def validate_workflow(workflow: List[str]) -> None:
    """Reject workflows the agent cannot execute.

    Unknown steps raise; required steps must all be present. Order is
    not enforced beyond that, but `agent.solve_task` reads steps in
    declared order.
    """
    unknown = [s for s in workflow if s not in WORKFLOW_STEPS]
    if unknown:
        raise ValueError(f"unknown workflow step(s): {unknown}")
    missing = [s for s in _WORKFLOW_REQUIRED if s not in workflow]
    if missing:
        raise ValueError(f"workflow missing required step(s): {missing}")


@dataclass
class Blueprint:
    """A specialized-agent configuration.

    The stem (initial) blueprint is intentionally domain-agnostic:
    every primitive equally weighted, conservative budget. It is the
    no-data baseline.

    `lineage` records the candidate names that produced this blueprint
    via successive evolve perturbations (newest last). It is purely
    informational and is not consumed by the agent.
    """

    name: str = "stem"
    description: str = "Domain-agnostic stem agent."
    workflow: List[str] = field(default_factory=lambda: list(WORKFLOW_DEFAULT))
    primitive_priority: List[str] = field(
        default_factory=lambda: list(PRIMITIVE_NAMES)
    )
    primitive_budget: int = 32
    early_stop_no_progress: int = 32
    lineage: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path, *, strict: bool = True) -> "Blueprint":
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")), strict=strict
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = True) -> "Blueprint":
        """Construct from a dict.

        Strict mode (default) raises on unknown keys so blueprint typos
        do not silently degrade behavior. Pass `strict=False` to log a
        warning and drop the unknowns instead, which is useful when
        loading a blueprint authored against a newer revision.
        """
        keys = {f.name for f in fields(cls)}
        unknown = [k for k in data if k not in keys]
        if unknown:
            if strict:
                raise ValueError(
                    f"unknown blueprint field(s): {sorted(unknown)}. "
                    f"Pass strict=False to drop them with a warning instead."
                )
            log.warning("dropping unknown blueprint field(s): %s", sorted(unknown))
        clean = {k: v for k, v in data.items() if k in keys}
        return cls(**clean)


@dataclass
class DomainProfile:
    """Output of the domain-analysis step.

    `primitive_frequencies` is normalized so values sum to 1 (or 0 if
    no fixes were observed). `recommended_budget` is observed
    iters-to-solve plus headroom: the agent's domain-aware sense of
    how many candidates it should be willing to try before giving up.
    """

    primitive_frequencies: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    recommended_budget: int = 0
    max_iters_observed: int = 0
    notes: List[str] = field(default_factory=list)

    def ranked_primitives(self) -> List[str]:
        ranked = sorted(
            PRIMITIVE_NAMES,
            key=lambda n: (-self.primitive_frequencies.get(n, 0.0), PRIMITIVE_NAMES.index(n)),
        )
        return ranked

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "DomainProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        keys = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in keys}
        return cls(**clean)
