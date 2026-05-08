"""Blueprint and DomainProfile data models with JSON serialization.

A Blueprint is the artifact that a stem agent produces (and an evolved
agent consumes). It encodes the choices the brief calls out: configuration,
tool/primitive selection, workflow, and stopping rules. Every field is
plain JSON so it can be diffed and version-controlled.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List


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
WORKFLOW_LOCALIZED: List[str] = ["run_tests", "localize", "propose", "apply_check"]


@dataclass
class Blueprint:
    """A specialized-agent configuration.

    The stem (initial) blueprint is intentionally domain-agnostic: every
    primitive equally weighted, no localization, conservative budget.
    """

    name: str = "stem"
    description: str = "Domain-agnostic stem agent."
    workflow: List[str] = field(default_factory=lambda: list(WORKFLOW_DEFAULT))
    primitive_priority: List[str] = field(
        default_factory=lambda: list(PRIMITIVE_NAMES)
    )
    primitive_budget: int = 32
    use_localization: bool = False
    use_llm_proposal: bool = False
    llm_system_prompt: str = ""
    early_stop_no_progress: int = 32
    max_iterations: int = 32

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Blueprint":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Blueprint":
        keys = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in keys}
        return cls(**clean)


@dataclass
class DomainProfile:
    """Output of the domain-analysis step.

    `primitive_frequencies` is normalized so values sum to 1 (or 0 if no
    fixes were observed). `localization_useful` is set when test
    tracebacks consistently reference solution.py source lines.
    `recommended_budget` is observed-iters-to-solve-max plus headroom —
    the agent's domain-aware sense of how many candidates it should be
    willing to try before giving up.
    """

    primitive_frequencies: Dict[str, float] = field(default_factory=dict)
    localization_useful: bool = False
    llm_hint: str = ""
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
