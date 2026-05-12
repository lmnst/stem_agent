"""Documentation contract tests.

Pins:
- the writeup contains exactly one of the three required sentences;
- README documents the perturbation command and links to the
  committed canonical report;
- every fenced bash invocation in the README that calls the CLI is
  parsed by the live argument parser without raising;
- the deployed evolved blueprint, when present, does not carry the
  rejected policy fields in its JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from blueprint_repair.blueprint import Blueprint
from blueprint_repair.cli import make_parser


_REPO = Path(__file__).resolve().parent.parent
_README = _REPO / "README.md"
_WRITEUP = _REPO / "docs" / "writeup.md"
_DEPLOYED = _REPO / "artifacts" / "evolved_blueprint.json"


_REQUIRED_SENTENCES = [
    "This submission demonstrates a learned specialization mechanism that survives controlled ablation.",
    "This submission attempted a learned specialization mechanism; ablation rejected it, and the surviving result is budget plus ordering.",
    "This submission ports the pipeline to a public corpus and reports the result.",
]


def test_writeup_contains_exactly_one_required_sentence():
    text = _WRITEUP.read_text(encoding="utf-8")
    hits = [s for s in _REQUIRED_SENTENCES if s in text]
    assert len(hits) == 1, (
        f"writeup must contain exactly one required sentence verbatim; "
        f"found {len(hits)}: {hits!r}"
    )


def test_writeup_does_not_lead_with_old_75_to_100_claim():
    text = _WRITEUP.read_text(encoding="utf-8")
    forbidden = ("75 -> 100", "75% -> 100%", "75 to 100")
    found = [f for f in forbidden if f in text]
    assert not found, (
        f"the '75% to 100%' framing mixed budget and ordering effects; "
        f"the controlled view splits them. Found {found!r}; lead with the "
        f"perturbation table."
    )


def test_readme_documents_perturb_command():
    text = _README.read_text(encoding="utf-8")
    assert "blueprint_repair.cli perturb" in text
    assert "docs/evaluation/perturbation_report.json" in text


def test_readme_documents_required_pipeline_commands():
    text = _README.read_text(encoding="utf-8")
    for tok in ("evolve", "eval", "compare", "perturb"):
        assert f"blueprint_repair.cli {tok}" in text, f"README must document `{tok}` command"


def _readme_cli_invocations() -> list[list[str]]:
    text = _README.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    invocations: list[list[str]] = []
    for block in blocks:
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "blueprint_repair.cli" not in line:
                continue
            tokens = line.split()
            try:
                idx = tokens.index("blueprint_repair.cli")
            except ValueError:
                continue
            invocations.append(tokens[idx + 1 :])
    return invocations


def test_readme_cli_invocations_parse():
    parser = make_parser()
    invocations = _readme_cli_invocations()
    assert invocations, "README must contain at least one blueprint_repair.cli invocation"
    for argv in invocations:
        try:
            parser.parse_args(argv)
        except SystemExit as e:
            pytest.fail(f"README invocation failed to parse: {' '.join(argv)} ({e})")


def test_deployed_evolved_blueprint_carries_no_policy_keys_in_json():
    """If the artifact exists (evolve has been run), assert it is
    path-B compliant. The artifact is regenerable, not committed; the
    test no-ops when it has not been generated yet."""
    if not _DEPLOYED.exists():
        pytest.skip("artifacts/evolved_blueprint.json not present; run `evolve` to produce it")
    raw = json.loads(_DEPLOYED.read_text(encoding="utf-8"))
    for forbidden in (
        "policy_weights",
        "policy_confidence_threshold",
        "policy_fallback_budget",
    ):
        assert forbidden not in raw, (
            f"deployed blueprint JSON must not contain {forbidden!r}; "
            f"present keys: {sorted(raw)}"
        )


def test_deployed_evolved_blueprint_loads_to_a_no_policy_blueprint():
    if not _DEPLOYED.exists():
        pytest.skip("artifacts/evolved_blueprint.json not present; run `evolve` to produce it")
    bp = Blueprint.from_json(_DEPLOYED)
    assert bp.policy_weights == {}
    assert bp.policy_confidence_threshold == 0.0
    assert bp.policy_fallback_budget == 0
