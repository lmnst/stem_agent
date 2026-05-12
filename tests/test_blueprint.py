import logging

import pytest

from blueprint_repair.blueprint import (
    PRIMITIVE_NAMES,
    WORKFLOW_DEFAULT,
    Blueprint,
    DomainProfile,
    validate_workflow,
)


def test_blueprint_default_is_domain_agnostic():
    b = Blueprint()
    assert b.primitive_priority == PRIMITIVE_NAMES
    assert b.workflow == WORKFLOW_DEFAULT
    assert b.lineage == []


def test_blueprint_roundtrip(tmp_path):
    b = Blueprint(name="x", primitive_budget=10, lineage=["a", "b"])
    p = tmp_path / "bp.json"
    b.to_json(p)
    b2 = Blueprint.from_json(p)
    assert b2 == b


def test_blueprint_from_dict_strict_raises_on_unknown():
    with pytest.raises(ValueError) as excinfo:
        Blueprint.from_dict({"name": "x", "primitive_budget": 5, "unknown_field": 42})
    assert "unknown_field" in str(excinfo.value)


def test_blueprint_from_dict_non_strict_drops_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="blueprint_repair.blueprint"):
        b = Blueprint.from_dict(
            {"name": "x", "primitive_budget": 5, "stale_key": True},
            strict=False,
        )
    assert b.name == "x"
    assert b.primitive_budget == 5
    assert any("stale_key" in r.message for r in caplog.records)


def test_blueprint_from_json_strict_default(tmp_path):
    p = tmp_path / "bp.json"
    p.write_text(
        '{"name": "x", "primitive_budget": 5, "max_iterations": 99}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        Blueprint.from_json(p)
    # non-strict reads it
    b = Blueprint.from_json(p, strict=False)
    assert b.name == "x"


def test_validate_workflow_accepts_default():
    validate_workflow(list(WORKFLOW_DEFAULT))


def test_validate_workflow_rejects_unknown_step():
    with pytest.raises(ValueError):
        validate_workflow(["run_tests", "do_magic", "apply_check"])


def test_validate_workflow_rejects_missing_required():
    with pytest.raises(ValueError):
        validate_workflow(["run_tests", "apply_check"])  # missing propose


def test_validate_workflow_rejects_legacy_localize_step():
    """`localize` was removed; loading old blueprints that include it raises."""
    with pytest.raises(ValueError):
        validate_workflow(["run_tests", "localize", "propose", "apply_check"])


def test_domain_profile_ranked():
    p = DomainProfile(
        primitive_frequencies={
            "swap_arith_pair": 0.6,
            "shift_const_pm1": 0.3,
            "swap_compare_strict": 0.1,
        }
    )
    rk = p.ranked_primitives()
    assert rk[0] == "swap_arith_pair"
    assert rk[1] == "shift_const_pm1"
    assert rk[2] == "swap_compare_strict"
    assert set(rk) == set(PRIMITIVE_NAMES)


def test_domain_profile_empty_ranked_uses_default_order():
    p = DomainProfile(primitive_frequencies={})
    assert p.ranked_primitives() == list(PRIMITIVE_NAMES)


def test_domain_profile_roundtrip(tmp_path):
    p = DomainProfile(
        primitive_frequencies={"shift_const_pm1": 1.0},
        sample_size=5,
        notes=["a", "b"],
    )
    path = tmp_path / "profile.json"
    p.to_json(path)
    p2 = DomainProfile.from_json(path)
    assert p2 == p
