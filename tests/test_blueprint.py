from stem_agent.blueprint import PRIMITIVE_NAMES, Blueprint, DomainProfile


def test_blueprint_default_is_domain_agnostic():
    b = Blueprint()
    assert b.primitive_priority == PRIMITIVE_NAMES
    assert b.use_localization is False
    assert b.use_llm_proposal is False
    assert b.llm_system_prompt == ""


def test_blueprint_roundtrip(tmp_path):
    b = Blueprint(name="x", primitive_budget=10, use_localization=True)
    p = tmp_path / "bp.json"
    b.to_json(p)
    b2 = Blueprint.from_json(p)
    assert b2 == b


def test_blueprint_from_dict_ignores_extras():
    b = Blueprint.from_dict({"name": "x", "primitive_budget": 5, "unknown_field": 42})
    assert b.name == "x"
    assert b.primitive_budget == 5


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
        localization_useful=True,
        llm_hint="some hint",
        sample_size=5,
        notes=["a", "b"],
    )
    path = tmp_path / "profile.json"
    p.to_json(path)
    p2 = DomainProfile.from_json(path)
    assert p2 == p
