from stem_agent.primitives import (
    PRIMITIVE_REGISTRY,
    generate_variants,
    primitive_flip_compare,
    primitive_shift_const_pm1,
    primitive_swap_and_or,
    primitive_swap_arith_pair,
    primitive_swap_call_args,
    primitive_swap_compare_strict,
    primitive_swap_eq_neq,
    primitive_swap_true_false,
)
from stem_agent.blueprint import PRIMITIVE_NAMES


def test_swap_compare_strict_basic():
    src = "def f(x):\n    return x < 1\n"
    out = primitive_swap_compare_strict(src)
    assert len(out) == 1
    assert "x <= 1" in out[0].source


def test_swap_compare_strict_two_sites():
    src = "def f(x):\n    return x < 1 and x > 0\n"
    out = primitive_swap_compare_strict(src)
    assert len(out) == 2
    sources = {v.source for v in out}
    assert any("x <= 1" in s for s in sources)
    assert any("x >= 0" in s for s in sources)


def test_flip_compare_basic():
    src = "def f(x):\n    return x < 1\n"
    out = primitive_flip_compare(src)
    assert len(out) == 1
    assert "x > 1" in out[0].source


def test_swap_eq_neq_basic():
    src = "def f(x):\n    return x == 0\n"
    out = primitive_swap_eq_neq(src)
    assert len(out) == 1
    assert "x != 0" in out[0].source


def test_shift_const_pm1_basic():
    src = "def f():\n    return 5\n"
    out = primitive_shift_const_pm1(src)
    sources = {v.source for v in out}
    assert any("return 4" in s for s in sources)
    assert any("return 6" in s for s in sources)


def test_shift_const_pm1_skips_bool():
    src = "def f():\n    return True\n"
    out = primitive_shift_const_pm1(src)
    assert out == []


def test_swap_arith_pair_basic():
    src = "def f(a, b):\n    return a + b\n"
    out = primitive_swap_arith_pair(src)
    sources = {v.source for v in out}
    assert any("a - b" in s for s in sources)
    assert any("a * b" in s for s in sources)
    assert any("a / b" in s for s in sources)
    assert any("a // b" in s for s in sources)
    assert all(v.detail.startswith("Add->") for v in out)


def test_swap_arith_pair_skips_identity():
    src = "def f(a, b):\n    return a + b\n"
    out = primitive_swap_arith_pair(src)
    assert all("Add->Add" not in v.detail for v in out)


def test_swap_and_or_basic():
    src = "def f(a, b):\n    return a and b\n"
    out = primitive_swap_and_or(src)
    assert len(out) == 1
    assert "a or b" in out[0].source


def test_swap_true_false_basic():
    src = "def f():\n    return True\n"
    out = primitive_swap_true_false(src)
    assert len(out) == 1
    assert "return False" in out[0].source


def test_swap_call_args_basic():
    src = "def f(a, b):\n    return divmod(a, b)\n"
    out = primitive_swap_call_args(src)
    assert len(out) == 1
    assert "divmod(b, a)" in out[0].source


def test_swap_call_args_skips_one_arg_calls():
    src = "def f(a):\n    return abs(a)\n"
    out = primitive_swap_call_args(src)
    assert out == []


def test_generate_variants_dedup():
    src = "def f(x):\n    return x < 1\n"
    variants = generate_variants(src, ["swap_compare_strict", "flip_compare"])
    sources = [v.source for v in variants]
    assert len(set(sources)) == len(sources)


def test_generate_variants_skips_zero_increment():
    src = (
        "def f(n):\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        i += 1\n"
        "    return i\n"
    )
    variants = generate_variants(src, list(PRIMITIVE_NAMES))
    for v in variants:
        assert "i += 0" not in v.source
        assert "i -= 0" not in v.source


def test_generate_variants_handles_syntax_error():
    assert generate_variants("def f(:\n", list(PRIMITIVE_NAMES)) == []


def test_registry_has_all_primitives():
    for name in PRIMITIVE_NAMES:
        assert name in PRIMITIVE_REGISTRY


def test_variants_carry_lineno():
    src = "def f(x):\n    if x < 1:\n        return 0\n    return 1\n"
    variants = generate_variants(src, ["swap_compare_strict"])
    assert variants
    assert all(v.target_lineno >= 1 for v in variants)
