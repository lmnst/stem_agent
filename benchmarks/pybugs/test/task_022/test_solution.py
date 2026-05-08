from solution import first_n_pow_two


def test_basic():
    assert first_n_pow_two(4) == [1, 2, 4, 8]


def test_one():
    assert first_n_pow_two(1) == [1]


def test_zero():
    assert first_n_pow_two(0) == []
