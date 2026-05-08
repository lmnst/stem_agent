from solution import power_of


def test_basic():
    assert power_of(2, 3) == 8


def test_one_exp():
    assert power_of(5, 1) == 5


def test_zero_exp():
    assert power_of(7, 0) == 1
