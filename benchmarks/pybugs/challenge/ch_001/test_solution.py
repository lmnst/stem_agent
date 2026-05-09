from solution import add_pair


def test_basic():
    assert add_pair(2, 3, 1) == 6


def test_zero_carry():
    assert add_pair(2, 3, 0) == 5


def test_negatives():
    assert add_pair(-1, 4, 0) == 3
