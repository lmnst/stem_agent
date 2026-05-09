from solution import sum_first_n


def test_basic():
    assert sum_first_n(3) == 6


def test_five():
    assert sum_first_n(5) == 15


def test_one():
    assert sum_first_n(1) == 1


def test_zero():
    assert sum_first_n(0) == 0
