from solution import first_n_squares


def test_basic():
    assert first_n_squares(4) == [1, 4, 9, 16]


def test_one():
    assert first_n_squares(1) == [1]


def test_zero():
    assert first_n_squares(0) == []
