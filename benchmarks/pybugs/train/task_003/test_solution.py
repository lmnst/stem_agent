from solution import difference


def test_basic():
    assert difference(5, 3) == 2


def test_zero():
    assert difference(0, 0) == 0


def test_negative():
    assert difference(-2, 3) == -5
