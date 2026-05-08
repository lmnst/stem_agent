from solution import percent


def test_basic():
    assert percent(2, 5) == 40.0


def test_full():
    assert percent(5, 5) == 100.0


def test_zero():
    assert percent(0, 1) == 0.0
