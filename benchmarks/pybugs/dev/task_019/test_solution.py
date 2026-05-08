from solution import avg_int


def test_basic():
    assert avg_int(4, 6) == 5


def test_same():
    assert avg_int(7, 7) == 7


def test_zero():
    assert avg_int(0, 0) == 0
