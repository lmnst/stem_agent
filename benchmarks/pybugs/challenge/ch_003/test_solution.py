from solution import f


def test_basic():
    assert f(5, 3) == 3


def test_zero():
    # 0 - 0 + 1 == 1; check that ANY single-site swap that happens to
    # match one test does not also match this one.
    assert f(0, 0) == 1


def test_neg():
    assert f(2, 7) == -4
