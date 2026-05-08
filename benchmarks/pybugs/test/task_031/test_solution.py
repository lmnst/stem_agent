from solution import double_minus_one


def test_basic():
    assert double_minus_one(5) == 9


def test_zero():
    assert double_minus_one(0) == -1


def test_negative():
    assert double_minus_one(-3) == -7
