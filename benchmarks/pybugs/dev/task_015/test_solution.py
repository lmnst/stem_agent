from solution import half


def test_even():
    assert half(10) == 5


def test_odd():
    assert half(7) == 3


def test_zero():
    assert half(0) == 0
