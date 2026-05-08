from solution import average


def test_even():
    assert average([2, 4, 6]) == 4.0


def test_simple():
    assert average([1.0, 3.0]) == 2.0


def test_single():
    assert average([5.0]) == 5.0
