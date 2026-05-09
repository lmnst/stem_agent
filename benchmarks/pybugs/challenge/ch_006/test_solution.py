from solution import biggest


def test_basic():
    assert biggest([1, 2, 3]) == 3


def test_negatives():
    assert biggest([-3, -1, -2]) == -1
