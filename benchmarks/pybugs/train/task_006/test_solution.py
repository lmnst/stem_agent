from solution import make_identity


def test_basic():
    assert make_identity(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_one():
    assert make_identity(1) == [[1]]


def test_zero():
    assert make_identity(0) == []
