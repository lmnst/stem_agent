from solution import find_min


def test_basic():
    assert find_min([3, 1, 4, 1, 5, 9, 2, 6]) == 1


def test_single():
    assert find_min([42]) == 42


def test_negative():
    assert find_min([-1, -2, -3]) == -3


def test_empty():
    assert find_min([]) is None
