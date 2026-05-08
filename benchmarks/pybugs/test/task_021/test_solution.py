from solution import below_threshold


def test_basic():
    assert below_threshold([1, 2, 3, 4, 5], 3) == [1, 2]


def test_strict():
    assert below_threshold([5, 5, 5], 5) == []


def test_empty():
    assert below_threshold([], 10) == []
