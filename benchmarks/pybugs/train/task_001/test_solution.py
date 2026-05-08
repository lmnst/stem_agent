from solution import count_in_range


def test_basic():
    assert count_in_range([1, 2, 3, 4, 5], 2, 4) == 3


def test_full_inclusive():
    assert count_in_range([0, 1, 2, 3], 0, 3) == 4


def test_empty():
    assert count_in_range([], 0, 10) == 0
