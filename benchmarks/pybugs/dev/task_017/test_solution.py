from solution import negative_count


def test_some():
    assert negative_count([1, -2, 3, -4, -5]) == 3


def test_none():
    assert negative_count([1, 2, 3]) == 0


def test_empty():
    assert negative_count([]) == 0


def test_zero():
    assert negative_count([0, -1, 0]) == 1
