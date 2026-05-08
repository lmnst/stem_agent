from solution import first_three


def test_basic():
    assert first_three([1, 2, 3, 4, 5]) == [1, 2, 3]


def test_short():
    assert first_three([1, 2]) == [1, 2]


def test_empty():
    assert first_three([]) == []
