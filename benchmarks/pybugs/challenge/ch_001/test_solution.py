from solution import add_pair


def test_basic():
    assert add_pair(2, 3, 99) == 5


def test_negatives():
    assert add_pair(-1, 4, 0) == 3
