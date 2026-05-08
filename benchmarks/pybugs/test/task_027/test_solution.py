from solution import total_distance


def test_basic():
    assert total_distance(3, 4) == 12


def test_zero():
    assert total_distance(0, 5) == 0


def test_one():
    assert total_distance(1, 7) == 7
