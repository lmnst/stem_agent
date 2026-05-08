from solution import even_indices


def test_basic():
    assert even_indices([10, 20, 30, 40, 50]) == [10, 30, 50]


def test_two():
    assert even_indices([1, 2]) == [1]


def test_empty():
    assert even_indices([]) == []
