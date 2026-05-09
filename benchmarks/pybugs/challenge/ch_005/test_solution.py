from solution import reverse_list


def test_basic():
    assert reverse_list([1, 2, 3]) == [3, 2, 1]


def test_empty():
    assert reverse_list([]) == []
