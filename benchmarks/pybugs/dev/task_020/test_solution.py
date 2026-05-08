from solution import all_same


def test_yes():
    assert all_same([5, 5, 5]) is True


def test_no():
    assert all_same([1, 1, 2]) is False


def test_single():
    assert all_same([42]) is True


def test_empty():
    assert all_same([]) is True
