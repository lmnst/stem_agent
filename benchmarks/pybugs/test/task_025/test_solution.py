from solution import is_ascending


def test_yes():
    assert is_ascending([1, 2, 3]) is True


def test_equal():
    assert is_ascending([1, 2, 2, 3]) is False


def test_descending():
    assert is_ascending([3, 2, 1]) is False


def test_single():
    assert is_ascending([5]) is True
