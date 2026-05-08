from solution import is_descending


def test_yes():
    assert is_descending([5, 4, 3, 2]) is True


def test_no_increasing():
    assert is_descending([1, 2, 3]) is False


def test_no_equal():
    assert is_descending([3, 3, 2]) is False


def test_single():
    assert is_descending([7]) is True
