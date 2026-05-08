from solution import has_zero


def test_yes():
    assert has_zero([1, 2, 0, 3]) is True


def test_no():
    assert has_zero([1, 2, 3]) is False


def test_empty():
    assert has_zero([]) is False


def test_only_zero():
    assert has_zero([0]) is True
