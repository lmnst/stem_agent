from solution import has_negative


def test_yes():
    assert has_negative([1, 2, -3, 4]) is True


def test_no():
    assert has_negative([1, 2, 3]) is False


def test_empty():
    assert has_negative([]) is False
