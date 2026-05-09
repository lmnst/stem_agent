from solution import is_empty


def test_empty():
    assert is_empty([]) is True


def test_nonempty():
    assert is_empty([1]) is False
