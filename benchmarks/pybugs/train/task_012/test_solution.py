from solution import divmod_first


def test_basic():
    assert divmod_first(7, 2) == (3, 1)


def test_exact():
    assert divmod_first(10, 5) == (2, 0)


def test_one():
    assert divmod_first(5, 1) == (5, 0)
