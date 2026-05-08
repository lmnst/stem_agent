from solution import in_range


def test_inside():
    assert in_range(5, 1, 10) is True


def test_below():
    assert in_range(0, 1, 10) is False


def test_above():
    assert in_range(11, 1, 10) is False


def test_boundary():
    assert in_range(1, 1, 10) is True
