from solution import signs_match


def test_both_positive():
    assert signs_match(2, 3) is True


def test_both_negative():
    assert signs_match(-2, -3) is True


def test_mixed():
    assert signs_match(-1, 1) is False
