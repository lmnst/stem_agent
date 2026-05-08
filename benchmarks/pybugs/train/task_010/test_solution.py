from solution import repeat_string


def test_basic():
    assert repeat_string("ab", 3) == "ababab"


def test_one():
    assert repeat_string("x", 1) == "x"


def test_zero():
    assert repeat_string("y", 0) == ""
