from solution import split_half


def test_basic():
    assert split_half("abcd") == ("ab", "cd")


def test_six():
    assert split_half("abcdef") == ("abc", "def")


def test_empty():
    assert split_half("") == ("", "")
