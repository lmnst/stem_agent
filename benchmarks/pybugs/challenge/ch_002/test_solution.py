from solution import percent_value


def test_one():
    assert percent_value(1) == 5


def test_ten():
    assert percent_value(10) == 50
