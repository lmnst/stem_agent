from solution import cube


def test_two():
    assert cube(2) == 8


def test_three():
    assert cube(3) == 27


def test_zero():
    assert cube(0) == 0


def test_negative():
    assert cube(-2) == -8
