from solution import is_weekend


def test_sat():
    assert is_weekend("sat") is True


def test_sun():
    assert is_weekend("sun") is True


def test_mon():
    assert is_weekend("mon") is False
