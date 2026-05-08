from solution import total_with_tax


def test_basic():
    assert total_with_tax(100, 0.1) == 110.0


def test_zero_tax():
    assert total_with_tax(50, 0.0) == 50.0


def test_zero_price():
    assert total_with_tax(0, 0.2) == 0.0
