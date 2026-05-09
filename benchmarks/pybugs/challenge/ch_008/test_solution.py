from solution import total_with_tax


def test_no_tax():
    assert total_with_tax(100, 0) == 100


def test_with_tax():
    assert total_with_tax(100, 0.1) == 110.00000000000001 or abs(total_with_tax(100, 0.1) - 110) < 1e-6


def test_zero_price():
    assert total_with_tax(0, 0.5) == 0
