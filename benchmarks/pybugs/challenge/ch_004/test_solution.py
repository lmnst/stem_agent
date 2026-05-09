from solution import safe_div


def test_normal():
    assert safe_div(10, 2) == 5.0


def test_zero_denominator():
    assert safe_div(5, 0) == 0.0
