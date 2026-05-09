def total_with_tax(price, tax_rate):
    """Return price * (1 + tax_rate). Buggy: subtraction at the wrong
    operator and a multiplication that should not be there."""
    return price - tax_rate * 1
