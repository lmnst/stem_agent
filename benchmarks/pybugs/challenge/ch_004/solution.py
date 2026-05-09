def safe_div(a, b):
    """Return a / b, or 0.0 when b == 0 (a missing-branch repair)."""
    return a / b
