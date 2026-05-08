def signs_match(a, b):
    """True if a and b have the same nonzero sign."""
    return (a > 0 and b > 0) and (a < 0 and b < 0)
