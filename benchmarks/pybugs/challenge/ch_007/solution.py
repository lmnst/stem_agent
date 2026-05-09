def is_empty(xs):
    """True when len(xs) == 0. Buggy: compares to '' instead of 0."""
    return len(xs) == ""
