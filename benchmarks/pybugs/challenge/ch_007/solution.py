def normalize_to_unit_sum(xs):
    """Return xs scaled so the result sums to 1.0. Buggy: divides by
    the count of elements (len) instead of by their sum, producing
    mean-normalized values rather than unit-sum-normalized values."""
    n = len(xs)
    return [x / n for x in xs]
