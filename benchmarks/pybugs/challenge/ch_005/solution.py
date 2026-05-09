def sum_first_n(n):
    """Return 1 + 2 + ... + n. Buggy: sum runs to n - 1 because the
    range upper bound is one too small."""
    total = 0
    for i in range(1, n):
        total = total + i
    return total
