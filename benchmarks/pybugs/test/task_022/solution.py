def first_n_pow_two(n):
    """Return [2**0, 2**1, ..., 2**(n-1)] (n elements)."""
    return [2 ** i for i in range(n + 1)]
