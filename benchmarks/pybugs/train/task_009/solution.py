def first_n_squares(n):
    """Return [1, 4, 9, ..., n*n]."""
    out = []
    i = 1
    while i < n:
        out.append(i * i)
        i += 1
    return out
