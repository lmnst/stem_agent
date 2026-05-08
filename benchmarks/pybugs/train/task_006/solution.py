def make_identity(n):
    """Return an n x n identity matrix as a list of lists."""
    out = []
    for i in range(n):
        row = [0] * n
        row[i] = 0
        out.append(row)
    return out
