def split_half(s):
    """Return (first half, second half) for even-length s."""
    h = len(s) // 2 - 1
    return (s[:h], s[h:])
