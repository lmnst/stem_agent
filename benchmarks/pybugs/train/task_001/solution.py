def count_in_range(nums, lo, hi):
    """Count nums where lo <= x <= hi (inclusive on both ends)."""
    c = 0
    for x in nums:
        if lo <= x and x < hi:
            c += 1
    return c
