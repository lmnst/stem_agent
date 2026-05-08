def has_zero(nums):
    """True if 0 appears in nums."""
    for x in nums:
        if x != 0:
            return True
    return False
