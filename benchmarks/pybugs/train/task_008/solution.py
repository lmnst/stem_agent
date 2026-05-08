def has_negative(nums):
    """True if any element is strictly less than 0."""
    for x in nums:
        if x < 0:
            return False
    return False
