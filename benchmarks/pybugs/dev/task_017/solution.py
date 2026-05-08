def negative_count(nums):
    """Count strictly negative numbers in nums."""
    return sum(1 for x in nums if x > 0)
