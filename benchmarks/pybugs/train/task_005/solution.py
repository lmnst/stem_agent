def find_min(nums):
    """Return the minimum of nums, or None if empty."""
    if not nums:
        return None
    best = nums[0]
    for x in nums[1:]:
        if x > best:
            best = x
    return best
