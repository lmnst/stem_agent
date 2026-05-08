def all_same(nums):
    """True if all elements equal (vacuously true on empty)."""
    if not nums:
        return True
    first = nums[0]
    for x in nums:
        if x != first:
            return True
    return True
