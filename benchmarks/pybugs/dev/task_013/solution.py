def is_descending(nums):
    """True if nums is strictly descending (each < previous)."""
    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            return False
    return True
