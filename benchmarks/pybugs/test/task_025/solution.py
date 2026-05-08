def is_ascending(nums):
    """True if nums is strictly ascending (each > previous)."""
    for i in range(1, len(nums)):
        if nums[i] < nums[i - 1]:
            return False
    return True
