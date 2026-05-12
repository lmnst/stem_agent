"""Small statistics helpers used for evaluation honesty.

`wilson_interval` produces a Wilson score 95% confidence interval for a
binomial proportion. Use this instead of normal-approximation intervals
because the test split is small (n=12) and the proportion is often near
0 or 1, where the normal approximation overcovers and lies. See
Wilson (1927); the formula is in any standard reference.
"""
from __future__ import annotations

import math
from typing import Tuple


# Two-sided 95% z-score. Hard-coded to keep this module dependency-free.
_Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = _Z_95) -> Tuple[float, float]:
    """Return (lo, hi), the Wilson score 95% CI for `successes`/`n`.

    Returns (0.0, 0.0) when n == 0 (no observations). Both bounds are
    clipped to [0, 1] so the report is always a valid probability.
    """
    if n <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > n:
        raise ValueError(
            f"successes={successes} out of range for n={n}"
        )
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    halfwidth = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    lo = max(0.0, centre - halfwidth)
    hi = min(1.0, centre + halfwidth)
    return (lo, hi)


def fmt_rate(successes: int, n: int) -> str:
    """Format a pass rate with its Wilson 95% CI as 'k/n = pp.pp% [lo, hi]'."""
    if n <= 0:
        return "0/0 = n/a"
    rate = successes / n
    lo, hi = wilson_interval(successes, n)
    return f"{successes}/{n} = {rate * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
