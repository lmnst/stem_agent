"""Wilson 95% CI behaves correctly at boundary cases."""
import math

import pytest

from stem_agent.stats import fmt_rate, wilson_interval


def test_wilson_zero_observations_returns_zero_zero():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_full_pass_lower_bound_strictly_below_one():
    lo, hi = wilson_interval(12, 12)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_full_fail_upper_bound_strictly_above_zero():
    lo, hi = wilson_interval(0, 12)
    assert math.isclose(lo, 0.0, abs_tol=1e-12)
    assert 0.0 < hi < 1.0


def test_wilson_midpoint_centered_around_proportion():
    lo, hi = wilson_interval(6, 12)
    centre = (lo + hi) / 2
    assert abs(centre - 0.5) < 0.05


def test_wilson_known_value_9_of_12():
    """Known Wilson 95% CI for 9/12 (~0.75) approx [0.4677, 0.9111]."""
    lo, hi = wilson_interval(9, 12)
    assert math.isclose(lo, 0.4677, abs_tol=0.001)
    assert math.isclose(hi, 0.9111, abs_tol=0.001)


def test_wilson_invalid_args_raise():
    with pytest.raises(ValueError):
        wilson_interval(13, 12)
    with pytest.raises(ValueError):
        wilson_interval(-1, 12)


def test_fmt_rate_renders_ci():
    s = fmt_rate(9, 12)
    assert "9/12" in s
    assert "75.0" in s
    assert "[" in s and "]" in s


def test_fmt_rate_zero_n_does_not_divide_by_zero():
    s = fmt_rate(0, 0)
    assert "n/a" in s
