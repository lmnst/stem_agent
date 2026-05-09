from solution import normalize_to_unit_sum


def test_three_distinct():
    out = normalize_to_unit_sum([1.0, 2.0, 3.0])
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, [1 / 6, 2 / 6, 3 / 6]))


def test_two_distinct():
    out = normalize_to_unit_sum([3.0, 1.0])
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, [0.75, 0.25]))


def test_pair_of_twos():
    out = normalize_to_unit_sum([2.0, 2.0])
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, [0.5, 0.5]))
