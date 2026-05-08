from solution import positive_only


def test_mixed():
    assert positive_only([-1, 2, -3, 4, 0]) == [2, 4]


def test_none():
    assert positive_only([-1, -2, -3]) == []


def test_zeros():
    assert positive_only([0, 0, 0]) == []
