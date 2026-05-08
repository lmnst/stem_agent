from solution import matrix_n_zeros


def test_three():
    assert matrix_n_zeros(3) == [0, 0, 0]


def test_one():
    assert matrix_n_zeros(1) == [0]


def test_zero():
    assert matrix_n_zeros(0) == []
