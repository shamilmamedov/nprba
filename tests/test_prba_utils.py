from nprba.utils import prba


def test_compute_rfe_lengths():
    L = 1
    n_seg = 1
    lengths = prba.compute_rfe_lengths(L, n_seg)
    expected = [0.5, 0.5]
    assert all(a==b for a, b in zip(lengths, expected))
    assert sum(lengths)==L

    n_seg = 2
    lengths = prba.compute_rfe_lengths(L, n_seg)
    expected = [0.25, 0.5, 0.25]
    assert all(a==b for a, b in zip(lengths, expected))
    assert sum(lengths)==L

    n_seg = 4
    lengths = prba.compute_rfe_lengths(L, n_seg)
    expected = [0.125, 0.25, 0.25, 0.25, 0.125]
    assert all(a==b for a, b in zip(lengths, expected))
    assert sum(lengths)==L

if __name__ == "__main__":
    pass