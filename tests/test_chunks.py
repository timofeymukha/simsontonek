from simsontonek.chunks import chunks_and_offsets


def test_chunks_single_proc():
    np = 1

    n = 10
    c, o = chunks_and_offsets(np, n)

    assert n == c
    assert o == 0


def test_chunks_n_equals_proc():
    np = 10
    n = 10
    c, o = chunks_and_offsets(np, n)

    #    assert n == c
    #    assert o == 0

    pass
