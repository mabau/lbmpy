import itertools

from pystencils.sympyextensions import kronecker_delta as kd


def delta4(i, j, k, l):
    """See Silva: Truncation error paper, Eq13.a"""
    return kd(i, j) * kd(k, l) + kd(i, k) * kd(j, l) + kd(i, l) * kd(j, k)


def delta42(*args):
    """See Silva: Truncation error paper, Eq13.b"""
    assert len(args) == 6
    res = 0
    for selected in itertools.combinations(args, 2):
        rest = list(args)
        del rest[rest.index(selected[0])]
        del rest[rest.index(selected[1])]
        res += kd(*selected) * kd(*rest)
    return res


def delta6(i, j, k, l, m, n):
    """See Silva: Truncation error paper, Eq13.c"""
    return kd(i, j) * delta4(k, l, m, n) + \
           kd(i, k) * delta4(j, l, m, n) + \
           kd(i, l) * delta4(j, k, m, n) + \
           kd(i, m) * delta4(j, k, l, n) + \
           kd(i, n) * delta4(j, k, l, m)


def test_rudimentary():
    from pystencils.sympyextensions import multidimensional_sum as s
    assert sum(delta4(*t) for t in s(4, dim=3)) == 27
    assert sum(delta42(*t) for t in s(6, dim=2)) == 60
    assert sum(delta6(*t) for t in s(6, dim=2)) == 120
