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