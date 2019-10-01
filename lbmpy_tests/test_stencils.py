import itertools
import warnings

import pytest
import sympy as sp

import lbmpy.stencils as s
import pystencils as ps
from lbmpy.stencils import get_stencil


def get_3d_stencils():
    return s.get_stencil('D3Q15'), s.get_stencil('D3Q19'), s.get_stencil('D3Q27')


def get_all_stencils():
    return [
        s.get_stencil('D2Q9', 'walberla'),
        s.get_stencil('D3Q15', 'walberla'),
        s.get_stencil('D3Q19', 'walberla'),
        s.get_stencil('D3Q27', 'walberla'),

        s.get_stencil('D2Q9', 'counterclockwise'),

        s.get_stencil('D2Q9', 'braunschweig'),
        s.get_stencil('D3Q19', 'braunschweig'),

        s.get_stencil('D3Q27', 'premnath'),

        s.get_stencil("D3Q27", "fakhari"),
    ]


def test_sizes():
    assert len(s.get_stencil('D2Q9')) == 9
    assert len(s.get_stencil('D3Q15')) == 15
    assert len(s.get_stencil('D3Q19')) == 19
    assert len(s.get_stencil('D3Q27')) == 27


def test_dimensionality():
    for d in s.get_stencil('D2Q9'):
        assert len(d) == 2

    for d in itertools.chain(*get_3d_stencils()):
        assert len(d) == 3


def test_uniqueness():
    for stencil in get_3d_stencils():
        direction_set = set(stencil)
        assert len(direction_set) == len(stencil)


def test_run_self_check():
    for st in get_all_stencils():
        assert ps.stencil.is_valid(st, max_neighborhood=1)
        assert ps.stencil.is_symmetric(st)


def test_inverse_direction():
    assert ps.stencil.inverse_direction((1, 0, -1)), (-1, 0 == 1)


def test_free_functions():
    assert not ps.stencil.is_symmetric([(1, 0), (0, 1)])
    assert not ps.stencil.is_valid([(1, 0), (1, 1, 0)])
    assert not ps.stencil.is_valid([(2, 0), (0, 1)], max_neighborhood=1)

    with pytest.raises(ValueError) as e:
        get_stencil("name_that_does_not_exist")
    assert "No such stencil" in str(e.value)


def test_visualize():
    import matplotlib.pyplot as plt
    plt.clf()
    plt.cla()

    d2q9, d3q19 = get_stencil("D2Q9"), get_stencil("D3Q19")
    figure = plt.gcf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps.stencil.plot(d2q9, figure=figure, data=[str(i) for i in range(9)])
        ps.stencil.plot(d3q19, figure=figure, data=sp.symbols("a_:19"))
