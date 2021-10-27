import warnings

import pytest
import sympy as sp
import matplotlib.pyplot as plt

import pystencils as ps
from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil


def get_3d_stencils():
    return LBStencil(Stencil.D3Q15), LBStencil(Stencil.D3Q19), LBStencil(Stencil.D3Q27)


def get_all_stencils():
    return [
        LBStencil(Stencil.D2Q9, 'walberla'),
        LBStencil(Stencil.D3Q15, 'walberla'),
        LBStencil(Stencil.D3Q19, 'walberla'),
        LBStencil(Stencil.D3Q27, 'walberla'),

        LBStencil(Stencil.D2Q9, 'counterclockwise'),

        LBStencil(Stencil.D2Q9, 'braunschweig'),
        LBStencil(Stencil.D3Q19, 'braunschweig'),

        LBStencil(Stencil.D3Q27, 'premnath'),

        LBStencil(Stencil.D3Q27, "fakhari"),
    ]


def test_sizes():

    assert LBStencil(Stencil.D2Q9).Q == 9
    assert LBStencil(Stencil.D3Q15).Q == 15
    assert LBStencil(Stencil.D3Q19).Q == 19
    assert LBStencil(Stencil.D3Q27).Q == 27


def test_dimensionality():
    for d in LBStencil(Stencil.D2Q9).stencil_entries:
        assert len(d) == 2

    assert LBStencil(Stencil.D2Q9).D == 2

    for stencil in get_3d_stencils():
        assert stencil.D == 3


def test_uniqueness():
    for stencil in get_3d_stencils():
        direction_set = set(stencil.stencil_entries)
        assert len(direction_set) == len(stencil.stencil_entries)


def test_run_self_check():
    for st in get_all_stencils():
        assert ps.stencil.is_valid(st.stencil_entries, max_neighborhood=1)
        assert ps.stencil.is_symmetric(st.stencil_entries)


def test_inverse_direction():
    stencil = LBStencil(Stencil.D2Q9)

    for i in range(stencil.Q):
        assert ps.stencil.inverse_direction(stencil.stencil_entries[i]) == stencil.inverse_stencil_entries[i]


def test_free_functions():
    assert not ps.stencil.is_symmetric([(1, 0), (0, 1)])
    assert not ps.stencil.is_valid([(1, 0), (1, 1, 0)])
    assert not ps.stencil.is_valid([(2, 0), (0, 1)], max_neighborhood=1)

    with pytest.raises(ValueError) as e:
        LBStencil("name_that_does_not_exist")
    assert "No such stencil" in str(e.value)


def test_visualize():
    plt.clf()
    plt.cla()

    d2q9, d3q19 = LBStencil(Stencil.D2Q9), LBStencil(Stencil.D3Q19)
    figure = plt.gcf()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d2q9.plot(figure=figure, data=[str(i) for i in range(9)])
        d3q19.plot(figure=figure, data=sp.symbols("a_:19"))


def test_comparability_of_stencils():
    stencil_1 = LBStencil(Stencil.D2Q9)
    stencil_2 = LBStencil(Stencil.D2Q9)
    stencil_3 = LBStencil(Stencil.D2Q9, ordering="braunschweig")
    stencil_4 = LBStencil(stencil_1.stencil_entries)
    stencil_5 = LBStencil(stencil_3.stencil_entries)
    stencil_6 = LBStencil(stencil_1.stencil_entries)

    assert stencil_1 == stencil_2
    assert stencil_1 != stencil_3
    assert stencil_1 != stencil_4
    assert stencil_1 != stencil_5
    assert stencil_4 == stencil_6
