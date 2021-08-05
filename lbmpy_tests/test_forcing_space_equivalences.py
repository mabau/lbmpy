import pytest
import sympy as sp

from lbmpy.creationfunctions import create_lb_method
from lbmpy.stencils import get_stencil
from lbmpy.forcemodels import Simple, Luo, Guo


@pytest.mark.parametrize('fmodel_class', [Simple, Luo, Guo])
def test_forcing_space_equivalences(fmodel_class):
    d, q = 3, 27
    stencil = get_stencil(f"D{d}Q{q}")
    force = sp.symbols(f"F_:{d}")
    fmodel = fmodel_class(force)
    lb_method = create_lb_method(stencil=stencil, method='srt')
    inv_moment_matrix = lb_method.moment_matrix.inv()

    force_pdfs = sp.Matrix(fmodel(lb_method))
    force_moments = fmodel.moment_space_forcing(lb_method)

    diff = (force_pdfs - (inv_moment_matrix * force_moments)).expand()
    for i, d in enumerate(diff):
        assert d == 0, f"Mismatch between population and moment space forcing " \
                       f"in force model {fmodel_class}, population f_{i}"
