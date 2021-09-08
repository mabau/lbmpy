import pytest
import sympy as sp

from lbmpy.stencils import get_stencil
from lbmpy.moments import get_default_moment_set_for_stencil

from lbmpy.moment_transforms import (
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform,
    PdfsToCentralMomentsByShiftMatrix, PdfsToCentralMomentsByMatrix, FastCentralMomentTransform
)

transforms = [
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform,
    PdfsToCentralMomentsByShiftMatrix, PdfsToCentralMomentsByMatrix, FastCentralMomentTransform
]

@pytest.mark.parametrize('stencil', ['D2Q9'])
@pytest.mark.parametrize('transform_class', transforms)
def test_monomial_equations(stencil, transform_class):
    stencil = get_stencil(stencil)
    rho = sp.symbols("rho")
    u = sp.symbols(f"u_:{len(stencil[0])}")
    moment_polynomials = get_default_moment_set_for_stencil(stencil)
    transform = transform_class(stencil, moment_polynomials, rho, u)
    pdfs = sp.symbols(f"f_:{len(stencil)}")
    fw_eqs = transform.forward_transform(pdfs, return_monomials=True)
    bw_eqs = transform.backward_transform(pdfs, start_from_monomials=True)

    mono_symbols_pre = set(transform.pre_collision_monomial_symbols)
    mono_symbols_post = set(transform.post_collision_monomial_symbols)

    assert mono_symbols_pre <= set(fw_eqs.defined_symbols)
    assert mono_symbols_post <= set(bw_eqs.free_symbols)

    symbols_pre = set(transform.pre_collision_symbols)
    symbols_post = set(transform.post_collision_symbols)

    assert symbols_pre.isdisjoint(set(fw_eqs.atoms(sp.Symbol)))
    assert symbols_post.isdisjoint(set(bw_eqs.atoms(sp.Symbol)))
