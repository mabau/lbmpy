import sympy as sp
from pystencils.stencil import have_same_entries
from lbmpy.moments import get_default_moment_set_for_stencil
from lbmpy.methods.creationfunctions import cascaded_moment_sets_literature
import pytest

from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil
from lbmpy.moment_transforms import (
    PdfsToCentralMomentsByMatrix, FastCentralMomentTransform,
    BinomialChimeraTransform, PdfsToCentralMomentsByShiftMatrix)


@pytest.mark.parametrize('central_moments', ['monomial', 'polynomial'])
@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('transform_class', [BinomialChimeraTransform, FastCentralMomentTransform, PdfsToCentralMomentsByShiftMatrix])
def test_forward_transform(central_moments, stencil, transform_class):
    stencil = LBStencil(stencil)
    if central_moments == 'monomial':
        moment_polynomials = get_default_moment_set_for_stencil(stencil)
    elif central_moments == 'polynomial':
        moment_polynomials = [item for sublist in cascaded_moment_sets_literature(stencil)
                              for item in sublist]
    pdfs = sp.symbols(f"f_:{stencil.Q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{stencil.D}")

    matrix_transform = PdfsToCentralMomentsByMatrix(stencil, moment_polynomials, rho, u)
    test_transform = transform_class(stencil, moment_polynomials, rho, u)
   
    if central_moments == 'monomial' and not have_same_entries(stencil, LBStencil(Stencil.D3Q15)):
        assert test_transform.mono_to_poly_matrix == sp.eye(stencil.Q)
    else:
        assert not test_transform.mono_to_poly_matrix == sp.eye(stencil.Q)

    f_to_k_matrix = matrix_transform.forward_transform(pdfs)
    f_to_k_matrix = f_to_k_matrix.new_without_subexpressions().main_assignments_dict

    f_to_k_test = test_transform.forward_transform(pdfs)
    f_to_k_test = f_to_k_test.new_without_subexpressions().main_assignments_dict

    cm_symbols = matrix_transform.pre_collision_symbols

    for moment_symbol in cm_symbols:
        rhs_matrix = f_to_k_matrix[moment_symbol].expand()
        rhs_test = f_to_k_test[moment_symbol].expand()
        assert (rhs_matrix - rhs_test) == 0, \
            f"Mismatch between matrix transform and {transform_class.__name__} at {moment_symbol}."


@pytest.mark.parametrize('central_moments', ['monomial', 'polynomial'])
@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('transform_class', [BinomialChimeraTransform, FastCentralMomentTransform, PdfsToCentralMomentsByShiftMatrix])
def test_backward_transform(central_moments, stencil, transform_class):
    stencil = LBStencil(stencil)
    if central_moments == 'monomial':
        moment_polynomials = get_default_moment_set_for_stencil(stencil)
    elif central_moments == 'polynomial':
        moment_polynomials = [item for sublist in cascaded_moment_sets_literature(stencil)
                              for item in sublist]
    pdfs = sp.symbols(f"f_:{stencil.Q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{stencil.D}")

    matrix_transform = PdfsToCentralMomentsByMatrix(stencil, moment_polynomials, rho, u)
    test_transform = transform_class(stencil, moment_polynomials, rho, u)

    k_to_f_matrix = matrix_transform.backward_transform(pdfs)
    k_to_f_matrix = k_to_f_matrix.new_without_subexpressions().main_assignments_dict

    k_to_f_test = test_transform.backward_transform(pdfs)
    k_to_f_test = k_to_f_test.new_without_subexpressions().main_assignments_dict

    for f in pdfs:
        rhs_matrix = k_to_f_matrix[f].expand()
        rhs_test = k_to_f_test[f].expand()
        assert (rhs_matrix - rhs_test) == 0, \
            f"Mismatch between matrix transform and {transform_class.__name__} at {f}."
