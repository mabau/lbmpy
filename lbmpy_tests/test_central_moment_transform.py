import sympy as sp
from pystencils.stencil import have_same_entries
from lbmpy.moments import get_default_moment_set_for_stencil, extract_monomials, statistical_quantity_symbol
from lbmpy.methods.creationfunctions import cascaded_moment_sets_literature
import pytest

from lbmpy.stencils import get_stencil
from lbmpy.moment_transforms import (
    PdfsToCentralMomentsByMatrix, FastCentralMomentTransform, PdfsToCentralMomentsByShiftMatrix,
    PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, POST_COLLISION_MONOMIAL_CENTRAL_MOMENT
)


@pytest.mark.parametrize('type', ['monomial', 'polynomial'])
@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'])
def test_forward_transform(type, stencil):
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    q = len(stencil)
    if type == 'monomial':
        moment_polynomials = get_default_moment_set_for_stencil(stencil)
    elif type == 'polynomial':
        moment_polynomials = [item for sublist in cascaded_moment_sets_literature(stencil)
                              for item in sublist]
    pdfs = sp.symbols(f"f_:{q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{dim}")

    matrix_transform = PdfsToCentralMomentsByMatrix(stencil, moment_polynomials, rho, u)
    fast_transform = FastCentralMomentTransform(stencil, moment_polynomials, rho, u)
    shift_transform = PdfsToCentralMomentsByShiftMatrix(stencil, moment_polynomials, rho, u)

    assert shift_transform.moment_exponents == fast_transform.moment_exponents

    if type == 'monomial' and not have_same_entries(stencil, get_stencil('D3Q15')):
        assert fast_transform.mono_to_poly_matrix == sp.eye(q)
        assert shift_transform.mono_to_poly_matrix == sp.eye(q)
    else:
        assert not fast_transform.mono_to_poly_matrix == sp.eye(q)
        assert not shift_transform.mono_to_poly_matrix == sp.eye(q)

    f_to_k_matrix = matrix_transform.forward_transform(pdfs)
    f_to_k_matrix = f_to_k_matrix.new_without_subexpressions().main_assignments_dict

    f_to_k_fast = fast_transform.forward_transform(pdfs)
    f_to_k_fast = f_to_k_fast.new_without_subexpressions().main_assignments_dict

    f_to_k_shift = shift_transform.forward_transform(pdfs, simplification=False)
    f_to_k_shift = f_to_k_shift.new_without_subexpressions().main_assignments_dict

    cm_symbols = matrix_transform.pre_collision_symbols

    for moment_symbol in cm_symbols:
        rhs_matrix = f_to_k_matrix[moment_symbol].expand()
        rhs_fast = f_to_k_fast[moment_symbol].expand()
        rhs_shift = f_to_k_shift[moment_symbol].expand()
        assert (rhs_matrix - rhs_fast) == 0, f"Mismatch between matrix and fast transform at {moment_symbol}."
        assert (rhs_matrix - rhs_shift) == 0, f"Mismatch between matrix and shift-matrix transform at {moment_symbol}."

@pytest.mark.parametrize('type', ['monomial', 'polynomial'])
@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'])
def test_backward_transform(type, stencil):
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    q = len(stencil)
    if type == 'monomial':
        moment_polynomials = get_default_moment_set_for_stencil(stencil)
    elif type == 'polynomial':
        moment_polynomials = [item for sublist in cascaded_moment_sets_literature(stencil)
                              for item in sublist]
    pdfs = sp.symbols(f"f_:{q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{dim}")

    matrix_transform = PdfsToCentralMomentsByMatrix(stencil, moment_polynomials, rho, u)
    fast_transform = FastCentralMomentTransform(stencil, moment_polynomials, rho, u)
    shift_transform = PdfsToCentralMomentsByShiftMatrix(stencil, moment_polynomials, rho, u)

    assert shift_transform.moment_exponents == fast_transform.moment_exponents

    k_to_f_matrix = matrix_transform.backward_transform(pdfs)
    k_to_f_matrix = k_to_f_matrix.new_without_subexpressions().main_assignments_dict

    k_to_f_fast = fast_transform.backward_transform(pdfs)
    k_to_f_fast = k_to_f_fast.new_without_subexpressions().main_assignments_dict

    k_to_f_shift = shift_transform.backward_transform(pdfs)
    k_to_f_shift = k_to_f_shift.new_without_subexpressions().main_assignments_dict

    for f in pdfs:
        rhs_matrix = k_to_f_matrix[f].expand()
        rhs_fast = k_to_f_fast[f].expand()
        rhs_shift = k_to_f_shift[f].expand()
        assert (rhs_matrix - rhs_fast) == 0, f"Mismatch between matrix and fast transform at {f}."
        assert (rhs_matrix - rhs_shift) == 0, f"Mismatch between matrix and shift-matrix transform at {f}."
