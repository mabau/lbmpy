import sympy as sp
from lbmpy.moments import get_default_moment_set_for_stencil, extract_monomials, statistical_quantity_symbol
import pytest

from lbmpy.stencils import get_stencil
from lbmpy.methods.momentbased.moment_transforms import (
    PdfsToCentralMomentsByMatrix, FastCentralMomentTransform, PdfsToCentralMomentsByShiftMatrix,
    PRE_COLLISION_CENTRAL_MOMENT, POST_COLLISION_CENTRAL_MOMENT
)


@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q19', 'D3Q27'])
def test_forward_transform(stencil):
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    q = len(stencil)
    moment_exponents = list(extract_monomials(get_default_moment_set_for_stencil(stencil)))
    moment_exponents = sorted(moment_exponents, key=sum)
    pdfs = sp.symbols(f"f_:{q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{dim}")

    matrix_transform = PdfsToCentralMomentsByMatrix(stencil, moment_exponents, rho, u)
    fast_transform = FastCentralMomentTransform(stencil, moment_exponents, rho, u)
    shift_transform = PdfsToCentralMomentsByShiftMatrix(stencil, moment_exponents, rho, u)

    f_to_k_matrix = matrix_transform.forward_transform(pdfs, moment_symbol_base=PRE_COLLISION_CENTRAL_MOMENT)
    f_to_k_matrix = f_to_k_matrix.new_without_subexpressions().main_assignments_dict

    f_to_k_fast = fast_transform.forward_transform(pdfs, moment_symbol_base=PRE_COLLISION_CENTRAL_MOMENT)
    f_to_k_fast = f_to_k_fast.new_without_subexpressions().main_assignments_dict

    f_to_k_shift = shift_transform.forward_transform(pdfs, moment_symbol_base=PRE_COLLISION_CENTRAL_MOMENT,
                                                     simplification=False)
    f_to_k_shift = f_to_k_shift.new_without_subexpressions().main_assignments_dict

    for e in moment_exponents:
        moment_symbol = statistical_quantity_symbol(PRE_COLLISION_CENTRAL_MOMENT, e)
        rhs_matrix = f_to_k_matrix[moment_symbol].expand()
        rhs_fast = f_to_k_fast[moment_symbol].expand()
        rhs_shift = f_to_k_shift[moment_symbol].expand()
        assert (rhs_matrix - rhs_fast) == 0, f"Mismatch between matrix and fast transform at {moment_symbol}."
        assert (rhs_matrix - rhs_shift) == 0, f"Mismatch between matrix and shift-matrix transform at {moment_symbol}."


@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q19', 'D3Q27'])
def test_backward_transform(stencil):
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    q = len(stencil)
    moment_exponents = list(extract_monomials(get_default_moment_set_for_stencil(stencil)))
    moment_exponents = sorted(moment_exponents, key=sum)
    pdfs = sp.symbols(f"f_:{q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{dim}")

    matrix_transform = PdfsToCentralMomentsByMatrix(stencil, moment_exponents, rho, u)
    fast_transform = FastCentralMomentTransform(stencil, moment_exponents, rho, u)
    shift_transform = PdfsToCentralMomentsByShiftMatrix(stencil, moment_exponents, rho, u)

    k_to_f_matrix = matrix_transform.backward_transform(pdfs, moment_symbol_base=POST_COLLISION_CENTRAL_MOMENT)
    k_to_f_matrix = k_to_f_matrix.new_without_subexpressions().main_assignments_dict

    k_to_f_fast = fast_transform.backward_transform(pdfs, moment_symbol_base=POST_COLLISION_CENTRAL_MOMENT)
    k_to_f_fast = k_to_f_fast.new_without_subexpressions().main_assignments_dict

    k_to_f_shift = shift_transform.backward_transform(pdfs, moment_symbol_base=POST_COLLISION_CENTRAL_MOMENT)
    k_to_f_shift = k_to_f_shift.new_without_subexpressions().main_assignments_dict

    for f in pdfs:
        rhs_matrix = k_to_f_matrix[f].expand()
        rhs_fast = k_to_f_fast[f].expand()
        rhs_shift = k_to_f_shift[f].expand()
        assert (rhs_matrix - rhs_fast) == 0, f"Mismatch between matrix and fast transform at {f}."
        assert (rhs_matrix - rhs_shift) == 0, f"Mismatch between matrix and shift-matrix transform at {f}."
