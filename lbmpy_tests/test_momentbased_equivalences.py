import pytest
import sympy as sp
from sympy.polys.polytools import monic

from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule
from lbmpy.moments import (
    extract_monomials, get_default_moment_set_for_stencil, non_aliased_polynomial_raw_moments,
    exponent_tuple_sort_key)
from lbmpy.stencils import get_stencil

from lbmpy.methods.creationfunctions import mrt_orthogonal_modes_literature

from lbmpy.moment_transforms import (
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform
)


@pytest.mark.parametrize('stencil', ['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'])
@pytest.mark.parametrize('monomials', [False, True])
def test_moment_transform_equivalences(stencil, monomials):
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    q = len(stencil)

    pdfs = sp.symbols(f"f_:{q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{dim}")

    moment_polynomials = get_default_moment_set_for_stencil(stencil)
    if monomials:
        polys_nonaliased = non_aliased_polynomial_raw_moments(moment_polynomials, stencil)
        moment_exponents = sorted(extract_monomials(polys_nonaliased), key=exponent_tuple_sort_key)
        moment_polynomials = None
    else:
        moment_exponents = None

    matrix_transform = PdfsToMomentsByMatrixTransform(stencil, moment_polynomials, rho, u, moment_exponents=moment_exponents)
    chimera_transform = PdfsToMomentsByChimeraTransform(stencil, moment_polynomials, rho, u, moment_exponents=moment_exponents)

    f_to_m_matrix = matrix_transform.forward_transform(pdfs, return_monomials=monomials)
    f_to_m_matrix = f_to_m_matrix.new_without_subexpressions().main_assignments_dict

    f_to_m_chimera = chimera_transform.forward_transform(pdfs, return_monomials=monomials)
    f_to_m_chimera = f_to_m_chimera.new_without_subexpressions().main_assignments_dict

    m_to_f_matrix = matrix_transform.backward_transform(pdfs, start_from_monomials=monomials)
    m_to_f_matrix = m_to_f_matrix.new_without_subexpressions().main_assignments_dict

    m_to_f_chimera = chimera_transform.backward_transform(pdfs, start_from_monomials=monomials)
    m_to_f_chimera = m_to_f_chimera.new_without_subexpressions().main_assignments_dict

    m_pre_matrix = matrix_transform.pre_collision_monomial_symbols if monomials else matrix_transform.pre_collision_symbols
    m_pre_chimera = chimera_transform.pre_collision_monomial_symbols if monomials else chimera_transform.pre_collision_symbols

    for m1, m2 in zip(m_pre_matrix, m_pre_chimera):
        rhs_matrix = f_to_m_matrix[m1]
        rhs_chimera = f_to_m_chimera[m2]
        diff = (rhs_matrix - rhs_chimera).expand()
        assert diff == 0, f"Mismatch between matrix and chimera forward transform at {m1}, {m2}."

    for f in pdfs:
        rhs_matrix = m_to_f_matrix[f]
        rhs_chimera = m_to_f_chimera[f]
        diff = (rhs_matrix - rhs_chimera).expand()
        assert diff == 0, f"Mismatch between matrix and chimera backward transform at {f}"


d3q15_literature = mrt_orthogonal_modes_literature(get_stencil('D3Q15'), True)
d3q19_literature = mrt_orthogonal_modes_literature(get_stencil('D3Q19'), True)

setups = [
    ('D2Q9', 'srt', None, 'Guo'), 
    ('D2Q9', 'mrt', None, 'Simple'), 
    ('D3Q15', 'mrt', None, 'Simple'),
    ('D3Q15', 'mrt', d3q15_literature, 'Simple'),
    ('D3Q19', 'trt', None, 'Simple'),
    ('D3Q19', 'mrt', d3q19_literature, 'Guo'),
    ('D3Q27', 'srt', None, 'Guo')
]

@pytest.mark.parametrize('setup', setups)
def test_population_and_moment_space_equivalence(setup):
    stencil = get_stencil(setup[0])
    d = len(stencil[0])
    q = len(stencil)
    method = setup[1]
    nested_moments = setup[2]
    fmodel = setup[3]
    force = sp.symbols(f'F_:{d}')

    params = {
        'stencil': stencil,
        'method': method,
        'relaxation_rates': sp.symbols(f'omega_:{q}'),
        'nested_moments' : nested_moments,
        'force_model': fmodel,
        'force': force,
        'weighted' : True,
        'compressible': True
    }

    optimization = {
        'cse_global': False,
        'cse_pdfs': False,
        'pre_simplification': True,
    }

    lb_method_moment_space = create_lb_method(moment_transform_class=PdfsToMomentsByChimeraTransform, **params)
    lb_method_pdf_space = create_lb_method(moment_transform_class=None, **params)

    rho = lb_method_moment_space.zeroth_order_equilibrium_moment_symbol
    u = lb_method_moment_space.first_order_equilibrium_moment_symbols
    keep = set((rho,) + u)

    optimization['simplification'] = False
    cr_moment_space = create_lb_collision_rule(lb_method=lb_method_moment_space, optimization=optimization)
    cr_moment_space = cr_moment_space.new_without_subexpressions(subexpressions_to_keep=keep)

    optimization['simplification'] = 'auto'
    cr_pdf_space = create_lb_collision_rule(lb_method=lb_method_pdf_space, optimization=optimization)
    cr_pdf_space = cr_pdf_space.new_without_subexpressions(subexpressions_to_keep=keep)

    for a,b in zip(cr_moment_space.main_assignments, cr_pdf_space.main_assignments):
        diff = (a.rhs - b.rhs).expand()
        assert diff == 0, f"Mismatch between population- and moment-space equations in PDFs {a.lhs}, {b.lhs}"
