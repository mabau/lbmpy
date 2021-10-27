import pytest
import sympy as sp
from dataclasses import replace

from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule, LBMConfig, LBMOptimisation
from lbmpy.enums import Method, ForceModel, Stencil
from lbmpy.moments import (
    extract_monomials, get_default_moment_set_for_stencil, non_aliased_polynomial_raw_moments,
    exponent_tuple_sort_key)
from lbmpy.stencils import LBStencil

from lbmpy.methods.default_moment_sets import mrt_orthogonal_modes_literature

from lbmpy.moment_transforms import (
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform
)


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('monomials', [False, True])
def test_moment_transform_equivalences(stencil, monomials):
    stencil = LBStencil(stencil)

    pdfs = sp.symbols(f"f_:{stencil.Q}")
    rho = sp.Symbol('rho')
    u = sp.symbols(f"u_:{stencil.D}")

    moment_polynomials = get_default_moment_set_for_stencil(stencil)
    if monomials:
        polys_nonaliased = non_aliased_polynomial_raw_moments(moment_polynomials, stencil)
        moment_exponents = sorted(extract_monomials(polys_nonaliased), key=exponent_tuple_sort_key)
        moment_polynomials = None
    else:
        moment_exponents = None

    matrix_transform = PdfsToMomentsByMatrixTransform(stencil, moment_polynomials, rho, u,
                                                      moment_exponents=moment_exponents)
    chimera_transform = PdfsToMomentsByChimeraTransform(stencil, moment_polynomials, rho, u,
                                                        moment_exponents=moment_exponents)

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


d3q15_literature = mrt_orthogonal_modes_literature(LBStencil(Stencil.D3Q15), True)
d3q19_literature = mrt_orthogonal_modes_literature(LBStencil(Stencil.D3Q19), True)

setups = [
    (Stencil.D2Q9, Method.SRT, None, ForceModel.GUO),
    (Stencil.D2Q9, Method.MRT, None, ForceModel.SIMPLE),
    (Stencil.D3Q15, Method.MRT, None, ForceModel.SIMPLE),
    (Stencil.D3Q15, Method.MRT, d3q15_literature, ForceModel.SIMPLE),
    (Stencil.D3Q19, Method.TRT, None, ForceModel.SIMPLE),
    (Stencil.D3Q19, Method.MRT, d3q19_literature, ForceModel.GUO),
    (Stencil.D3Q27, Method.SRT, None, ForceModel.GUO)
]


@pytest.mark.parametrize('setup', setups)
def test_population_and_moment_space_equivalence(setup):
    stencil = LBStencil(setup[0])
    method = setup[1]
    nested_moments = setup[2]
    fmodel = setup[3]
    force = sp.symbols(f'F_:{stencil.D}')
    conserved_moments = 1 + stencil.D

    rr = [*[0] * conserved_moments, *sp.symbols(f'omega_:{stencil.Q - conserved_moments}')]
    lbm_config = LBMConfig(stencil=stencil, method=method, relaxation_rates=rr,
                           nested_moments=nested_moments, force_model=fmodel, force=force,
                           weighted=True, compressible=True, moment_transform_class=PdfsToMomentsByChimeraTransform)

    lbm_opt = LBMOptimisation(cse_global=False, cse_pdfs=False, pre_simplification=True, simplification=False)

    lb_method_moment_space = create_lb_method(lbm_config=lbm_config)

    lbm_config = replace(lbm_config, moment_transform_class=None)
    lb_method_pdf_space = create_lb_method(lbm_config=lbm_config)

    rho = lb_method_moment_space.zeroth_order_equilibrium_moment_symbol
    u = lb_method_moment_space.first_order_equilibrium_moment_symbols
    keep = set((rho,) + u)
    cr_moment_space = create_lb_collision_rule(lb_method=lb_method_moment_space, lbm_optimisation=lbm_opt)
    cr_moment_space = cr_moment_space.new_without_subexpressions(subexpressions_to_keep=keep)

    lbm_opt = replace(lbm_opt, simplification='auto')
    cr_pdf_space = create_lb_collision_rule(lb_method=lb_method_pdf_space, lbm_optimisation=lbm_opt)
    cr_pdf_space = cr_pdf_space.new_without_subexpressions(subexpressions_to_keep=keep)

    for a, b in zip(cr_moment_space.main_assignments, cr_pdf_space.main_assignments):
        diff = (a.rhs - b.rhs).expand()
        assert diff == 0, f"Mismatch between population- and moment-space equations in PDFs {a.lhs}, {b.lhs}"
