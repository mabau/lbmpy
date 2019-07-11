from lbmpy.creationfunctions import create_lb_method
from lbmpy.cumulants import *
from lbmpy.moments import (
    discrete_moment, exponent_to_polynomial_representation, exponents_to_polynomial_representations)
from lbmpy.stencils import get_stencil


def test_cumulants_from_pdfs():
    """
    Tests if the following transformations are equivalent:
      - directly pdfs to cumulant
      - indirect pdfs -> raw moments -> cumulants
    """
    stencil = get_stencil("D2Q9")
    dim = len(stencil[0])
    indices = moments_up_to_component_order(2, dim=dim)

    pdf_symbols = sp.symbols("f_:%d" % (len(stencil),))
    direct_version = cumulants_from_pdfs(stencil, pdf_symbols=pdf_symbols, cumulant_indices=indices)
    polynomial_moment_indices = exponents_to_polynomial_representations(indices)
    direct_version2 = cumulants_from_pdfs(stencil, pdf_symbols=pdf_symbols, cumulant_indices=polynomial_moment_indices)
    for idx, value in direct_version.items():
        poly = exponent_to_polynomial_representation(idx)
        assert direct_version2[poly] == value

    moment_dict = {idx: discrete_moment(pdf_symbols, idx, stencil) for idx in indices}
    indirect_version = {idx: cumulant_as_function_of_raw_moments(idx, moment_dict) for idx in indices}

    for idx in indices:
        assert sp.simplify(direct_version[idx] - indirect_version[idx]) == 0


def test_raw_moment_to_cumulant_transformation():
    """Transforms from raw moments to cumulants and back, then checks for identity"""
    for stencil in [get_stencil("D2Q9"), get_stencil("D3Q27")]:
        dim = len(stencil[0])
        indices = moments_up_to_component_order(2, dim=dim)

        symbol_format = "m_%d_%d_%d" if dim == 3 else "m_%d_%d"
        moment_symbols = {idx: sp.Symbol(symbol_format % idx) for idx in indices}

        forward = {idx: cumulant_as_function_of_raw_moments(idx, moments_dict=moment_symbols)
                   for idx in indices}
        backward = {idx: sp.simplify(raw_moment_as_function_of_cumulants(idx, cumulants_dict=forward))
                    for idx in indices}
        assert backward == moment_symbols


def test_central_moment_to_cumulant_transformation():
    """Transforms from central moments to cumulants and back, then checks for identity"""
    for stencil in [get_stencil("D2Q9"), get_stencil("D3Q27")]:
        dim = len(stencil[0])
        indices = moments_up_to_component_order(2, dim=dim)

        symbol_format = "m_%d_%d_%d" if dim == 3 else "m_%d_%d"
        moment_symbols = {idx: sp.Symbol(symbol_format % idx) for idx in indices}

        forward = {idx: cumulant_as_function_of_central_moments(idx, moments_dict=moment_symbols)
                   for idx in indices}
        backward = {idx: sp.simplify(central_moment_as_function_of_cumulants(idx, cumulants_dict=forward))
                    for idx in indices}
        for idx in indices:
            if sum(idx) == 1:
                continue
            assert backward[idx] == moment_symbols[idx]


def test_collision_rule():
    cumulant_method = create_lb_method(stencil='D2Q9', compressible=True, cumulant=True, relaxation_rate=1.5)
    cumulant_method.set_first_moment_relaxation_rate(1.5)
    assert cumulant_method.force_model is None
    assert cumulant_method.zeroth_order_equilibrium_moment_symbol == sp.symbols("rho")
    assert cumulant_method.first_order_equilibrium_moment_symbols == sp.symbols("u_:2")
    assert all(e.relaxation_rate == 1.5 for e in cumulant_method.relaxation_info_dict.values())
    cumulant_method.get_equilibrium_terms()

    cr1 = cumulant_method.get_collision_rule(moment_subexpressions=True, post_collision_subexpressions=True,
                                             pre_collision_subexpressions=True)
    cr2 = cumulant_method.get_collision_rule(moment_subexpressions=False, post_collision_subexpressions=False,
                                             pre_collision_subexpressions=False)
    cr1_inserted = cr1.new_without_subexpressions()
    cr2_inserted = cr2.new_without_subexpressions()

    t = cr1_inserted.main_assignments[8].rhs - cr2_inserted.main_assignments[8].rhs
    assert t == 0

    html = cumulant_method._repr_html_()
    assert 'Cumulant' in html


def test_cumulants_from_pdf():
    res = cumulants_from_pdfs(get_stencil("D2Q9"))
    assert res[(0, 0)] == sp.log(sum(sp.symbols("f_:9")))
