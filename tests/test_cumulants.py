from lbmpy.cumulants import *
from lbmpy.moments import (
    discrete_moment, exponent_to_polynomial_representation, exponents_to_polynomial_representations)
from lbmpy.enums import Stencil
from lbmpy.stencils import LBStencil

def test_cumulants_from_pdfs():
    """
    Tests if the following transformations are equivalent:
      - directly pdfs to cumulant
      - indirect pdfs -> raw moments -> cumulants
    """
    stencil = LBStencil(Stencil.D2Q9)
    indices = moments_up_to_component_order(2, dim=stencil.D)

    pdf_symbols = sp.symbols(f"f_:{stencil.Q}")
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
    for stencil in [LBStencil(Stencil.D2Q9), LBStencil(Stencil.D3Q27)]:
        indices = moments_up_to_component_order(2, dim=stencil.D)

        symbol_format = "m_%d_%d_%d" if stencil.D == 3 else "m_%d_%d"
        moment_symbols = {idx: sp.Symbol(symbol_format % idx) for idx in indices}

        forward = {idx: cumulant_as_function_of_raw_moments(idx, moments_dict=moment_symbols)
                   for idx in indices}
        backward = {idx: sp.simplify(raw_moment_as_function_of_cumulants(idx, cumulants_dict=forward))
                    for idx in indices}
        assert backward == moment_symbols


def test_central_moment_to_cumulant_transformation():
    """Transforms from central moments to cumulants and back, then checks for identity"""
    for stencil in [LBStencil(Stencil.D2Q9), LBStencil(Stencil.D3Q27)]:
        indices = moments_up_to_component_order(2, dim=stencil.D)

        symbol_format = "m_%d_%d_%d" if stencil.D == 3 else "m_%d_%d"
        moment_symbols = {idx: sp.Symbol(symbol_format % idx) for idx in indices}

        forward = {idx: cumulant_as_function_of_central_moments(idx, moments_dict=moment_symbols)
                   for idx in indices}
        backward = {idx: sp.simplify(central_moment_as_function_of_cumulants(idx, cumulants_dict=forward))
                    for idx in indices}
        for idx in indices:
            if sum(idx) == 1:
                continue
            assert backward[idx] == moment_symbols[idx]


def test_cumulants_from_pdf():
    res = cumulants_from_pdfs(LBStencil(Stencil.D2Q9))
    assert res[(0, 0)] == sp.log(sum(sp.symbols(f"f_:{9}")))
