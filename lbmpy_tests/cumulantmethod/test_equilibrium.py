from lbmpy.creationfunctions import LBMConfig
from lbmpy.enums import Method, Stencil, CollisionSpace
from lbmpy.maxwellian_equilibrium import generate_equilibrium_by_matching_moments
from lbmpy.moments import extract_monomials
from lbmpy.stencils import LBStencil
import pytest
import sympy as sp

from pystencils.simp import AssignmentCollection
from pystencils import Assignment
from lbmpy.creationfunctions import create_lb_method
from lbmpy.methods import CollisionSpaceInfo
from lbmpy.moment_transforms import (
    FastCentralMomentTransform,
    PdfsToCentralMomentsByMatrix,
    PdfsToCentralMomentsByShiftMatrix,
    BinomialChimeraTransform)

sympy_numeric_version = [int(x, 10) for x in sp.__version__.split('.')]
if len(sympy_numeric_version) < 3:
    sympy_numeric_version.append(0)
sympy_numeric_version.reverse()
sympy_version = sum(x * (100 ** i) for i, x in enumerate(sympy_numeric_version))


reference_equilibria = dict()    


@pytest.mark.skipif(sympy_version < 10200,
                    reason="Old Sympy Versions take too long to form the inverse moment matrix")
@pytest.mark.parametrize('stencil_name', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('cm_transform', [PdfsToCentralMomentsByMatrix,
                                          FastCentralMomentTransform,
                                          PdfsToCentralMomentsByShiftMatrix,
                                          BinomialChimeraTransform])
def test_equilibrium_pdfs(stencil_name, cm_transform):
    stencil = LBStencil(stencil_name)
    cspace = CollisionSpaceInfo(CollisionSpace.CUMULANTS, central_moment_transform_class=cm_transform)
    lbm_config = LBMConfig(stencil=stencil, method=Method.CUMULANT, compressible=True, zero_centered=False,
                           collision_space_info=cspace)

    c_lb_method = create_lb_method(lbm_config=lbm_config)
    rho = c_lb_method.zeroth_order_equilibrium_moment_symbol
    u = c_lb_method.first_order_equilibrium_moment_symbols
    pdfs = c_lb_method.post_collision_pdf_symbols
    cqe_assignments = [Assignment(sym, sym) for sym in u] + [Assignment(rho, rho)]
    cqe = AssignmentCollection(main_assignments=cqe_assignments)
    method_equilibrium_eqs = c_lb_method.get_equilibrium(cqe, False, False).main_assignments_dict

    #   Reference Equations
    ref_equilibrium = reference_equilibria.get(stencil_name, None)
    if ref_equilibrium is None:
        raw_moments = list(extract_monomials(c_lb_method.cumulants, dim=stencil.D))
        ref_equilibrium = generate_equilibrium_by_matching_moments(
            stencil, tuple(raw_moments), rho=rho, u=u, c_s_sq=sp.Rational(1, 3), order=2*stencil.D)
        reference_equilibria[stencil_name] = ref_equilibrium

    for i in range(stencil.Q):
        method_eq = method_equilibrium_eqs[pdfs[i]]
        ref_eq = ref_equilibrium[i]
        assert method_eq.expand() - ref_eq.expand() == sp.sympify(0), \
            f"Equilibrium equation for pdf index {i} did not match." 
