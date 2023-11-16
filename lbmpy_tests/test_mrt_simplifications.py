import pytest

import sympy as sp

from pystencils.sympyextensions import is_constant

from lbmpy import Stencil, LBStencil, Method, create_lb_collision_rule, LBMConfig, LBMOptimisation

# TODO:
# Fully simplified kernels should NOT contain
#  - Any aliases
#  - Any in-line constants (all constants should be in subexpressions!)

@pytest.mark.parametrize('method', [Method.MRT, Method.CENTRAL_MOMENT, Method.CUMULANT])
def test_mrt_simplifications(method: Method):
    stencil = Stencil.D3Q19
    lbm_config = LBMConfig(stencil=stencil, method=method, compressible=True)
    lbm_opt = LBMOptimisation(simplification='auto')

    cr = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    
    for subexp in cr.subexpressions:
        rhs = subexp.rhs
        #   Check for aliases
        assert not isinstance(rhs, sp.Symbol)

        #   Check for logarithms
        assert not rhs.atoms(sp.log)

        #   Check for nonextracted constant summands or factors
        exprs = rhs.atoms(sp.Add, sp.Mul)
        for expr in exprs:
            for arg in expr.args:
                if isinstance(arg, sp.Number):
                    assert arg in {sp.Number(1), sp.Number(-1)}
                    
        #   Check for divisions
        if not (isinstance(rhs, sp.Pow) and rhs.args[1] < 0):
            powers = rhs.atoms(sp.Pow)
            for p in powers:
                assert p.args[1] > 0
