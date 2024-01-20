import numpy as np
import pytest

from lbmpy.creationfunctions import create_lb_ast, LBMConfig, LBMOptimisation
from lbmpy.enums import ForceModel, Method, Stencil
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.stencils import LBStencil
from pystencils.sympyextensions import count_operations_in_ast
from sympy.core.cache import clear_cache


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19])
@pytest.mark.parametrize('compressible', [True, False])
@pytest.mark.parametrize('method', [Method.SRT, Method.TRT])
def test_split_number_of_operations(stencil, compressible, method):
    # For the following configurations the number of operations for splitted and un-splitted version are
    # exactly equal. This is not true for D3Q15 and D3Q27 because some sub-expressions are computed in multiple
    # splitted, inner loops.
    lbm_config = LBMConfig(stencil=LBStencil(stencil), method=method, compressible=compressible,
                           force_model=ForceModel.LUO, force=(1e-6, 1e-5, 1e-7))
    lbm_opt_split = LBMOptimisation(split=True)
    lbm_opt = LBMOptimisation(split=False)

    ast_with_splitting = create_lb_ast(lbm_config=lbm_config, lbm_optimisation=lbm_opt_split)
    ast_without_splitting = create_lb_ast(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    op_with_splitting = count_operations_in_ast(ast_with_splitting)
    op_without_splitting = count_operations_in_ast(ast_without_splitting)
    assert op_without_splitting['muls'] == op_with_splitting['muls']
    assert op_without_splitting['adds'] == op_with_splitting['adds']
    assert op_without_splitting['divs'] == op_with_splitting['divs']


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('compressible', [True, False])
@pytest.mark.parametrize('method', [Method.SRT, Method.TRT])
@pytest.mark.parametrize('force', [(0, 0, 0), (1e-6, 1e-7, 2e-6)])
@pytest.mark.longrun
def test_equivalence(stencil, compressible, method, force):
    relaxation_rates = [1.8, 1.7, 1.0, 1.0, 1.0, 1.0]
    stencil = LBStencil(stencil)
    clear_cache()
    domain_size = (10, 20) if stencil.D == 2 else (5, 10, 7)
    lbm_config = LBMConfig(stencil=stencil, method=method, compressible=compressible,
                           relaxation_rates=relaxation_rates,
                           force_model=ForceModel.GUO, force=force)
    lbm_opt_split = LBMOptimisation(split=True)
    lbm_opt = LBMOptimisation(split=False)

    with_split = create_lid_driven_cavity(domain_size=domain_size,
                                          lbm_config=lbm_config, lbm_optimisation=lbm_opt_split)
    without_split = create_lid_driven_cavity(domain_size=domain_size,
                                             lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    with_split.run(100)
    without_split.run(100)
    np.testing.assert_almost_equal(with_split.velocity_slice(), without_split.velocity_slice())


@pytest.mark.parametrize('setup', [(Stencil.D2Q9, True, Method.SRT, 1e-7), (Stencil.D3Q19, False, Method.MRT, 0)])
def test_equivalence_short(setup):
    relaxation_rates = [1.8, 1.7, 1.0, 1.0, 1.0, 1.0]
    stencil = LBStencil(setup[0])
    compressible = setup[1]
    method = setup[2]
    force = (setup[3], 0) if stencil.D == 2 else (setup[3], 0, 0)

    domain_size = (20, 30) if stencil.D == 2 else (10, 13, 7)
    lbm_config = LBMConfig(stencil=stencil, method=method, compressible=compressible,
                           relaxation_rates=relaxation_rates,
                           force_model=ForceModel.GUO, force=force)
    lbm_opt_split = LBMOptimisation(split=True)
    lbm_opt = LBMOptimisation(split=False)

    with_split = create_lid_driven_cavity(domain_size=domain_size,
                                          lbm_config=lbm_config, lbm_optimisation=lbm_opt_split)
    without_split = create_lid_driven_cavity(domain_size=domain_size,
                                             lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    with_split.run(100)
    without_split.run(100)
    np.testing.assert_almost_equal(with_split.velocity_slice(), without_split.velocity_slice())
