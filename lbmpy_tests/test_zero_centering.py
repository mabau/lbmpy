import pytest

import numpy as np

from pystencils import CreateKernelConfig, Target

from lbmpy import Method, LBMConfig
from lbmpy.stencils import Stencil, LBStencil
from lbmpy.maxwellian_equilibrium import get_weights

from lbmpy.scenarios import create_fully_periodic_flow

from numpy.testing import assert_allclose


@pytest.mark.parametrize('method', [Method.SRT, Method.MRT, Method.CENTRAL_MOMENT, Method.CUMULANT])
@pytest.mark.parametrize('delta_equilibrium', [False, True])
def test_periodic_shear_layers(method, delta_equilibrium):
    if method == Method.SRT and not delta_equilibrium:
        pytest.skip()
    if method == Method.CUMULANT and delta_equilibrium:
        pytest.skip()

    stencil = LBStencil(Stencil.D3Q27)
    omega_shear = 1.3

    #   Velocity Field
    L = (128, 32, 32)
    velocity_magnitude = 0.02
    velocity = np.zeros(L + (3,))
    velocity[:,:,:,0] = velocity_magnitude
    velocity[:, L[1]//3 : L[1]//3*2, L[2]//3 : L[2]//3*2, 0] = - velocity_magnitude
    velocity[:, :, :, 1] = 0.1 * velocity_magnitude * np.random.rand(*L)

    kernel_config = CreateKernelConfig(target=Target.CPU)
    
    config_full = LBMConfig(stencil=stencil, method=method, relaxation_rate=omega_shear, 
                            compressible=True, zero_centered=False)
    scenario_full = create_fully_periodic_flow(velocity, lbm_config=config_full, config=kernel_config)

    config_zero_centered = LBMConfig(stencil=stencil, method=method, relaxation_rate=omega_shear, 
                                     compressible=True, zero_centered=True, delta_equilibrium=delta_equilibrium)
    scenario_zero_centered = create_fully_periodic_flow(velocity, lbm_config=config_zero_centered, 
                                                        config=kernel_config)

    scenario_full.run(20)
    scenario_zero_centered.run(20)

    pdfs_full = scenario_full.data_handling.cpu_arrays[scenario_full.pdf_array_name]
    pdfs_zero_centered = scenario_zero_centered.data_handling.cpu_arrays[scenario_zero_centered.pdf_array_name]
    difference = pdfs_full - pdfs_zero_centered

    weights = np.array(get_weights(stencil))
    reference = np.zeros(L + (stencil.Q, ))
    reference[:,:,:] = weights

    if delta_equilibrium and method == Method.CENTRAL_MOMENT:
        #   Much less agreement is expected here, as the delta-equilibrium's velocity dependence
        #   lets the CM method's numerical quality degrade.
        assert_allclose(difference[1:-1, 1:-1, 1:-1], reference, rtol=1e-5)
    else:
        assert_allclose(difference[1:-1, 1:-1, 1:-1], reference)
