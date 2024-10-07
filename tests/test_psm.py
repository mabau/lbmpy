import pytest
import numpy as np

from pystencils import fields, CreateKernelConfig, Target, create_kernel, get_code_str, create_data_handling
from lbmpy import LBMConfig, Stencil, Method, LBStencil, create_lb_method, create_lb_collision_rule, LBMOptimisation, \
    create_lb_update_rule
from lbmpy.partially_saturated_cells import PSMConfig


@pytest.mark.parametrize("stencil", [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize("method_enum", [Method.SRT, Method.MRT, Method.CUMULANT])
def test_psm_integration(stencil, method_enum):
    stencil = LBStencil(stencil)

    fraction_field = fields("fraction_field: double[10, 10, 5]", layout=(2, 1, 0))
    object_vel = fields("object_vel(3): double[10, 10, 5]", layout=(3, 2, 1, 0))
    psm_config = PSMConfig(fraction_field=fraction_field, object_velocity_field=object_vel)

    lbm_config = LBMConfig(stencil=stencil, method=method_enum, relaxation_rate=1.5, compressible=True,
                           psm_config=psm_config)
    config = CreateKernelConfig(target=Target.CPU)#

    collision_rule = create_lb_collision_rule(lbm_config=lbm_config)

    ast = create_kernel(collision_rule, config=config)
    code_str = get_code_str(ast)


def get_data_handling_for_psm(use_psm):
    domain_size = (10, 10)
    stencil = LBStencil(Stencil.D2Q9)
    dh = create_data_handling(domain_size=domain_size, periodicity=(True, False))

    f = dh.add_array('f', values_per_cell=len(stencil))
    dh.fill(f.name, 0.0, ghost_layers=True)
    f_tmp = dh.add_array('f_tmp', values_per_cell=len(stencil))
    dh.fill(f_tmp.name, 0.0, ghost_layers=True)

    psm_config = None
    if use_psm:
        fraction_field = dh.add_array('fraction_field', values_per_cell=1)
        dh.fill(fraction_field.name, 0.0, ghost_layers=True)
        object_vel = dh.add_array('object_vel', values_per_cell=dh.dim)
        dh.fill(object_vel.name, 0.0, ghost_layers=True)
        psm_config = PSMConfig(fraction_field=fraction_field, object_velocity_field=object_vel)

    lbm_config = LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.5, psm_config=psm_config)

    lbm_optimisation = LBMOptimisation(symbolic_field=f, symbolic_temporary_field=f_tmp)
    update_rule = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_optimisation)
    kernel_lb_step_with_psm = create_kernel(update_rule).compile()
    return (dh, kernel_lb_step_with_psm)


def test_lbm_vs_psm():

    psm_dh, psm_kernel = get_data_handling_for_psm(True)
    lbm_dh, lbm_kernel = get_data_handling_for_psm(False)

    for i in range(20):
        psm_dh.run_kernel(psm_kernel)
        psm_dh.swap('f', 'f_tmp')

        lbm_dh.run_kernel(lbm_kernel)
        lbm_dh.swap('f', 'f_tmp')

    max_vel_error = np.max(np.abs(psm_dh.gather_array('f') - lbm_dh.gather_array('f')))
    np.testing.assert_allclose(max_vel_error, 0, atol=1e-14)
