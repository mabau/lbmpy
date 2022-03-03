import numpy as np
from dataclasses import replace

from pystencils.datahandling import create_data_handling
from pystencils import create_kernel, Target, CreateKernelConfig

from lbmpy.creationfunctions import create_lb_collision_rule, create_lb_function, LBMConfig, LBMOptimisation
from lbmpy.enums import Method, Stencil
from lbmpy.macroscopic_value_kernels import macroscopic_values_getter, macroscopic_values_setter
from lbmpy.stencils import LBStencil

from lbmpy.advanced_streaming import LBMPeriodicityHandling
from lbmpy.advanced_streaming.utility import is_inplace, streaming_patterns, get_timesteps

import pytest
from numpy.testing import assert_allclose

all_results = dict()

targets = [Target.CPU]

try:
    import pycuda.autoinit
    targets += [Target.GPU]
except Exception:
    pass


@pytest.mark.parametrize('target', targets)
@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('streaming_pattern', streaming_patterns)
@pytest.mark.longrun
def test_fully_periodic_flow(target, stencil, streaming_pattern):
    gpu = False
    if target == Target.GPU:
        gpu = True

    #   Stencil
    stencil = LBStencil(stencil)

    #   Streaming
    inplace = is_inplace(streaming_pattern)
    timesteps = get_timesteps(streaming_pattern)
    zeroth_timestep = timesteps[0]

    #   Data Handling and PDF fields
    domain_size = (30,) * stencil.D
    periodicity = (True,) * stencil.D

    dh = create_data_handling(domain_size=domain_size, periodicity=periodicity, default_target=target)

    pdfs = dh.add_array('pdfs', stencil.Q)
    if not inplace:
        pdfs_tmp = dh.add_array_like('pdfs_tmp', pdfs.name)

    #   LBM Streaming and Collision
    lbm_config = LBMConfig(stencil=stencil, method=Method.SRT,
                           relaxation_rate=1.0, streaming_pattern=streaming_pattern)

    lbm_opt = LBMOptimisation(symbolic_field=pdfs)
    config = CreateKernelConfig(target=target)

    if not inplace:
        lbm_opt = replace(lbm_opt, symbolic_temporary_field=pdfs_tmp)

    lb_collision = create_lb_collision_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt, config=config)
    lb_method = lb_collision.method

    lb_kernels = []
    for t in timesteps:
        lbm_config = replace(lbm_config, timestep=t)
        lb_kernels.append(create_lb_function(collision_rule=lb_collision,
                                             lbm_config=lbm_config, lbm_optimisation=lbm_opt))

    #   Macroscopic Values
    density = 1.0
    density_field = dh.add_array('rho', 1)
    u_x = 0.01
    velocity = (u_x,) * stencil.D
    velocity_field = dh.add_array('u', stencil.D)

    u_ref = np.full(domain_size + (stencil.D,), u_x)

    setter = macroscopic_values_setter(
        lb_method, density, velocity, pdfs,
        streaming_pattern=streaming_pattern, previous_timestep=zeroth_timestep)
    setter_kernel = create_kernel(setter,
                                  config=CreateKernelConfig(target=target, ghost_layers=1)).compile()

    getter_kernels = []
    for t in timesteps:
        getter = macroscopic_values_getter(
            lb_method, density_field, velocity_field, pdfs,
            streaming_pattern=streaming_pattern, previous_timestep=t)
        getter_kernels.append(create_kernel(getter,
                                            config=CreateKernelConfig(target=target, ghost_layers=1)).compile())

    #   Periodicity
    periodicity_handler = LBMPeriodicityHandling(stencil, dh, pdfs.name, streaming_pattern=streaming_pattern)

    # Initialization and Timestep
    current_timestep = zeroth_timestep

    def init():
        global current_timestep
        current_timestep = zeroth_timestep
        dh.run_kernel(setter_kernel)

    def one_step():
        global current_timestep

        # Periodicty
        periodicity_handler(current_timestep)

        # Here, the next time step begins
        current_timestep = current_timestep.next()

        # LBM Step
        dh.run_kernel(lb_kernels[current_timestep.idx])

        # Field Swaps
        if not inplace:
            dh.swap(pdfs.name, pdfs_tmp.name)

        # Macroscopic Values
        dh.run_kernel(getter_kernels[current_timestep.idx])

    #   Run the simulation
    init()

    for _ in range(100):
        one_step()

    #   Evaluation
    if gpu:
        dh.to_cpu(velocity_field.name)
    u = dh.gather_array(velocity_field.name)

    #   Equal to the steady-state velocity field up to numerical errors
    assert_allclose(u, u_ref)

    #   Flow must be equal up to numerical error for all streaming patterns
    global all_results
    for key, prev_u in all_results.items():
        if key[0] == stencil:
            prev_pattern = key[1]
            assert_allclose(
                u, prev_u, err_msg=f'Velocity field for {streaming_pattern} differed from {prev_pattern}!')
    all_results[(stencil, streaming_pattern)] = u
