import pytest
import numpy as np
from dataclasses import replace

from lbmpy.advanced_streaming import BetweenTimestepsIndexing, Timestep, get_timesteps
from lbmpy.creationfunctions import create_lb_function, create_lb_collision_rule,\
    create_lb_method, LBMConfig, LBMOptimisation
from lbmpy.boundaries import LatticeBoltzmannBoundaryHandling, NoSlip, UBB
from lbmpy.enums import Method, Stencil

from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments
from lbmpy.stencils import LBStencil

from pystencils import create_kernel, create_data_handling, Assignment, Target, CreateKernelConfig
from pystencils.slicing import slice_from_direction, get_slice_before_ghost_layer


def flow_around_sphere(stencil, galilean_correction, L_LU, total_steps):
    if galilean_correction and stencil.Q != 27:
        pytest.skip()

    target = Target.GPU
    streaming_pattern = 'aa'
    timesteps = get_timesteps(streaming_pattern)

    u_max = 0.05
    Re = 500000

    kinematic_viscosity = (L_LU * u_max) / Re
    initial_velocity = (u_max,) + (0,) * (stencil.D - 1)

    omega_v = relaxation_rate_from_lattice_viscosity(kinematic_viscosity)

    channel_size = (10 * L_LU,) + (5 * L_LU,) * (stencil.D - 1)
    sphere_position = (channel_size[0] // 3,) + (channel_size[1] // 2,) * (stencil.D - 1)
    sphere_radius = L_LU // 2

    lbm_config = LBMConfig(stencil=stencil, method=Method.CUMULANT, relaxation_rate=omega_v,
                           compressible=True,
                           galilean_correction=galilean_correction)

    lbm_opt = LBMOptimisation(pre_simplification=True)
    config = CreateKernelConfig(target=target)

    lb_method = create_lb_method(lbm_config=lbm_config)

    def get_extrapolation_kernel(timestep):
        boundary_assignments = []
        indexing = BetweenTimestepsIndexing(
            pdf_field, stencil, streaming_pattern=streaming_pattern, prev_timestep=timestep)
        f_out, _ = indexing.proxy_fields
        for i, d in enumerate(stencil):
            if d[0] == -1:
                asm = Assignment(f_out.neighbor(0, 1)(i), f_out.center(i))
                boundary_assignments.append(asm)
        boundary_assignments = indexing.substitute_proxies(
            boundary_assignments)
        iter_slice = get_slice_before_ghost_layer((1,) + (0,) * (stencil.D - 1))
        extrapolation_ast = create_kernel(
            boundary_assignments, config=CreateKernelConfig(iteration_slice=iter_slice, ghost_layers=1, target=target))
        return extrapolation_ast.compile()

    dh = create_data_handling(channel_size, periodicity=False, default_layout='fzyx', default_target=target)

    u_field = dh.add_array('u', stencil.D)
    rho_field = dh.add_array('rho', 1)
    pdf_field = dh.add_array('pdfs', stencil.Q)

    dh.fill(u_field.name, 0.0, ghost_layers=True)
    dh.fill(rho_field.name, 0.0, ghost_layers=True)

    dh.to_gpu(u_field.name)
    dh.to_gpu(rho_field.name)

    lbm_opt = replace(lbm_opt, symbolic_field=pdf_field)

    bh = LatticeBoltzmannBoundaryHandling(lb_method, dh, pdf_field.name,
                                          streaming_pattern=streaming_pattern, target=target)
    wall = NoSlip()
    inflow = UBB(initial_velocity)

    bh.set_boundary(inflow, slice_from_direction('W', stencil.D))

    directions = ('N', 'S', 'T', 'B') if stencil.D == 3 else ('N', 'S')
    for direction in directions:
        bh.set_boundary(wall, slice_from_direction(direction, stencil.D))

    outflow_kernels = [get_extrapolation_kernel(Timestep.EVEN), get_extrapolation_kernel(Timestep.ODD)]

    def sphere_boundary_callback(x, y, z=None):
        x = x - sphere_position[0]
        y = y - sphere_position[1]
        z = z - sphere_position[2] if z is not None else 0
        return np.sqrt(x ** 2 + y ** 2 + z ** 2) <= sphere_radius

    bh.set_boundary(wall, mask_callback=sphere_boundary_callback)

    init_eqs = pdf_initialization_assignments(lb_method, 1.0, initial_velocity, pdf_field,
                                              streaming_pattern=streaming_pattern,
                                              previous_timestep=timesteps[0])
    init_kernel = create_kernel(init_eqs, config=config).compile()

    output = {
        'density': rho_field,
        'velocity': u_field
    }
    lbm_config = replace(lbm_config, output=output)

    lb_collision_rule = create_lb_collision_rule(lb_method=lb_method, lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    lb_kernels = []
    for t in timesteps:
        lbm_config = replace(lbm_config, timestep=t)
        lbm_config = replace(lbm_config, streaming_pattern=streaming_pattern)
        lb_kernels.append(create_lb_function(collision_rule=lb_collision_rule,
                                             lbm_config=lbm_config, lbm_optimisation=lbm_opt, config=config))

    timestep = timesteps[0]

    dh.run_kernel(init_kernel)

    stability_check_frequency = 1000

    for i in range(total_steps):
        bh(prev_timestep=timestep)
        dh.run_kernel(outflow_kernels[timestep.idx])
        timestep = timestep.next()
        dh.run_kernel(lb_kernels[timestep.idx])

        if i % stability_check_frequency == 0:
            dh.to_cpu(u_field.name)
            assert np.isfinite(dh.cpu_arrays[u_field.name]).all()


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('galilean_correction', [False, True])
def test_flow_around_sphere_short(stencil, galilean_correction):
    pytest.importorskip('pycuda')
    flow_around_sphere(LBStencil(stencil), galilean_correction, 5, 200)


@pytest.mark.parametrize('stencil', [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize('galilean_correction', [False, True])
@pytest.mark.longrun
def test_flow_around_sphere_long(stencil, galilean_correction):
    pytest.importorskip('pycuda')
    flow_around_sphere(LBStencil(stencil), galilean_correction, 20, 3000)
