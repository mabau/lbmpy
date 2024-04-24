import pytest

import pystencils as ps
from lbmpy import Stencil, LBStencil, LBMConfig, Method, lattice_viscosity_from_relaxation_rate, \
    LatticeBoltzmannStep, pdf_initialization_assignments
from lbmpy.boundaries.boundaryconditions import UBB, WallFunctionBounce, NoSlip, FreeSlip
from lbmpy.boundaries.wall_function_models import SpaldingsLaw, LogLaw, MoninObukhovSimilarityTheory, MuskerLaw
from pystencils.slicing import slice_from_direction


def check_velocity(noslip_velocity, freeslip_velocity, wfb_results):
    for wfb, result in wfb_results.items():
        assert result[0] > noslip_velocity[0], f"{wfb} enforced velocity below no-slip velocity"
        assert result[0] < freeslip_velocity[0], f"{wfb} enforced velocity above free-slip velocity"


@pytest.mark.parametrize('stencil', [Stencil.D3Q19, Stencil.D3Q27])
def test_wfb(stencil):
    stencil = LBStencil(stencil)
    wall_velocity = (0.01, 0, 0)
    periodicity = (True, False, True)
    domain_size = (30, 15, 30)
    dim = len(domain_size)

    omega = 1.8
    nu = lattice_viscosity_from_relaxation_rate(omega)

    lbm_config = LBMConfig(stencil=stencil,
                           method=Method.SRT,
                           relaxation_rate=omega,
                           compressible=True)

    moving_wall = UBB(wall_velocity)

    normal = (0, 1, 0)

    # NO-SLIP

    lb_step_noslip = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                          lbm_config=lbm_config, compute_velocity_in_every_step=True)

    noslip = NoSlip()

    lb_step_noslip.boundary_handling.set_boundary(noslip, slice_from_direction('S', dim))
    lb_step_noslip.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    # FREE-SLIP

    lb_step_freeslip = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                            lbm_config=lbm_config, compute_velocity_in_every_step=True)

    freeslip = FreeSlip(stencil=stencil, normal_direction=(0, -1, 0))

    lb_step_freeslip.boundary_handling.set_boundary(noslip, slice_from_direction('S', dim))
    lb_step_freeslip.boundary_handling.set_boundary(freeslip, slice_from_direction('N', dim))

    # WFB - CASE I

    lb_step_wfb_i = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                         lbm_config=lbm_config, compute_velocity_in_every_step=True)

    init = pdf_initialization_assignments(lb_step_wfb_i.method, 1.0, (1e-6, 0, 0),
                                          lb_step_wfb_i.data_handling.fields[lb_step_wfb_i._pdf_arr_name].center_vector)

    config = ps.CreateKernelConfig(target=lb_step_wfb_i.data_handling.default_target, cpu_openmp=False)
    ast_init = ps.create_kernel(init, config=config)
    kernel_init = ast_init.compile()

    lb_step_wfb_i.data_handling.run_kernel(kernel_init)

    wall_i = WallFunctionBounce(lb_method=lb_step_wfb_i.method, name="wall_i", normal_direction=normal,
                                pdfs=lb_step_wfb_i.data_handling.fields[lb_step_wfb_i._pdf_arr_name],
                                wall_function_model=SpaldingsLaw(viscosity=nu),
                                weight_method=WallFunctionBounce.WeightMethod.GEOMETRIC_WEIGHT)

    lb_step_wfb_i.boundary_handling.set_boundary(wall_i, slice_from_direction('S', dim))
    lb_step_wfb_i.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    # WFB - CASE II

    lb_step_wfb_ii = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                          lbm_config=lbm_config, compute_velocity_in_every_step=True)

    meanVelField_ii = lb_step_wfb_ii.data_handling.add_array('mean_velocity_field_ii', values_per_cell=stencil.D)
    lb_step_wfb_ii.data_handling.fill('mean_velocity_field_ii', 1e-4, value_idx=0, ghost_layers=True)

    wall_ii = WallFunctionBounce(lb_method=lb_step_wfb_ii.method, name="wall_ii", normal_direction=normal,
                                 pdfs=lb_step_wfb_ii.data_handling.fields[lb_step_wfb_ii._pdf_arr_name],
                                 wall_function_model=LogLaw(viscosity=nu),
                                 weight_method=WallFunctionBounce.WeightMethod.GEOMETRIC_WEIGHT,
                                 mean_velocity=meanVelField_ii)

    lb_step_wfb_ii.boundary_handling.set_boundary(wall_ii, slice_from_direction('S', dim))
    lb_step_wfb_ii.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    # WFB - CASE III

    lb_step_wfb_iii = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                           lbm_config=lbm_config, compute_velocity_in_every_step=True)

    meanVelField_iii = lb_step_wfb_iii.data_handling.add_array('mean_velocity_field_iii', values_per_cell=stencil.D)
    lb_step_wfb_iii.data_handling.fill('mean_velocity_field_iii', 1e-4, value_idx=0, ghost_layers=True)

    wall_iii = WallFunctionBounce(lb_method=lb_step_wfb_iii.method, name="wall_iii", normal_direction=normal,
                                  pdfs=lb_step_wfb_iii.data_handling.fields[lb_step_wfb_iii._pdf_arr_name],
                                  wall_function_model=MoninObukhovSimilarityTheory(z0=1e-2),
                                  weight_method=WallFunctionBounce.WeightMethod.LATTICE_WEIGHT,
                                  mean_velocity=meanVelField_iii, sampling_shift=1)

    lb_step_wfb_iii.boundary_handling.set_boundary(wall_iii, slice_from_direction('S', dim))
    lb_step_wfb_iii.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    # WFB - CASE IV

    lb_step_wfb_iv = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                          lbm_config=lbm_config, compute_velocity_in_every_step=True)

    meanVelField_iv = lb_step_wfb_iv.data_handling.add_array('mean_velocity_field_iv', values_per_cell=stencil.D)
    lb_step_wfb_iv.data_handling.fill('mean_velocity_field_iv', 1e-4, value_idx=0, ghost_layers=True)

    wall_iv = WallFunctionBounce(lb_method=lb_step_wfb_iv.method, name="wall_iv", normal_direction=normal,
                                 pdfs=lb_step_wfb_iv.data_handling.fields[lb_step_wfb_iv._pdf_arr_name],
                                 wall_function_model=MuskerLaw(viscosity=nu),
                                 weight_method=WallFunctionBounce.WeightMethod.LATTICE_WEIGHT,
                                 mean_velocity=meanVelField_iv, maronga_sampling_shift=4)

    lb_step_wfb_iv.boundary_handling.set_boundary(wall_iv, slice_from_direction('S', dim))
    lb_step_wfb_iv.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    timesteps = 2000
    lb_step_noslip.run(timesteps)
    lb_step_freeslip.run(timesteps)
    lb_step_wfb_i.run(timesteps)
    lb_step_wfb_ii.run(timesteps)
    lb_step_wfb_iii.run(timesteps)
    lb_step_wfb_iv.run(timesteps)

    noslip_velocity = lb_step_noslip.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0]
    freeslip_velocity = lb_step_freeslip.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0]

    results = {
        'wfb_i': lb_step_wfb_i.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0],
        'wfb_ii': lb_step_wfb_ii.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0],
        'wfb_iii': lb_step_wfb_iii.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0],
        'wfb_iv': lb_step_wfb_iv.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0]
    }

    check_velocity(noslip_velocity=noslip_velocity, freeslip_velocity=freeslip_velocity, wfb_results=results)
