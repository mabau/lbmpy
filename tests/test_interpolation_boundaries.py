import pytest
import numpy as np
from pystencils.slicing import slice_from_direction

from lbmpy import LBMConfig, LBStencil, Stencil, Method
from lbmpy.boundaries.boundaryconditions import NoSlip, NoSlipLinearBouzidi, QuadraticBounceBack, UBB
from lbmpy.lbstep import LatticeBoltzmannStep


def check_velocity(noslip_velocity, interpolated_velocity, wall_distance):
    # First we check if the flow is fully developed
    np.testing.assert_almost_equal(np.gradient(np.gradient(noslip_velocity)), 0, decimal=8)
    np.testing.assert_almost_equal(np.gradient(np.gradient(interpolated_velocity)), 0, decimal=8)

    # If the wall is closer to the first fluid cell we expect a lower velocity at the first fluid cell
    if wall_distance < 0.5:
        assert noslip_velocity[0] > interpolated_velocity[0]
    # If the wall is further away from the first fluid cell we expect a higher velocity at the first fluid cell
    if wall_distance > 0.5:
        assert noslip_velocity[0] < interpolated_velocity[0]
    # If the wall cuts the cell halfway the interpolation BC should approximately fall back to a noslip boundary
    if wall_distance == 0.5:
        np.testing.assert_almost_equal(noslip_velocity[0], interpolated_velocity[0], decimal=2)


def couette_flow(stencil, method_enum, zero_centered, wall_distance, compressible):
    dim = stencil.D
    if dim == 2:
        domain_size = (50, 25)
        wall_velocity = (0.01, 0)
        periodicity = (True, False)
    else:
        domain_size = (50, 25, 25)
        wall_velocity = (0.01, 0, 0)
        periodicity = (True, False, True)

    timesteps = 10000
    omega = 1.1

    lbm_config = LBMConfig(stencil=stencil,
                           method=method_enum,
                           relaxation_rate=omega,
                           zero_centered=zero_centered,
                           compressible=compressible,
                           weighted=True)

    lb_step_noslip = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                          lbm_config=lbm_config, compute_velocity_in_every_step=True)
    lb_step_bouzidi = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                           lbm_config=lbm_config, compute_velocity_in_every_step=True)

    lb_step_quadratic_bb = LatticeBoltzmannStep(domain_size=domain_size, periodicity=periodicity,
                                                lbm_config=lbm_config, compute_velocity_in_every_step=True)

    def init_wall_distance(boundary_data, **_):
        for cell in boundary_data.index_array:
            cell['q'] = wall_distance

    moving_wall = UBB(wall_velocity)
    noslip = NoSlip("wall")
    bouzidi = NoSlipLinearBouzidi("wall", init_wall_distance=init_wall_distance)
    quadratic_bb = QuadraticBounceBack(omega, "wall", init_wall_distance=init_wall_distance)

    lb_step_noslip.boundary_handling.set_boundary(noslip, slice_from_direction('S', dim))
    lb_step_noslip.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    lb_step_bouzidi.boundary_handling.set_boundary(bouzidi, slice_from_direction('S', dim))
    lb_step_bouzidi.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    lb_step_quadratic_bb.boundary_handling.set_boundary(quadratic_bb, slice_from_direction('S', dim))
    lb_step_quadratic_bb.boundary_handling.set_boundary(moving_wall, slice_from_direction('N', dim))

    lb_step_noslip.run(timesteps)
    lb_step_bouzidi.run(timesteps)
    lb_step_quadratic_bb.run(timesteps)

    if dim == 2:
        noslip_velocity = lb_step_noslip.velocity[domain_size[0] // 2, :, 0]
        bouzidi_velocity = lb_step_bouzidi.velocity[domain_size[0] // 2, :, 0]
        quadratic_bb_velocity = lb_step_quadratic_bb.velocity[domain_size[0] // 2, :, 0]
    else:
        noslip_velocity = lb_step_noslip.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0]
        bouzidi_velocity = lb_step_bouzidi.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0]
        quadratic_bb_velocity = lb_step_quadratic_bb.velocity[domain_size[0] // 2, :, domain_size[2] // 2, 0]

    check_velocity(noslip_velocity, bouzidi_velocity, wall_distance)
    check_velocity(noslip_velocity, quadratic_bb_velocity, wall_distance)


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("wall_distance", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("compressible", [True, False])
def test_couette_flow_short(zero_centered, wall_distance, compressible):
    stencil = LBStencil(Stencil.D2Q9)
    couette_flow(stencil, Method.SRT, zero_centered, wall_distance, compressible)


@pytest.mark.parametrize("method_enum", [Method.MRT, Method.CENTRAL_MOMENT, Method.CUMULANT])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("wall_distance", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("compressible", [True, False])
@pytest.mark.longrun
def test_couette_flow_long(method_enum, zero_centered, wall_distance, compressible):
    if method_enum is Method.CUMULANT and compressible is False:
        pytest.skip("incompressible cumulant is not supported")

    stencil = LBStencil(Stencil.D2Q9)
    couette_flow(stencil, method_enum, zero_centered, wall_distance, compressible)


@pytest.mark.parametrize("method_enum", [Method.SRT, Method.MRT, Method.CENTRAL_MOMENT, Method.CUMULANT])
@pytest.mark.parametrize("wall_distance", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("stencil", [Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.longrun
def test_couette_flow_d3d(method_enum, wall_distance, stencil):
    stencil = LBStencil(stencil)
    couette_flow(stencil, method_enum, True, wall_distance, True)


