"""
2D Benchmarks described in the paper
Schäfer, M., Turek, S., Durst, F., Krause, E., & Rannacher, R. (1996). Benchmark computations of laminar flow around 
a cylinder. In Flow simulation with high-performance computers II (pp. 547-566). Vieweg+ Teubner Verlag.

- boundaries are not set correctly yet (halfway-bounce back is not considered)
"""
import warnings

import numpy as np
import pytest

from pystencils.backends.simd_instruction_sets import get_supported_instruction_sets
from lbmpy.boundaries.boundaryconditions import NoSlip
from lbmpy.geometry import get_pipe_velocity_field
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.scenarios import create_channel


def geometry_2d(dx):
    """Geometry setup for the Schaefer Turek benchmark as described in the paper.
    Returns the domain size in cells, a callback function that sets the obstacle and a dict with parameter information
    """
    cylinder_offset_bottom = 0.15
    cylinder_offset_top = 0.16
    cylinder_offset_inflow = 0.15
    cylinder_diameter = 0.1
    channel_length = 2.2

    cylinder_midpoint = np.array([cylinder_offset_inflow + cylinder_diameter / 2,
                                 cylinder_offset_bottom + cylinder_diameter / 2])

    channel_height = cylinder_offset_bottom + cylinder_diameter + cylinder_offset_top

    def to_lattice_units(x):
        result = x / dx
        if abs(round(result) - result) > 1e-10:
            warnings.warn("dx does not divide on of the lengths. Geometry might be slightly inaccurate")
        return round(result)

    domain_size_l = [to_lattice_units(channel_length), to_lattice_units(channel_height)]
    to_lattice_units(cylinder_diameter)
    cylinder_midpoint_l = [i / dx for i in cylinder_midpoint]
    cylinder_radius_l = cylinder_diameter / 2 / dx

    def sphere_geometry_callback(x, y):
        return (x - cylinder_midpoint_l[0]) ** 2 + (y - cylinder_midpoint_l[1]) ** 2 < cylinder_radius_l ** 2

    parameter_info = {
        'cylinder_diameter_l': cylinder_radius_l * 2,
        'cylinder_midpoint_l': cylinder_midpoint_l,
        'dx': dx,
    }
    return domain_size_l, sphere_geometry_callback, parameter_info


def compute_delta_t(max_lattice_velocity, max_velocity, dx):
    """Computes length of a time step given a physical and a lattice velocity"""
    # latticeVelocity * dx / dt = velocity  -> dt = latticeVelocity / velocity * dx
    return max_lattice_velocity / max_velocity * dx


def evaluate_static_quantities(scenario):
    """
    Evaluates drag coefficient, lift coefficient, pressure drop over obstacle and the (approximate) recirculation length
    given a Schaefer Turek scenario object
    :return: a dictionary with the results
    """
    force_on_cylinder = scenario.boundary_handling.force_on_boundary(NoSlip("obstacle"))
    pi = scenario.parameterInfo
    drag_coefficient = force_on_cylinder[0] * 2 / (pi['u_bar_l'] ** 2 * pi['cylinder_diameter_l'])
    lift_coefficient = force_on_cylinder[1] * 2 / (pi['u_bar_l'] ** 2 * pi['cylinder_diameter_l'])

    obstacle_midpoint_height = int(round(pi['cylinder_midpoint_l'][1]))
    density_slice = scenario.density[:, obstacle_midpoint_height]
    last_cell_x_before_obstacle = np.argmax(density_slice.mask) - 1
    obstacle_width = np.argmin(density_slice.mask[last_cell_x_before_obstacle + 1:])
    first_cell_x_after_obstacle = obstacle_width + last_cell_x_before_obstacle + 1

    pressures = [density_slice[x] / 3
                 for x in [last_cell_x_before_obstacle, first_cell_x_after_obstacle]]
    pressure_difference = pressures[0] - pressures[1]
    pressure_difference *= pi['dx'] ** 2 / (pi['dt'] ** 2)

    # Velocity in a line starting directly after the obstacle
    # recirculation is (somewhat inaccurately) determined as the number of cells behind
    # obstacle with x velocity smaller than zero
    vel_slice = scenario.velocity[first_cell_x_after_obstacle:, obstacle_midpoint_height, 0]
    recirculation_length = np.argmax(vel_slice > 0) * pi['dx']
    return {
        'c_D': drag_coefficient,
        'c_L': lift_coefficient,
        'DeltaP': pressure_difference,
        'L_a': recirculation_length,
    }


def schaefer_turek_2d(cells_per_diameter, u_max=0.3, max_lattice_velocity=0.05, **kwargs):
    """Creates a 2D Schaefer Turek Benchmark.

    Args:
        cells_per_diameter: how many lattice cells are used to resolve the obstacle diameter
        u_max: called U_m in the paper: the maximum inflow velocity in physical units, for the first setup
                  it is 0.3, for the second setup 1.5
        max_lattice_velocity: maximum lattice velocity, the lower the more accurate is the simulation
                              should not be larger than 0.1, if chosen too small the relaxation rate gets near 2 and
                              simulation might also get unstable
        kwargs: parameters forwarded to the lattice boltzmann method

    Returns:
        scenario object
    """
    dx = 0.1 / cells_per_diameter
    viscosity = 1e-3

    dt = compute_delta_t(max_lattice_velocity, u_max, dx)
    lattice_viscosity = viscosity / dx / dx * dt
    omega = relaxation_rate_from_lattice_viscosity(lattice_viscosity)
    domain_size, geometry_callback, parameter_info = geometry_2d(dx)
    cylinder_diameter_l = parameter_info['cylinder_diameter_l']
    u_bar_l = 2 / 3 * max_lattice_velocity
    re_lattice = u_bar_l * cylinder_diameter_l / lattice_viscosity
    print("Schaefer-Turek 2D: U_m = %.2f m/s  cells=%s, dx=%f,  dt=%f,  omega=%f, Re=%.1f" %
          (u_max, domain_size, dx, dt, omega, re_lattice))

    initial_velocity = get_pipe_velocity_field(domain_size, max_lattice_velocity)
    scenario = create_channel(domain_size=domain_size, u_max=max_lattice_velocity, relaxation_rate=omega,
                              initial_velocity=initial_velocity, **kwargs)
    scenario.boundary_handling.set_boundary(NoSlip('obstacle'), mask_callback=geometry_callback)
    parameter_info['u_bar_l'] = u_bar_l
    parameter_info['dt'] = dt
    scenario.parameterInfo = parameter_info
    return scenario


def long_run(steady=True, **kwargs):
    if steady:  # scenario 2D-1 in the paper
        sc = schaefer_turek_2d(60, max_lattice_velocity=0.05, **kwargs)
    else:  # Scenario 2D-2 (unsteady)
        sc = schaefer_turek_2d(40, u_max=1.5, max_lattice_velocity=0.01)

    for i in range(100):
        sc.run(10000)
        res = evaluate_static_quantities(sc)
        print(res)
    import lbmpy.plot as plt
    plt.vector_field_magnitude(sc.velocity[:, :])
    plt.show()


@pytest.mark.skipif(not get_supported_instruction_sets(), reason='cannot detect CPU instruction set')
def test_schaefer_turek():
    opt = {'vectorization': {'instruction_set': get_supported_instruction_sets()[-1], 'assume_aligned': True}, 'openmp': 2}
    sc_2d_1 = schaefer_turek_2d(30, max_lattice_velocity=0.08, optimization=opt)
    sc_2d_1.run(30000)
    result = evaluate_static_quantities(sc_2d_1)
    assert 5.5 < result['c_D'] < 5.8
    assert 0.117 < result['DeltaP'] < 0.118


if __name__ == '__main__':
    long_run(entropic=True, method='trt_kbc_n1', compressible=True,
             optimization={'target': Target.GPU, 'gpuIndexingParams': {'blockSize': (16, 8, 2)}})
