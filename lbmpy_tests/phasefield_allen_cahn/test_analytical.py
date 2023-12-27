import numpy as np

from lbmpy.phasefield_allen_cahn.parameter_calculation import calculate_dimensionless_rising_bubble, \
    calculate_parameters_rti, calculate_parameters_taylor_bubble

from lbmpy.phasefield_allen_cahn.analytical import analytic_rising_speed


def test_analytical():
    parameters = calculate_dimensionless_rising_bubble(reference_time=18000,
                                                       density_heavy=1.0,
                                                       bubble_radius=16,
                                                       bond_number=30,
                                                       reynolds_number=420,
                                                       density_ratio=1000,
                                                       viscosity_ratio=100)

    assert np.isclose(parameters.density_light, 0.001)
    assert np.isclose(parameters.gravitational_acceleration, -9.876543209876543e-08)

    parameters = calculate_parameters_rti(reference_length=128,
                                          reference_time=18000,
                                          density_heavy=1.0,
                                          capillary_number=9.1,
                                          reynolds_number=128,
                                          atwood_number=1.0,
                                          peclet_number=744,
                                          density_ratio=3,
                                          viscosity_ratio=3)

    assert np.isclose(parameters.density_light, 1/3)
    assert np.isclose(parameters.gravitational_acceleration, -3.9506172839506174e-07)
    assert np.isclose(parameters.mobility, 0.0012234169653524492)

    rs = analytic_rising_speed(1 - 6, 20, 0.01)
    assert np.isclose(rs, 16666.666666666668)

    parameters = calculate_parameters_taylor_bubble(reference_length=192,
                                                    reference_time=36000,
                                                    density_heavy=1.0,
                                                    diameter_outer=0.0254,
                                                    diameter_inner=0.0127)

    assert np.isclose(parameters.density_heavy, 1.0)
    assert np.isclose(parameters.density_light, 0.001207114228456914)
    assert np.isclose(parameters.dynamic_viscosity_heavy, 5.733727652152216e-05)
    assert np.isclose(parameters.dynamic_viscosity_light, 0.0008630017037694861)
    assert np.isclose(parameters.gravitational_acceleration, -7.407407407407407e-08)
    assert np.isclose(parameters.surface_tension, 3.149857262258028e-05, rtol=1e-05)
