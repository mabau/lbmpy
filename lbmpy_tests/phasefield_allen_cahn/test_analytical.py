import numpy as np

from lbmpy.phasefield_allen_cahn.parameter_calculation import calculate_dimensionless_rising_bubble, \
    calculate_parameters_rti

from lbmpy.phasefield_allen_cahn.analytical import analytic_rising_speed


def test_analytical():
    parameters = calculate_dimensionless_rising_bubble(reference_time=18000,
                                                       density_heavy=1.0,
                                                       bubble_radius=16,
                                                       bond_number=30,
                                                       reynolds_number=420,
                                                       density_ratio=1000,
                                                       viscosity_ratio=100)

    np.isclose(parameters.density_light, 0.001, rtol=1e-05, atol=1e-08, equal_nan=False)
    np.isclose(parameters.gravitational_acceleration, -9.876543209876543e-08, rtol=1e-05, atol=1e-08, equal_nan=False)

    parameters = calculate_parameters_rti(reference_length=128,
                                          reference_time=18000,
                                          density_heavy=1.0,
                                          capillary_number=9.1,
                                          reynolds_number=128,
                                          atwood_number=1.0,
                                          peclet_number=744,
                                          density_ratio=3,
                                          viscosity_ratio=3)

    np.isclose(parameters.density_light, 1/3, rtol=1e-05, atol=1e-08, equal_nan=False)
    np.isclose(parameters.gravitational_acceleration, -3.9506172839506174e-07,
               rtol=1e-05, atol=1e-08, equal_nan=False)
    np.isclose(parameters.mobility, 0.0012234169653524492, rtol=1e-05, atol=1e-08, equal_nan=False)

    rs = analytic_rising_speed(1-6, 20, 0.01)
    np.isclose(rs, 16666.666666666668, rtol=1e-05, atol=1e-08, equal_nan=False)
