
def analytic_rising_speed(gravitational_acceleration, bubble_diameter, viscosity_gas):
    r"""
    Calculated the analytical rising speed of a bubble. This is the expected end rising speed.
    Args:
        gravitational_acceleration: the gravitational acceleration acting in the simulation scenario. Usually it gets
                                    calculated based on dimensionless parameters which describe the scenario
        bubble_diameter: the diameter of the bubble at the beginning of the simulation
        viscosity_gas: the viscosity of the fluid inside the bubble
    """
    result = -(gravitational_acceleration * bubble_diameter * bubble_diameter) / (12.0 * viscosity_gas)
    return result
