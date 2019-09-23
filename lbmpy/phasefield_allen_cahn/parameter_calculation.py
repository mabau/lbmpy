import math


def calculate_parameters_rti(reference_length=256,
                             reference_time=16000,
                             density_heavy=1.0,
                             capillary_number=0.26,
                             reynolds_number=3000,
                             atwood_number=0.5,
                             peclet_number=1000,
                             density_ratio=3,
                             viscosity_ratio=1):
    r"""
    Calculate the simulation parameters for the Rayleigh-Taylor instability.
    The calculation is done according to the description in part B of PhysRevE.96.053301.

    Args:
        reference_length: reference length of the RTI
        reference_time: chosen reference time
        density_heavy: density of the heavier fluid
        capillary_number: capillary number of the simulation
        reynolds_number: reynolds number of the simulation
        atwood_number: atwood number of the simulation
        peclet_number: peclet number of the simulation
        density_ratio: density ration of the heavier and the lighter fluid
        viscosity_ratio: viscosity ratio of the heavier and the lighter fluid
    """
    # calculate the gravitational acceleration and the reference velocity
    gravitational_acceleration = reference_length / (reference_time ** 2 * atwood_number)
    reference_velocity = math.sqrt(gravitational_acceleration * reference_length)

    dynamic_viscosity_heavy = (density_heavy * reference_velocity * reference_length) / reynolds_number
    dynamic_viscosity_light = dynamic_viscosity_heavy / viscosity_ratio

    density_light = density_heavy / density_ratio

    kinematic_viscosity_heavy = dynamic_viscosity_heavy / density_heavy
    kinematic_viscosity_light = dynamic_viscosity_light / density_light

    relaxation_time_heavy = 3.0 * kinematic_viscosity_heavy
    relaxation_time_light = 3.0 * kinematic_viscosity_light

    surface_tension = (dynamic_viscosity_heavy * reference_velocity) / capillary_number
    mobility = (reference_velocity * reference_length) / peclet_number

    parameters = {
        "density_light": density_light,
        "dynamic_viscosity_heavy": dynamic_viscosity_heavy,
        "dynamic_viscosity_light": dynamic_viscosity_light,
        "relaxation_time_heavy": relaxation_time_heavy,
        "relaxation_time_light": relaxation_time_light,
        "gravitational_acceleration": -gravitational_acceleration,
        "mobility": mobility,
        "surface_tension": surface_tension
    }
    return parameters
