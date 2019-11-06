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
        capillary_number: capillary number of the simulation represents the relative effect of viscous drag
                          forces versus surface tension forces
        reynolds_number: reynolds number of the simulation is the ratio between the viscous forces in a fluid
                         and the inertial forces
        atwood_number: atwood number of the simulation is a dimensionless density ratio
        peclet_number: peclet number of the simulation is the ratio of the rate of advection
                       of a physical quantity by the flow to the rate of diffusion of the same quantity
                       driven by an appropriate gradient
        density_ratio: density ratio of the heavier and the lighter fluid
        viscosity_ratio: viscosity ratio of the heavier and the lighter fluid
    """

    # calculate the gravitational acceleration and the reference velocity
    g = reference_length / (reference_time ** 2 * atwood_number)
    reference_velocity = math.sqrt(g * reference_length)

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
        "gravitational_acceleration": -g,
        "mobility": mobility,
        "surface_tension": surface_tension
    }
    return parameters


def calculate_dimensionless_rising_bubble(reference_time=18000,
                                          density_heavy=1.0,
                                          bubble_radius=16,
                                          bond_number=1,
                                          reynolds_number=40,
                                          density_ratio=1000,
                                          viscosity_ratio=100):
    r"""
    Calculate the simulation parameters for a rising bubble. The parameter calculation follows the ideas of Weber and
    is based on the Reynolds number. This means the rising velocity of the bubble is implicitly stated with the
    Reynolds number

    Args:
        reference_time: chosen reference time
        density_heavy: density of the heavier fluid
        bubble_radius: initial radius of the rising bubble
        bond_number: also called eötvös number is measuring the importance of gravitational forces compared to
                     surface tension forces
        reynolds_number: reynolds number of the simulation is the ratio between the viscous forces in a fluid
                 and the inertial forces
        density_ratio: density ratio of the heavier and the lighter fluid
        viscosity_ratio: viscosity ratio of the heavier and the lighter fluid
    """

    bubble_diameter = bubble_radius * 2
    g = bubble_diameter / (reference_time ** 2)

    density_light = density_heavy / density_ratio

    dynamic_viscosity_heavy = (density_heavy * math.sqrt(g * bubble_diameter ** 3)) / reynolds_number
    dynamic_viscosity_light = dynamic_viscosity_heavy / viscosity_ratio

    kinematic_viscosity_heavy = dynamic_viscosity_heavy / density_heavy
    kinematic_viscosity_light = dynamic_viscosity_light / density_light

    relaxation_time_heavy = 3 * kinematic_viscosity_heavy
    relaxation_time_light = 3 * kinematic_viscosity_light

    surface_tension = (density_heavy - density_light) * g * bubble_diameter ** 2 / bond_number
    # calculation of the Morton number
    # Mo = gravitational_acceleration * dynamic_viscosity_heavy / (density_heavy * surface_tension ** 3)

    parameters = {
        "density_light": density_light,
        "dynamic_viscosity_heavy": dynamic_viscosity_heavy,
        "dynamic_viscosity_light": dynamic_viscosity_light,
        "relaxation_time_heavy": relaxation_time_heavy,
        "relaxation_time_light": relaxation_time_light,
        "gravitational_acceleration": -g,
        "surface_tension": surface_tension
    }
    return parameters
