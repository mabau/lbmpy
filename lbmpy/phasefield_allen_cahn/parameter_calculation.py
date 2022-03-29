import math
import sympy as sp


class AllenCahnParameters:
    def __init__(self, density_heavy: float, density_light: float,
                 dynamic_viscosity_heavy: float, dynamic_viscosity_light: float,
                 surface_tension: float, mobility: float = 0.2,
                 gravitational_acceleration: float = 0.0, interface_thickness: int = 5):

        self.density_heavy = density_heavy
        self.density_light = density_light
        self.dynamic_viscosity_heavy = dynamic_viscosity_heavy
        self.dynamic_viscosity_light = dynamic_viscosity_light
        self.surface_tension = surface_tension
        self.mobility = mobility
        self.gravitational_acceleration = gravitational_acceleration
        self.interface_thickness = interface_thickness

    @property
    def kinematic_viscosity_heavy(self):
        return self.dynamic_viscosity_heavy / self.density_heavy

    @property
    def kinematic_viscosity_light(self):
        return self.dynamic_viscosity_light / self.density_light

    @property
    def relaxation_time_heavy(self):
        return 3.0 * self.kinematic_viscosity_heavy

    @property
    def relaxation_time_light(self):
        return 3.0 * self.kinematic_viscosity_light

    @property
    def omega_phi(self):
        return 1.0 / (0.5 + (3.0 * self.mobility))

    @property
    def symbolic_density_heavy(self):
        return sp.Symbol("rho_H")

    @property
    def symbolic_density_light(self):
        return sp.Symbol("rho_L")

    @property
    def symbolic_tau_heavy(self):
        return sp.Symbol("tau_H")

    @property
    def symbolic_tau_light(self):
        return sp.Symbol("tau_L")

    @property
    def symbolic_omega_phi(self):
        return sp.Symbol("omega_phi")

    @property
    def symbolic_surface_tension(self):
        return sp.Symbol("sigma")

    @property
    def symbolic_mobility(self):
        return sp.Symbol("M_m")

    @property
    def symbolic_gravitational_acceleration(self):
        return sp.Symbol("F_g")

    @property
    def symbolic_interface_thickness(self):
        return sp.Symbol("W")

    @property
    def beta(self):
        return sp.Rational(12, 1) * (self.symbolic_surface_tension / self.symbolic_interface_thickness)

    @property
    def kappa(self):
        return sp.Rational(3, 2) * self.symbolic_surface_tension * self.symbolic_interface_thickness

    def omega(self, phase_field):
        tau_L = self.symbolic_tau_light
        tau_H = self.symbolic_tau_heavy
        tau = sp.Rational(1, 2) + tau_L + phase_field.center * (tau_H - tau_L)
        return sp.simplify(1 / tau)

    def parameter_map(self):
        result = {self.symbolic_density_heavy: self.density_heavy,
                  self.symbolic_density_light: self.density_light,
                  self.symbolic_tau_heavy: self.relaxation_time_heavy,
                  self.symbolic_tau_light: self.relaxation_time_light,
                  self.symbolic_omega_phi: self.omega_phi,
                  self.symbolic_gravitational_acceleration: self.gravitational_acceleration,
                  self.symbolic_interface_thickness: self.interface_thickness,
                  self.symbolic_mobility: self.mobility,
                  self.symbolic_surface_tension: self.surface_tension}
        return result

    @property
    def symbolic_to_numeric_map(self):
        return {t.name: self.parameter_map()[t] for t in self.parameter_map()}

    def _repr_html_(self):
        names = ("Density heavy phase",
                 "Density light phase",
                 "Relaxation time heavy phase",
                 "Relaxation time light phase",
                 "Relaxation rate Allen Cahn LB",
                 "Gravitational acceleration",
                 "Interface thickness",
                 "Mobility",
                 "Surface tension")

        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Name</th>
                <th {nb} >SymPy Symbol </th>
                <th {nb} >Value</th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for name, (symbol, value) in zip(names, self.parameter_map().items()):
            vals = {
                'Name': name,
                'Sympy Symbol': sp.latex(symbol),
                'numeric value': sp.latex(value),
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>{Name}</td>
                            <td {nb}>${Sympy Symbol}$</td>
                            <td {nb}>${numeric value}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')


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

    surface_tension = (dynamic_viscosity_heavy * reference_velocity) / capillary_number
    mobility = (reference_velocity * reference_length) / peclet_number

    parameters = AllenCahnParameters(density_heavy=density_heavy,
                                     density_light=density_light,
                                     dynamic_viscosity_heavy=dynamic_viscosity_heavy,
                                     dynamic_viscosity_light=dynamic_viscosity_light,
                                     surface_tension=surface_tension,
                                     mobility=mobility,
                                     gravitational_acceleration=-g)
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

    surface_tension = (density_heavy - density_light) * g * bubble_diameter ** 2 / bond_number
    # calculation of the Morton number
    # Mo = gravitational_acceleration * dynamic_viscosity_heavy / (density_heavy * surface_tension ** 3)

    parameters = AllenCahnParameters(density_heavy=density_heavy,
                                     density_light=density_light,
                                     dynamic_viscosity_heavy=dynamic_viscosity_heavy,
                                     dynamic_viscosity_light=dynamic_viscosity_light,
                                     surface_tension=surface_tension,
                                     gravitational_acceleration=-g)
    return parameters
