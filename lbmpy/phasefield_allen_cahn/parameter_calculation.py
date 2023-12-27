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

    @staticmethod
    def _parameter_strings():
        names = ("Density heavy phase",
                 "Density light phase",
                 "Relaxation time heavy phase",
                 "Relaxation time light phase",
                 "Relaxation rate Allen Cahn LB",
                 "Gravitational acceleration",
                 "Interface thickness",
                 "Mobility",
                 "Surface tension")
        return names

    def _repr_html_(self):
        names = self._parameter_strings()
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


class ThermocapillaryParameters(AllenCahnParameters):
    def __init__(self, density_heavy: float, density_light: float,
                 dynamic_viscosity_heavy: float, dynamic_viscosity_light: float,
                 surface_tension: float,
                 heat_conductivity_heavy: float, heat_conductivity_light: float,
                 mobility: float = 0.2,
                 gravitational_acceleration: float = 0.0, interface_thickness: int = 5,
                 sigma_ref: float = 5e-3, sigma_t: float = 2e-4, tmp_ref: float = 0,
                 velocity_wall: float = 0.0, reference_time: int = 10):

        self.heat_conductivity_heavy = heat_conductivity_heavy
        self.heat_conductivity_light = heat_conductivity_light
        self.sigma_ref = sigma_ref
        self.sigma_t = sigma_t
        self.tmp_ref = tmp_ref
        self.velocity_wall = velocity_wall
        self.reference_time = reference_time

        super(ThermocapillaryParameters, self).__init__(density_heavy, density_light,
                                                        dynamic_viscosity_heavy, dynamic_viscosity_light,
                                                        surface_tension, mobility,
                                                        gravitational_acceleration, interface_thickness)

    @property
    def symbolic_heat_conductivity_heavy(self):
        return sp.Symbol("kappa_H")

    @property
    def symbolic_heat_conductivity_light(self):
        return sp.Symbol("kappa_L")

    @property
    def symbolic_sigma_ref(self):
        return sp.Symbol("sigma_ref")

    @property
    def symbolic_sigma_t(self):
        return sp.Symbol("sigma_T")

    @property
    def symbolic_tmp_ref(self):
        return sp.Symbol("T_ref")

    def parameter_map(self):
        result = {self.symbolic_density_heavy: self.density_heavy,
                  self.symbolic_density_light: self.density_light,
                  self.symbolic_tau_heavy: self.relaxation_time_heavy,
                  self.symbolic_tau_light: self.relaxation_time_light,
                  self.symbolic_omega_phi: self.omega_phi,
                  self.symbolic_gravitational_acceleration: self.gravitational_acceleration,
                  self.symbolic_interface_thickness: self.interface_thickness,
                  self.symbolic_mobility: self.mobility,
                  self.symbolic_surface_tension: self.surface_tension,
                  self.symbolic_heat_conductivity_heavy: self.heat_conductivity_heavy,
                  self.symbolic_heat_conductivity_light: self.heat_conductivity_light,
                  self.symbolic_sigma_ref: self.sigma_ref,
                  self.symbolic_sigma_t: self.sigma_t,
                  self.symbolic_tmp_ref: self.tmp_ref}
        return result

    @staticmethod
    def _parameter_strings():
        names = ("Density heavy phase",
                 "Density light phase",
                 "Relaxation time heavy phase",
                 "Relaxation time light phase",
                 "Relaxation rate Allen Cahn LB",
                 "Gravitational acceleration",
                 "Interface thickness",
                 "Mobility",
                 "Surface tension",
                 "Heat Conductivity Heavy",
                 "Heat Conductivity Light",
                 "Sigma Ref",
                 "Sigma T",
                 "Temperature Ref")
        return names


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
    gravity_lattice_units = reference_length / (reference_time ** 2 * atwood_number)
    reference_velocity = math.sqrt(gravity_lattice_units * reference_length)

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
                                     gravitational_acceleration=-gravity_lattice_units)
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

    bubble_d = bubble_radius * 2
    gravity_lattice_units = bubble_d / (reference_time ** 2)

    density_light = density_heavy / density_ratio

    dynamic_viscosity_heavy = (density_heavy * math.sqrt(gravity_lattice_units * bubble_d ** 3)) / reynolds_number
    dynamic_viscosity_light = dynamic_viscosity_heavy / viscosity_ratio

    surface_tension = (density_heavy - density_light) * gravity_lattice_units * bubble_d ** 2 / bond_number
    # calculation of the Morton number
    # Mo = gravitational_acceleration * dynamic_viscosity_heavy / (density_heavy * surface_tension ** 3)

    parameters = AllenCahnParameters(density_heavy=density_heavy,
                                     density_light=density_light,
                                     dynamic_viscosity_heavy=dynamic_viscosity_heavy,
                                     dynamic_viscosity_light=dynamic_viscosity_light,
                                     surface_tension=surface_tension,
                                     gravitational_acceleration=-gravity_lattice_units)
    return parameters


def calculate_parameters_taylor_bubble(reference_length=128,
                                       reference_time=16000,
                                       density_heavy=1.0,
                                       diameter_outer=0.0254,
                                       diameter_inner=0.0127):
    r"""
    Calculate the simulation parameters for a rising Taylor bubble in an annulus pipe. The calculation can be found in
    10.1016/S0009-2509(97)00210-8 by G. Das

    Args:
        reference_length: chosen reference length in lattice cells
        reference_time: chosen reference time in latte timesteps
        density_heavy: density of water in lattice units
        diameter_outer: diameter of the outer tube
        diameter_inner: diameter of the inner cylinder
    """

    # physical parameters #
    water_rho = 998  # kg/m3
    air_rho = 1.2047  # kg/m3
    surface_tension = 0.07286  # kg/s2
    water_mu = 1.002e-3  # kg/ms
    air_mu = 1.8205e-5  # kg/ms
    gravity = 9.81  # m/s2

    # water_nu = water_mu / water_rho  # m2/s
    # air_nu = air_mu / air_rho  # m2/s

    diameter_fluid = diameter_outer - diameter_inner
    diameter_ratio = diameter_outer / diameter_inner

    inverse_viscosity_number = math.sqrt((water_rho - air_rho) * water_rho * gravity * diameter_fluid ** 3) / water_mu
    bond_number = (water_rho - air_rho) * gravity * diameter_fluid ** 2 / surface_tension
    # morton_number = gravity * water_mu ** 4 * (water_rho - air_rho) / (water_rho ** 2 * surface_tension ** 3)

    diameter_lattice_untis = reference_length / diameter_ratio

    density_light = 1.0 / (water_rho / air_rho)
    diameter_fluid = reference_length - diameter_lattice_untis
    gravity_lattice_units = diameter_fluid / reference_time ** 2

    density_diff = density_heavy - density_light

    grav_df_cube = gravity_lattice_units * diameter_fluid ** 3
    water_mu_lattice_units = math.sqrt(density_diff * density_heavy * grav_df_cube) / inverse_viscosity_number
    air_mu_lattice_units = water_mu_lattice_units / (water_mu / air_mu)

    dynamic_viscosity_heavy = water_mu_lattice_units / density_heavy
    dynamic_viscosity_light = air_mu_lattice_units / density_light

    surface_tension_lattice_units = density_diff * gravity_lattice_units * diameter_fluid ** 2 / bond_number

    parameters = AllenCahnParameters(density_heavy=density_heavy,
                                     density_light=density_light,
                                     dynamic_viscosity_heavy=dynamic_viscosity_heavy,
                                     dynamic_viscosity_light=dynamic_viscosity_light,
                                     surface_tension=surface_tension_lattice_units,
                                     gravitational_acceleration=-gravity_lattice_units)
    return parameters


def calculate_parameters_droplet_migration(radius=32,
                                           reynolds_number=0.16,
                                           capillary_number=0.01,
                                           marangoni_number=0.08,
                                           peclet_number=1,
                                           viscosity_ratio=1,
                                           heat_ratio=1,
                                           interface_width=4,
                                           reference_surface_tension=5e-3,
                                           height=None):
    r"""
    Calculate the simulation parameters moving droplet with a laser heat source. The calculation can be found in
    https://doi.org/10.1016/j.jcp.2013.08.054 by Liu et al.

    Args:
        radius: radius of the droplet which functions as the reference length
        reynolds_number: Reynolds number of the simulation
        capillary_number: Capillary number of the simulation
        marangoni_number: Marangoni number of the simulation
        peclet_number: Peclet number of the simulation
        viscosity_ratio: viscosity ratio between the two fluids
        heat_ratio: ratio of the heat conductivity in the two fluids
        interface_width: Resolution of cells for the interface
        reference_surface_tension: reference surface tension
        height: height of the simulation domain. If not defined it is asumed to be 2 * radius of the droplet.

    """

    if height is None:
        height = 2 * radius

    u_char = math.sqrt(reynolds_number * capillary_number * reference_surface_tension / radius)
    gamma = u_char / radius
    u_wall = gamma * height

    kinematic_viscosity_heavy = radius * u_char / reynolds_number
    kinematic_viscosity_light = kinematic_viscosity_heavy / viscosity_ratio

    heat_conductivity_heavy = radius * u_char / marangoni_number
    heat_conductivity_light = heat_conductivity_heavy / heat_ratio

    mobility_of_interface = gamma * radius * interface_width / peclet_number

    parameters = ThermocapillaryParameters(density_heavy=1.0,
                                           density_light=1.0,
                                           dynamic_viscosity_heavy=kinematic_viscosity_heavy,
                                           dynamic_viscosity_light=kinematic_viscosity_light,
                                           surface_tension=0.0,
                                           heat_conductivity_heavy=heat_conductivity_heavy,
                                           heat_conductivity_light=heat_conductivity_light,
                                           mobility=mobility_of_interface,
                                           velocity_wall=u_wall, reference_time=int(1.0 / gamma))
    return parameters
