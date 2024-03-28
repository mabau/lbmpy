import sympy as sp
from abc import ABC, abstractmethod

from pystencils import Assignment


class WallFunctionModel(ABC):
    def __init__(self, name):
        self._name = name

    @abstractmethod
    def shear_stress_assignments(self, density_symbol: sp.Symbol, shear_stress_symbol: sp.Symbol,
                                 velocity_symbol: sp.Symbol, wall_distance, u_tau_target):
        """
        Computes a symbolic representation for the log law.

        Args:
            density_symbol: symbol density, should be provided by the LB method's conserved quantity computation
            shear_stress_symbol: symbolic wall shear stress to which the calculated shear stress will be assigned
            velocity_symbol: symbolic velocity that is taken as a reference in the wall functions
            wall_distance: distance to the wall, equals to 0.5 in standard cell-centered LBM
            u_tau_target: in implicit wall functions, a target friction velocity can be provided which will be used as
                          initial guess in the Newton iteration. This target friction velocity can be obtained, e.g.,
                          from the target friction Reynolds number
        """
        pass


# end class WallFunctionModel


class ExplicitWallFunctionModel(WallFunctionModel, ABC):
    """
    Abstract base class for explicit wall functions that can be solved directly for the wall shear stress.
    """

    def __init__(self, name):
        super(ExplicitWallFunctionModel, self).__init__(name=name)


class MoninObukhovSimilarityTheory(ExplicitWallFunctionModel):
    def __init__(self, z0, kappa=0.41, phi=0, name="MOST"):
        self.z0 = z0
        self.kappa = kappa
        self.phi = phi

        super(MoninObukhovSimilarityTheory, self).__init__(name=name)

    def shear_stress_assignments(self, density_symbol: sp.Symbol, shear_stress_symbol: sp.Symbol,
                                 velocity_symbol: sp.Symbol, wall_distance, u_tau_target=None):
        u_tau = velocity_symbol * self.kappa / sp.ln(wall_distance / self.z0 + self.phi)
        return [Assignment(shear_stress_symbol, u_tau ** 2 * density_symbol)]


class ImplicitWallFunctionModel(WallFunctionModel, ABC):
    """
    Abstract base class for implicit wall functions that require a Newton procedure to solve for the wall shear stress.
    """

    def __init__(self, name, newton_steps, viscosity):
        self.newton_steps = newton_steps
        self.u_tau = sp.symbols(f"wall_function_u_tau_:{self.newton_steps + 1}")
        self.delta = sp.symbols(f"wall_function_delta_:{self.newton_steps}")

        self.viscosity = viscosity

        super(ImplicitWallFunctionModel, self).__init__(name=name)

    def newton_iteration(self, wall_law):
        m = -wall_law / wall_law.diff(self.u_tau[0])

        assignments = []
        for i in range(self.newton_steps):
            assignments.append(Assignment(self.delta[i], m.subs({self.u_tau[0]: self.u_tau[i]})))
            assignments.append(Assignment(self.u_tau[i + 1], self.u_tau[i] + self.delta[i]))

        return assignments


class LogLaw(ImplicitWallFunctionModel):
    """
    Analytical model for the velocity profile inside the boundary layer, obtained from the mean velocity gradient.
    Only valid in the log-law region.
    """

    def __init__(self, viscosity, newton_steps=5, kappa=0.41, b=5.2, name="LogLaw"):
        self.kappa = kappa
        self.b = b

        super(LogLaw, self).__init__(name=name, newton_steps=newton_steps, viscosity=viscosity)

    def shear_stress_assignments(self, density_symbol: sp.Symbol, shear_stress_symbol: sp.Symbol,
                                 velocity_symbol: sp.Symbol, wall_distance, u_tau_target=None):
        def law(u_p, y_p):
            return 1 / self.kappa * sp.ln(y_p) + self.b - u_p

        u_plus = velocity_symbol / self.u_tau[0]
        y_plus = (wall_distance * self.u_tau[0]) / self.viscosity

        u_tau_init = u_tau_target if u_tau_target else velocity_symbol / sp.Float(100)

        wall_law = law(u_plus, y_plus)
        assignments = [Assignment(self.u_tau[0], u_tau_init),  # initial guess
                       *self.newton_iteration(wall_law),  # newton iterations
                       Assignment(shear_stress_symbol, self.u_tau[-1] ** 2 * density_symbol)]  # final result

        return assignments


class SpaldingsLaw(ImplicitWallFunctionModel):
    """
    Single formula for the velocity profile inside the boundary layer, proposed by Spalding :cite:`spalding1961`.
    Valid in the inner and the outer layer.
    """

    def __init__(self, viscosity, newton_steps=5, kappa=0.41, b=5.5, name="Spalding"):
        self.kappa = kappa
        self.b = b

        super(SpaldingsLaw, self).__init__(name=name, newton_steps=newton_steps, viscosity=viscosity)

    def shear_stress_assignments(self, density_symbol: sp.Symbol, shear_stress_symbol: sp.Symbol,
                                 velocity_symbol: sp.Symbol, wall_distance, u_tau_target=None):
        def law(u_p, y_p):
            k_times_u = self.kappa * u_p
            frac_1 = (k_times_u ** 2) / sp.Float(2)
            frac_2 = (k_times_u ** 3) / sp.Float(6)
            return (u_p + sp.exp(-self.kappa * self.b) * (sp.exp(k_times_u) - sp.Float(1) - k_times_u - frac_1 - frac_2)
                    - y_p)

        u_plus = velocity_symbol / self.u_tau[0]
        y_plus = (wall_distance * self.u_tau[0]) / self.viscosity

        u_tau_init = u_tau_target if u_tau_target else velocity_symbol / sp.Float(100)

        wall_law = law(u_plus, y_plus)
        assignments = [Assignment(self.u_tau[0], u_tau_init),  # initial guess
                       *self.newton_iteration(wall_law),  # newton iterations
                       Assignment(shear_stress_symbol, self.u_tau[-1] ** 2 * density_symbol)]  # final result

        return assignments


class MuskerLaw(ImplicitWallFunctionModel):
    """
    Quasi-analytical model for the velocity profile inside the boundary layer, proposed by Musker. Valid in the inner
    and the outer layer.
    Formulation taken from :cite:`malaspinas2015`, Equation (59).
    """

    def __init__(self, viscosity, newton_steps=5, name="Musker"):

        super(MuskerLaw, self).__init__(name=name, newton_steps=newton_steps, viscosity=viscosity)

    def shear_stress_assignments(self, density_symbol: sp.Symbol, shear_stress_symbol: sp.Symbol,
                                 velocity_symbol: sp.Symbol, wall_distance, u_tau_target=None):
        def law(u_p, y_p):
            arctan = sp.Float(5.424) * sp.atan(sp.Float(0.119760479041916168) * y_p - sp.Float(0.488023952095808383))
            logarithm = (sp.Float(0.434) * sp.log((y_p + sp.Float(10.6)) ** sp.Float(9.6)
                                                  / (y_p ** 2 - sp.Float(8.15) * y_p + sp.Float(86)) ** 2), 10)
            return (arctan + logarithm - sp.Float(3.50727901936264842)) - u_p

        u_plus = velocity_symbol / self.u_tau[0]
        y_plus = (wall_distance * self.u_tau[0]) / self.viscosity

        u_tau_init = u_tau_target if u_tau_target else velocity_symbol / sp.Float(100)

        wall_law = law(u_plus, y_plus)
        assignments = [Assignment(self.u_tau[0], u_tau_init),  # initial guess
                       *self.newton_iteration(wall_law),  # newton iterations
                       Assignment(shear_stress_symbol, self.u_tau[-1] ** 2 * density_symbol)]  # final result

        return assignments
