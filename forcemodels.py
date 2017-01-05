"""
.. module:: forcemodels
    :synopsis: Collection of forcing terms for hydrodynamic LBM simulations

"""
import sympy as sp


class Simple:
    """
    A simple force model which introduces the following additional force term in the
    collision process: ::math:`3 * w_i * e_i * f_i` (often: force = rho * acceleration)
    Should only be used with constant forces!
    Shifts the macroscopic velocity by F/2, but does not change the equilibrium velocity.
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, lbmMethod, **kwargs):
        assert len(self._force) == lbmMethod.dim
        return [3 * w_i * sum([d_i * f_i for d_i, f_i in zip(direction, self._force)])
                for direction, w_i in zip(lbmMethod.stencil, lbmMethod.weights)]


class Luo:
    """
    Force model by Luo with the following forcing term

    .. math ::

            F_i = w_i * \left( \frac{c_i - u}{c_s^2} + \frac{c_i * (c_i * u)}{c_s^4} \right) * F

    Shifts the macroscopic velocity by F/2, but does not change the equilibrium velocity.
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, lbmMethod):
        u = sp.Matrix(lbmMethod.firstOrderEquilibriumMomentSymbols)
        force = sp.Matrix(self._force)

        result = []
        for direction, w_i in zip(lbmMethod.stencil, lbmMethod.weights):
            direction = sp.Matrix(direction)
            result.append(3 * w_i * force.dot(direction - u + 3 * direction * direction.dot(u)))
        return result

    def macroscopicVelocityShift(self, density):
        return defaultVelocityShift(density, self._force)


class Guo:
    """
     Force model by Guo with the following term:

    .. math ::

        F_i = w_i * ( 1 - \frac{1}{2 * tau} ) * \left( \frac{c_i - u}{c_s^2} + \frac{c_i * (c_i * u)}{c_s^4} \right) * F

    Adapts the calculation of the macroscopic velocity as well as the equilibrium velocity (both shifted by F/2)!
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, lbmMethod):
        luo = Luo(self._force)
        u = lbmMethod.firstOrderEquilibriumMomentSymbols
        shearRelaxationRate = lbmMethod.getShearRelaxationRate()
        correctionFactor = (1 - sp.Rational(1, 2) * shearRelaxationRate)
        return [correctionFactor * t for t in luo(lbmMethod)]

    def macroscopicVelocityShift(self, density):
        return defaultVelocityShift(density, self._force)

    def equilibriumVelocityShift(self, density):
        return defaultVelocityShift(density, self._force)


# --------------------------------  Helper functions  ------------------------------------------------------------------


def defaultVelocityShift(density, force):
    return [f_i / (2 * density) for f_i in force]

