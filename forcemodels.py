"""
.. module:: forcemodels
    :synopsis: Collection of forcing terms for hydrodynamic LBM simulations

"""
import sympy as sp
import lbmpy.util as util
from lbmpy.equilibria import getWeights
from lbmpy.util import scalarProduct


class Simple:
    """
    A simple force model which introduces the following additional force term in the
    collision process: ::math:`3 * w_i * e_i * f_i` (often: force = rho * acceleration)
    Should only be used with constant forces!
    Shifts the macroscopic velocity by F/2, but does not change the equilibrium velocity.
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, latticeModel):
        dim = len(latticeModel.stencil[0])
        assert len(self._force) == dim
        weights = getWeights(latticeModel.stencil)
        return [3 * w_i * scalarProduct(self._force, direction)
                for direction, w_i in zip(latticeModel.stencil, weights)]


class Luo:
    """
    Force model by Luo with the following forcing term

    .. math ::

            F_i = w_i * \left( \frac{c_i - u}{c_s^2} + \frac{c_i * (c_i * u)}{c_s^4} \right) * F

    Shifts the macroscopic velocity by F/2, but does not change the equilibrium velocity.
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, latticeModel):
        dim = len(latticeModel.stencil[0])
        assert len(self._force) == dim
        u = sp.Matrix(util.getSymbolicVelocityVector(dim))

        force = sp.Matrix(self._force)

        weights = getWeights(latticeModel.stencil)
        result = []
        for direction, w_i in zip(latticeModel.stencil, weights):
            direction = sp.Matrix(direction)
            result.append(3 * w_i * force.dot(direction - u + 3 * direction * direction.dot(u)))
        return result

    def macroscopicVelocity(self, latticeModel, vel, density):
        return defaultVelocityShift(vel, density, self._force, latticeModel.compressible)


class Guo:
    """
     Force model by Guo with the following term:

    .. math ::

        F_i = w_i * ( 1 - \frac{1}{2 * tau} ) * \left( \frac{c_i - u}{c_s^2} + \frac{c_i * (c_i * u)}{c_s^4} \right) * F

    Adapts the calculation of the macroscopic velocity as well as the equilibrium velocity (both shifted by F/2)!
    """
    def __init__(self, force, viscosityRelaxationRate):
        self._force = force
        self._viscosityRelaxationRate = viscosityRelaxationRate

    def __call__(self, latticeModel):
        luo = Luo(self._force)
        correctionFactor = (1 - sp.Rational(1, 2) * self._viscosityRelaxationRate)
        return [correctionFactor * t for t in luo(latticeModel.stencil)]

    def macroscopicVelocity(self, latticeModel, vel, density):
        return defaultVelocityShift(vel, density, self._force, latticeModel.compressible)

    def equilibriumVelocity(self, latticeModel, vel, density):
        return defaultVelocityShift(vel, density, self._force, latticeModel.compressible)


# --------------------------------  Helper functions  ------------------------------------------------------------------


def defaultVelocityShift(velocity, density, force, compressible):
    if compressible:
        return [v_i + f_i / (2 * density) for v_i, f_i in zip(velocity, force)]
    else:
        return [v_i + f_i / 2 for v_i, f_i in zip(velocity, force)]

