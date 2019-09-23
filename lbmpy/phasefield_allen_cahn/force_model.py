import sympy as sp
import numpy as np

from lbmpy.forcemodels import Simple


class MultiphaseForceModel:
    r"""
    A force model based on PhysRevE.96.053301. This model realises the modified equilibrium distributions meaning the
    force gets shifted by minus one half multiplied with the collision operator
    """
    def __init__(self, force, rho=1):
        self._force = force
        self._rho = rho

    def __call__(self, lb_method):

        simple = Simple(self._force)
        force = [f / self._rho for f in simple(lb_method)]

        moment_matrix = lb_method.moment_matrix
        relaxation_rates = sp.Matrix(np.diag(lb_method.relaxation_rates))
        mrt_collision_op = moment_matrix.inv() * relaxation_rates * moment_matrix

        return -0.5 * mrt_collision_op * sp.Matrix(force) + sp.Matrix(force)

    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(self._rho, self._force)

# --------------------------------  Helper functions  ------------------------------------------------------------------


def default_velocity_shift(density, force):
    return [f_i / (2 * density) for f_i in force]
