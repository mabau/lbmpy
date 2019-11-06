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
        stencil = lb_method.stencil

        force_symp = sp.symbols("force_:{}".format(lb_method.dim))
        simple = Simple(force_symp)
        force = [f / self._rho for f in simple(lb_method)]

        moment_matrix = lb_method.moment_matrix
        relaxation_rates = sp.Matrix(np.diag(lb_method.relaxation_rates))
        mrt_collision_op = moment_matrix.inv() * relaxation_rates * moment_matrix

        result = -0.5 * mrt_collision_op * sp.Matrix(force) + sp.Matrix(force)

        for i in range(0, len(stencil)):
            result[i] = result[i].simplify()

        subs_dict = dict(zip(force_symp, self._force))

        for i in range(0, len(stencil)):
            result[i] = result[i].subs(subs_dict)

        return result
