import sympy as sp

from pystencils import Assignment
from lbmpy.forcemodels import Simple, Luo


class MultiphaseForceModel:
    r"""
    A force model based on PhysRevE.96.053301. This model realises the modified equilibrium distributions meaning the
    force gets shifted by minus one half multiplied with the collision operator
    """
    def __init__(self, force, rho=1):
        self._force = force
        self._rho = rho
        self.force_symp = sp.symbols(f"F_:{len(force)}")
        self.subs_terms = [Assignment(rhs, lhs) for rhs, lhs in zip(self.force_symp, force)]

    def __call__(self, lb_method):
        simple = Simple(self.force_symp)
        force = sp.Matrix(simple(lb_method))

        moment_matrix = lb_method.moment_matrix

        return sp.simplify(moment_matrix * force) / self._rho


class CentralMomentMultiphaseForceModel:
    r"""
    A simple force model in the central moment space.
    """
    def __init__(self, force, rho=1):
        self._force = force
        self._rho = rho
        self.force_symp = sp.symbols(f"F_:{len(force)}")
        self.subs_terms = [Assignment(rhs, lhs) for rhs, lhs in zip(self.force_symp, force)]

    def __call__(self, lb_method, **kwargs):
        luo = Luo(self.force_symp)
        force = sp.Matrix(luo(lb_method))

        M = lb_method.moment_matrix
        N = lb_method.shift_matrix

        result = sp.simplify(M * force)
        return sp.simplify(N * result) / self._rho
