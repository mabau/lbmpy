import abc
from collections import namedtuple

import sympy as sp

from pystencils import AssignmentCollection

RelaxationInfo = namedtuple('RelaxationInfo', ['equilibrium_value', 'relaxation_rate'])


class LbmCollisionRule(AssignmentCollection):
    def __init__(self, lb_method, *args, **kwargs):
        super(LbmCollisionRule, self).__init__(*args, **kwargs)
        self.method = lb_method


class AbstractLbMethod(abc.ABC):
    """Abstract base class for all LBM methods."""

    def __init__(self, stencil):
        self._stencil = stencil

    @property
    def stencil(self):
        """Discrete set of velocities, represented as nested tuple"""
        return self._stencil

    @property
    def dim(self):
        return len(self.stencil[0])

    @property
    def pre_collision_pdf_symbols(self):
        """Tuple of symbols representing the pdf values before collision"""
        return sp.symbols("f_:%d" % (len(self.stencil),))

    @property
    def post_collision_pdf_symbols(self):
        """Tuple of symbols representing the pdf values after collision"""
        return sp.symbols("d_:%d" % (len(self.stencil),))

    # ------------------------- Abstract Methods & Properties ----------------------------------------------------------

    @abc.abstractmethod
    def conserved_quantity_computation(self):
        """Returns an instance of class :class:`lbmpy.methods.AbstractConservedQuantityComputation`"""

    @abc.abstractmethod
    def weights(self):
        """Returns a sequence of weights, one for each lattice direction"""

    @abc.abstractmethod
    def get_equilibrium(self):
        """Returns equation collection, to compute equilibrium values.
        The equations have the post collision symbols as left hand sides and are
        functions of the conserved quantities"""

    @abc.abstractmethod
    def get_collision_rule(self):
        """Returns an LbmCollisionRule i.e. an equation collection with a reference to the method.
         This collision rule defines the collision operator."""
