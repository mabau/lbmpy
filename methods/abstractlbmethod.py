import abc
import sympy as sp
from collections import namedtuple
from pystencils.equationcollection import EquationCollection


RelaxationInfo = namedtuple('Relaxationinfo', ['equilibriumValue', 'relaxationRate'])


class LbmCollisionRule(EquationCollection):
    def __init__(self, lbmMethod, *args, **kwargs):
        super(LbmCollisionRule, self).__init__(*args, **kwargs)
        self.method = lbmMethod


class AbstractLbMethod(abc.ABCMeta('ABC', (object,), {})):
    """
    Abstract base class for all LBM methods
    """

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
    def preCollisionPdfSymbols(self):
        """Tuple of symbols representing the pdf values before collision"""
        return sp.symbols("f_:%d" % (len(self.stencil),))

    @property
    def postCollisionPdfSymbols(self):
        """Tuple of symbols representing the pdf values after collision"""
        return sp.symbols("d_:%d" % (len(self.stencil),))

    # ------------------------- Abstract Methods & Properties ----------------------------------------------------------

    @abc.abstractproperty
    def conservedQuantityComputation(self):
        """Returns an instance of class :class:`lbmpy.methods.AbstractConservedQuantityComputation`"""

    @abc.abstractproperty
    def weights(self):
        """Returns a sequence of weights, one for each lattice direction"""

    @abc.abstractmethod
    def getEquilibrium(self):
        """Returns equation collection, to compute equilibrium values.
        The equations have the post collision symbols as left hand sides and are
        functions of the conserved quantities"""

    @abc.abstractmethod
    def getCollisionRule(self):
        """Returns an LbmCollisionRule i.e. an equation collection with a reference to the method.
         This collision rule defines the collision operator."""

