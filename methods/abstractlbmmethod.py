import abc
import sympy as sp


class AbstractLbmMethod(metaclass=abc.ABCMeta):

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
    def availableMacroscopicQuantities(self):
        """Returns a dict of string to symbol(s), where each entry is a computable macroscopic quantity"""
        pass

    @abc.abstractproperty
    def conservedQuantitiesSymbols(self):
        """Returns symbols representing conserved quantities (subset of macroscopic quantities)"""
        return

    @abc.abstractmethod
    def getMacroscopicQuantitiesEquations(self, macroscopicQuantities):
        """Returns equation collection defining conserved quantities. The passed symbols have to be keys in
        of the dict returned by :func:`availableMacroscopicQuantities`"""
        pass

    @abc.abstractmethod
    def getEquilibrium(self):
        """Returns equation collection, to compute equilibrium values.
        The equations have the post collision symbols as left hand sides and are
        functions of the conserved quantities"""
        return

    @abc.abstractmethod
    def getCollisionRule(self):
        """Returns an equation collection defining the collision operator."""
        return
