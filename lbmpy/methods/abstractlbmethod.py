import abc
from collections import namedtuple

import sympy as sp
from sympy.core.numbers import Zero

from pystencils import Assignment, AssignmentCollection
from lbmpy.stencils import LBStencil

RelaxationInfo = namedtuple('RelaxationInfo', ['equilibrium_value', 'relaxation_rate'])


class LbmCollisionRule(AssignmentCollection):
    """
    A pystencils AssignmentCollection that additionally holds an `AbstractLbMethod`
    """
    def __init__(self, lb_method, *args, **kwargs):
        super(LbmCollisionRule, self).__init__(*args, **kwargs)
        self.method = lb_method


class AbstractLbMethod(abc.ABC):
    """Abstract base class for all LBM methods."""

    def __init__(self, stencil: LBStencil):
        self._stencil = stencil

    @property
    def stencil(self):
        """Discrete set of velocities, represented by :class:`lbmpy.stencils.LBStencil`"""
        return self._stencil

    @property
    def dim(self):
        """The method's spatial dimensionality"""
        return self._stencil.D

    @property
    def pre_collision_pdf_symbols(self):
        """Tuple of symbols representing the pdf values before collision"""
        return sp.symbols(f"f_:{self._stencil.Q}")

    @property
    def post_collision_pdf_symbols(self):
        """Tuple of symbols representing the pdf values after collision"""
        return sp.symbols(f"d_:{self._stencil.Q}")

    @property
    @abc.abstractmethod
    def relaxation_rates(self):
        """Tuple containing the relaxation rates of each moment"""

    @property
    def relaxation_matrix(self):
        """Returns a qxq diagonal matrix which contains the relaxation rate for each moment on the diagonal"""
        d = sp.zeros(len(self.relaxation_rates))
        for i, w in enumerate(self.relaxation_rates):
            # note that 0.0 is converted to sp.Zero here. It is not possible to prevent this.
            d[i, i] = w
        return d

    @property
    def symbolic_relaxation_matrix(self):
        """Returns a qxq diagonal matrix which contains the relaxation rate for each moment on the diagonal.
           In contrast to the normal relaxation matrix all numeric values are replaced by sympy symbols"""
        _, d = self._generate_symbolic_relaxation_matrix()
        return d

    @property
    def subs_dict_relxation_rate(self):
        """returns a dictonary which maps the replaced numerical relaxation rates to its original numerical value"""
        result = dict()
        for i in range(self._stencil.Q):
            result[self.symbolic_relaxation_matrix[i, i]] = self.relaxation_matrix[i, i]
        return result

    # ------------------------- Abstract Methods & Properties ----------------------------------------------------------

    @property
    @abc.abstractmethod
    def conserved_quantity_computation(self):
        """Returns an instance of class :class:`lbmpy.methods.AbstractConservedQuantityComputation`"""

    @property
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

    # -------------------------------- Helper Functions ----------------------------------------------------------------

    def _generate_symbolic_relaxation_matrix(self, relaxation_rates=None):
        """
        This function replaces the numbers in the relaxation matrix with symbols in this case, and returns also
        the subexpressions, that assign the number to the newly introduced symbol
        """
        rr = relaxation_rates if relaxation_rates is not None else self.relaxation_rates
        unique_relaxation_rates = set()
        subexpressions = {}
        for relaxation_rate in rr:
            relaxation_rate = sp.sympify(relaxation_rate)
            if relaxation_rate not in unique_relaxation_rates:
                # special treatment for zero, sp.Zero would be an integer ..
                if isinstance(relaxation_rate, Zero):
                    relaxation_rate = 0.0
                if not isinstance(relaxation_rate, sp.Symbol):
                    rt_symbol = sp.Symbol(f"rr_{len(subexpressions)}")
                    subexpressions[relaxation_rate] = rt_symbol
            unique_relaxation_rates.add(relaxation_rate)

        new_rr = [subexpressions[sp.sympify(e)] if sp.sympify(e) in subexpressions else e for e in rr]
        substitutions = [Assignment(e[1], e[0]) for e in subexpressions.items()]

        return substitutions, sp.diag(*new_rr)
