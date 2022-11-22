import sympy as sp

from .abstract_equilibrium import AbstractEquilibrium

from lbmpy.moments import discrete_moment, moment_matrix
from lbmpy.cumulants import discrete_cumulant


def discrete_equilibrium_from_matching_moments(stencil, moment_constraints,
                                               zeroth_order_moment_symbol,
                                               first_order_moment_symbols,
                                               deviation_only=False):
    assert len(moment_constraints) == stencil.Q
    moments = tuple(moment_constraints.keys())
    mm = moment_matrix(moments, stencil)
    try:
        pdfs = mm.inv() * sp.Matrix(list(moment_constraints.values()))
        pdfs = pdfs.expand()
        return GenericDiscreteEquilibrium(stencil, pdfs, zeroth_order_moment_symbol,
                                          first_order_moment_symbols, deviation_only=deviation_only)
    except sp.matrices.inverse.NonInvertibleMatrixError as e:
        raise ValueError("Could not construct equilibrium from given moment constraints.") from e


class GenericDiscreteEquilibrium(AbstractEquilibrium):
    """
        Class for encapsulating arbitrary discrete equilibria, given by their equilibrium populations.

        This class takes both a stencil and a sequence of populations modelling a discrete distribution function
        and provides basic functionality for computing and caching that distribution's statistical modes.

        Parameters:
            stencil: Discrete velocity set, see :class:`lbmpy.stencils.LBStencil`.
            equilibrium_pdfs: List of q populations, describing the particle distribution on the discrete velocity
                              set given by the stencil.
            zeroth_order_moment_symbol: Symbol corresponding to the distribution's zeroth-order moment, the area under
                                        it's curve (see :attr:`zeroth_order_moment_symbol`).
            first_order_moment_symbols: Sequence of symbols corresponding to the distribution's first-order moment, the
                                        vector of its mean values (see :attr:`first_order_moment_symbols`).
            deviation_only: Set to `True` if the given populations model only the deviation from a rest state, to be
                            used in junction with the zero-centered storage format.
    """

    def __init__(self, stencil, equilibrium_pdfs,
                 zeroth_order_moment_symbol,
                 first_order_moment_symbols,
                 deviation_only=False):
        super().__init__(dim=stencil.D)

        if len(equilibrium_pdfs) != stencil.Q:
            raise ValueError(f"Wrong number of PDFs."
                             f"On the {stencil} stencil, exactly {stencil.Q} populations must be passed!")

        self._stencil = stencil
        self._pdfs = tuple(equilibrium_pdfs)
        self._zeroth_order_moment_symbol = zeroth_order_moment_symbol
        self._first_order_moment_symbols = first_order_moment_symbols
        self._deviation_only = deviation_only

    @property
    def stencil(self):
        return self._stencil

    @property
    def deviation_only(self):
        return self._deviation_only

    @property
    def continuous_equation(self):
        """Always returns `None`."""
        return None

    @property
    def discrete_populations(self):
        return self._pdfs

    @property
    def background_distribution(self):
        """Always returns `None`. To specify a background distribution, override this class."""
        return None

    @property
    def zeroth_order_moment_symbol(self):
        return self._zeroth_order_moment_symbol

    @property
    def first_order_moment_symbols(self):
        return self._first_order_moment_symbols

    #   Moment Computation

    def _monomial_raw_moment(self, exponents):
        return discrete_moment(self._pdfs, exponents, self._stencil)

    def _monomial_central_moment(self, exponents, frame_of_reference):
        return discrete_moment(self._pdfs, exponents, self._stencil, shift_velocity=frame_of_reference)

    def _monomial_cumulant(self, exponents, rescale):
        value = discrete_cumulant(self._pdfs, exponents, self._stencil)
        if rescale:
            value = self.zeroth_order_moment_symbol * value
        return value
