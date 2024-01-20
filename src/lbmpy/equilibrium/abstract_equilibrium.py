from abc import ABC, abstractmethod
import sympy as sp
from pystencils.cache import sharedmethodcache

from lbmpy.moments import polynomial_to_exponent_representation


class AbstractEquilibrium(ABC):
    """
    Abstract Base Class for description of equilibrium distribution functions used in lattice
    Boltzmann methods.

    **Equilibrium Representation:**

    This class provides the common interface for describing equilibrium distribution functions,
    which is then used by the various method classes in the derivation of collision equations.
    An equilibrium distribution is defined by either its continuous equation (see :attr:`continuous_equation`)
    or a set of discrete populations
    (see :attr:`discrete_populations` and :class:`lbmpy.equilibrium.GenericDiscreteEquilibrium`).
    The distribution function may be given either in its regular, absolute form; or only as its
    deviation from the rest state, represented by the background distribution (see :attr:`background_distribution`).

    **Computation of Statistical Modes:**

    The major computational task of an equilbrium class is the computation of the distribution's
    statistical modes. For discrete distributions, the subclass :class:`lbmpy.equilibrium.GenericDiscreteEquilibrium`
    provides a generic implementation for their computation. For continuous distributions, computation
    of raw moments, central moments, and cumulants is more complicated, but may also be simplified using special
    tricks.

    As the computation of statistical modes is a time-consuming process, the abstract base class provides caching
    functionality to avoid recomputing quantities that are already known.

    **Instructions to Override:**

    If you wish to model a simple custom discrete distribution, just using the class
    :class:`lbmpy.equilibrium.GenericDiscreteEquilibrium` might already be sufficient.
    If, however, you need to implement more specific functionality, custom properties, 
    a background distribution, etc., or if you wish to model a continuous distribution,
    you will have to set up a custom subclass of :class:`AbstractEquilibrium`.

    A subclass must implement all abstract properties according to their docstrings.
    For computation of statistical modes, a large part of the infrastructure is already given in the abstract base
    class. The public interface for computing e.g. raw moments reduces the computation of polynomial moments to their
    contained monomials (for details on how moments are represented in *lbmpy*, see :mod:`lbmpy.moments`). The values
    of both polynomial and monomial moments, once computed, will be cached per instance of the equilibrium class.
    To take full advantage of the caching functionality, you will have to override only :func:`_monomial_raw_moment`
    and its central moment and cumulant counterparts. These methods will be called only once for each monomial quantity
    when it is required for the first time. Afterward, the cached value will be used.
    """

    def __init__(self, dim=3):
        self._dim = dim

    @property
    def dim(self):
        """This distribution's spatial dimensionality."""
        return self._dim

    #   -------------- Abstract Properties, to be overridden in subclass ----------------------------------------------

    @property
    @abstractmethod
    def deviation_only(self):
        """Whether or not this equilibrium distribution is represented only by its deviation
        from the background distribution."""
        raise NotImplementedError("'deviation_only' must be provided by subclass.")

    @property
    @abstractmethod
    def continuous_equation(self):
        """Returns the continuous equation defining this equilibrium distribution,
        or `None` if no such equation is available."""
        raise NotImplementedError("'continuous_equation' must be provided by subclass.")

    @property
    @abstractmethod
    def discrete_populations(self):
        """Returns the discrete populations of this equilibrium distribution as a tuple, 
        or `None` if none are available."""
        raise NotImplementedError("'discrete_populations' must be provided by subclass.")

    @property
    @abstractmethod
    def background_distribution(self):
        """Returns this equilibrium distribution's background distribution, which is
        the distribution the discrete populations are centered around in the case of
        zero-centered storage. If no background distribution is available, `None` must be 
        returned."""
        raise NotImplementedError("'background_distribution' must be provided by subclass.")

    @property
    @abstractmethod
    def zeroth_order_moment_symbol(self):
        """Returns a symbol referring to the zeroth-order moment of this distribution,
        which is the area under it's curve."""
        raise NotImplementedError("'zeroth_order_moment' must be provided by subclass.")

    @property
    @abstractmethod
    def first_order_moment_symbols(self):
        """Returns a vector of symbols referring to the first-order moment of this distribution,
        which is its mean value."""
        raise NotImplementedError("'first_order_moments' must be provided by subclass.")

    #   -------------- Statistical Modes Interface --------------------------------------------------------------------

    @sharedmethodcache("_moment_cache")
    def moment(self, exponent_tuple_or_polynomial):
        """Returns this equilibrium distribution's moment specified by ``exponent_tuple_or_polynomial``.

        Args:
            exponent_tuple_or_polynomial: Moment specification, see :mod:`lbmpy.moments`.
        """
        monomials = []
        if isinstance(exponent_tuple_or_polynomial, tuple):
            monomials = [(1, exponent_tuple_or_polynomial)]
        else:
            monomials = polynomial_to_exponent_representation(exponent_tuple_or_polynomial, dim=self._dim)

        moment_value = sp.Integer(0)
        for coeff, moment in monomials:
            moment_value += coeff * self._cached_monomial_raw_moment(moment)

        return moment_value.expand()

    def moments(self, exponent_tuples_or_polynomials):
        """Returns a tuple of this equilibrium distribution's moments specified by 'exponent_tuple_or_polynomial'.

        Args:
            exponent_tuples_or_polynomials: Sequence of moment specifications, see :mod:`lbmpy.moments`.
        """
        return tuple(self.moment(m) for m in exponent_tuples_or_polynomials)

    @sharedmethodcache("_central_moment_cache")
    def central_moment(self, exponent_tuple_or_polynomial, frame_of_reference):
        """Returns this equilibrium distribution's central moment specified by
        ``exponent_tuple_or_polynomial``, computed according to the given ``frame_of_reference``.

        Args:
            exponent_tuple_or_polynomial: Moment specification, see :mod:`lbmpy.moments`.
            frame_of_reference: The frame of reference with respect to which the central moment should be computed.
        """
        monomials = []
        if isinstance(exponent_tuple_or_polynomial, tuple):
            monomials = [(1, exponent_tuple_or_polynomial)]
        else:
            monomials = polynomial_to_exponent_representation(exponent_tuple_or_polynomial, dim=self._dim)

        moment_value = sp.Integer(0)
        for coeff, moment in monomials:
            moment_value += coeff * self._cached_monomial_central_moment(moment, frame_of_reference)

        return moment_value.expand()

    def central_moments(self, exponent_tuples_or_polynomials, frame_of_reference):
        """Returns a list this equilibrium distribution's central moments specified by
        ``exponent_tuples_or_polynomials``, computed according to the given ``frame_of_reference``.

        Args:
            exponent_tuples_or_polynomials: Sequence of moment specifications, see :mod:`lbmpy.moments`.
            frame_of_reference: The frame of reference with respect to which the central moment should be computed.
        """
        return tuple(self.central_moment(m, frame_of_reference) for m in exponent_tuples_or_polynomials)

    @sharedmethodcache("_cumulant_cache")
    def cumulant(self, exponent_tuple_or_polynomial, rescale=True):
        """Returns this equilibrium distribution's cumulant specified by ``exponent_tuple_or_polynomial``.

        Args:
            exponent_tuple_or_polynomial: Moment specification, see :mod:`lbmpy.moments`.
            rescale: If ``True``, the cumulant value should be multiplied by the zeroth-order moment.
        """
        monomials = []
        if isinstance(exponent_tuple_or_polynomial, tuple):
            monomials = [(1, exponent_tuple_or_polynomial)]
        else:
            monomials = polynomial_to_exponent_representation(exponent_tuple_or_polynomial, dim=self._dim)

        cumulant_value = sp.Integer(0)
        for coeff, moment in monomials:
            cumulant_value += coeff * self._cached_monomial_cumulant(moment, rescale=rescale)

        return cumulant_value.expand()

    def cumulants(self, exponent_tuples_or_polynomials, rescale=True):
        """Returns a list of this equilibrium distribution's cumulants specified by ``exponent_tuples_or_polynomial``.

        Args:
            exponent_tuples_or_polynomials: Sequence of moment specifications, see :mod:`lbmpy.moments`.
            rescale: If ``True``, the cumulant value should be multiplied by the zeroth-order moment.
        """
        return tuple(self.cumulant(m, rescale) for m in exponent_tuples_or_polynomials)

    #   -------------- Monomial moment computation, to be overridden in subclass --------------------------------------

    @abstractmethod
    def _monomial_raw_moment(self, exponents):
        """See :func:`lbmpy.equilibrium.AbstractEquilibrium.moment`."""
        raise NotImplementedError("'_monomial_raw_moment' must be implemented by a subclass.")

    @abstractmethod
    def _monomial_central_moment(self, exponents, frame_of_reference):
        """See :func:`lbmpy.equilibrium.AbstractEquilibrium.central_moment`."""
        raise NotImplementedError("'_monomial_central_moment' must be implemented by a subclass.")

    @abstractmethod
    def _monomial_cumulant(self, exponents, rescale):
        """See :func:`lbmpy.equilibrium.AbstractEquilibrium.cumulant`."""
        raise NotImplementedError("'_monomial_cumulant' must be implemented by a subclass.")

    #   -------------- Cached monomial moment computation methods -----------------------------------------------------

    @sharedmethodcache("_moment_cache")
    def _cached_monomial_raw_moment(self, exponents):
        return self._monomial_raw_moment(exponents)

    @sharedmethodcache("_central_moment_cache")
    def _cached_monomial_central_moment(self, exponents, frame_of_reference):
        return self._monomial_central_moment(exponents, frame_of_reference)

    @sharedmethodcache("_cumulant_cache")
    def _cached_monomial_cumulant(self, exponents, rescale):
        return self._monomial_cumulant(exponents, rescale)

    #   -------------- HTML Representation ----------------------------------------------------------------------------

    def _repr_html_(self):
        html = f"""
        <table style="border:none; width: 100%">
            <tr>
                <th colspan="2" style="text-align: left">
                    Instance of {self.__class__.__name__}
                </th>
            </tr>
        """

        cont_eq = self.continuous_equation
        if cont_eq is not None:
            html += f"""
                <tr>
                    <td> Continuous Equation: </td>
                    <td style="text-align: center">
                        ${sp.latex(self.continuous_equation)}$
                    </td>
                </tr>
            """
        else:
            pdfs = self.discrete_populations
            if pdfs is not None:
                html += """
                    <tr>
                        <td colspan="2" style="text-align: right;"> Discrete Populations: </td>
                    </tr>
                    """
                for f, eq in zip(sp.symbols(f"f_:{len(pdfs)}"), pdfs):
                    html += f'<tr><td colspan="2" style="text-align: left;"> ${f} = {sp.latex(eq)}$ </td></tr>'

        html += "</table>"

        return html

#   end class AbstractEquilibrium
