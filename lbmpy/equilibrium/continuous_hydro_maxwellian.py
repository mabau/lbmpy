import sympy as sp

from .abstract_equilibrium import AbstractEquilibrium

from lbmpy.moments import contained_moments
from lbmpy.maxwellian_equilibrium import continuous_maxwellian_equilibrium
from lbmpy.continuous_distribution_measures import continuous_moment, continuous_cumulant

from pystencils.sympyextensions import remove_higher_order_terms, simplify_by_equality


def default_background_distribution(dim):
    return ContinuousHydrodynamicMaxwellian(dim=dim, compressible=True, deviation_only=False,
                                            rho=sp.Integer(1), delta_rho=0, u=(0,) * dim,
                                            c_s_sq=sp.Rational(1, 3))


class ContinuousHydrodynamicMaxwellian(AbstractEquilibrium):
    r"""
        The standard continuous Maxwellian equilibrium distribution for hydrodynamics.

        This class represents the Maxwellian equilibrium distribution of hydrodynamics in its continuous form
        in :math:`d` dimensions :cite:`lbm_book`:
        
        .. math::

            \Psi \left( \rho, \mathbf{u}, \mathbf{\xi} \right)
             = \rho \left( \frac{1}{2 \pi c_s^2} \right)^{d/2} 
                    \exp \left( \frac{- (\mathbf{\xi} - \mathbf{u})^2 }{2 c_s^2} \right)

        Beyond this classic, 'compressible' form of the equilibrium, an alternative form known as the
        incompressible equilibrium of the LBM can be obtained by setting the flag ``compressible=False``.
        The continuous incompressible equilibrium can be expressed as
        (:cite:`HeIncompressible,GruszczynskiCascadedPhaseFieldModel`):

        .. math::

            \Psi^{\mathrm{incomp}} \left( \rho, \mathbf{u}, \mathbf{\xi} \right)
                = \Psi \left( \rho_0, \mathbf{u}, \mathbf{\xi} \right) 
                  + \Psi \left( \delta\rho, \mathbf{0}, \mathbf{\xi} \right)

        Here, :math:`\rho_0` (typically :math:`\rho_0 = 1`) denotes the background density, and :math:`\delta\rho` is
        the density deviation, such that the total fluid density amounts to :math:`\rho = \rho_0 + \delta\rho`.

        To simplify computations when the zero-centered storage format is used for PDFs, both equilibrium variants can
        also be expressed in a *deviation-only* or *delta-equilibrium* form, which is obtained by subtracting the
        constant background distribution :math:`\Psi (\rho_0, \mathbf{0})`. The delta form expresses the equilibrium
        distribution only by its deviation from the rest state:

        .. math::

            \delta\Psi \left( \rho, \mathbf{u}, \mathbf{\xi} \right)
                &=   \Psi \left( \rho, \mathbf{u}, \mathbf{\xi} \right) 
                    - \Psi \left( \rho_0, \mathbf{0}, \mathbf{\xi} \right) \\
            \delta\Psi^{\mathrm{incomp}} \left( \rho, \mathbf{u}, \mathbf{\xi} \right)
                &=   \Psi^{\mathrm{incomp}} \left( \rho, \mathbf{u}, \mathbf{\xi} \right) 
                    - \Psi \left( \rho_0, \mathbf{0}, \mathbf{\xi} \right)

        Parameters:
            dim: Spatial dimensionality
            compressible: If `False`, the incompressible equilibrium is created
            deviation_only: If `True`, the delta-equilibrium is created
            order: The discretization order in velocity to which computed statistical modes should be truncated
            rho: Symbol or value for the density
            rho_background: Symbol or value for the background density
            delta_rho: Symbol or value for the density deviation
            u: Sequence of symbols for the macroscopic velocity
            v: Sequence of symbols for the particle velocity :math:`\xi`
            c_s_sq: Symbol or value for the squared speed of sound
    """

    def __init__(self, dim=3, compressible=True, deviation_only=False,
                 order=None,
                 rho=sp.Symbol("rho"),
                 rho_background=sp.Integer(1),
                 delta_rho=sp.Symbol("delta_rho"),
                 u=sp.symbols("u_:3"),
                 v=sp.symbols("v_:3"),
                 c_s_sq=sp.Symbol("c_s") ** 2):
        super().__init__(dim=dim)

        self._order = order
        self._compressible = compressible
        self._deviation_only = deviation_only
        self._rho = rho
        self._rho_background = rho_background
        self._delta_rho = delta_rho
        self._u = u[:dim]
        self._v = v[:dim]

        # trick to speed up sympy integration (otherwise it takes multiple minutes, or aborts):
        # use a positive, real symbol to represent c_s_sq -> then replace this symbol afterwards with the real c_s_sq
        # (see maxwellian_equilibrium.py)
        self._c_s_sq = c_s_sq
        self._c_s_sq_helper = sp.Symbol("csq_helper", positive=True, real=True)

        def psi(rho, u):
            return continuous_maxwellian_equilibrium(dim=self._dim,
                                                     rho=rho,
                                                     u=u,
                                                     v=self._v,
                                                     c_s_sq=self._c_s_sq_helper)

        zeroth_moment_arg = self._rho if self._compressible else self._rho_background
        self._base_equation = psi(zeroth_moment_arg, self._u)

        self._corrections = []
        
        if not self._compressible:
            zeroth_order_correction = psi(self._delta_rho, (sp.Integer(0), ) * self._dim)
            self._corrections.append((sp.Integer(1), zeroth_order_correction))

        if self._deviation_only:
            rest_state = psi(self._rho_background, (sp.Integer(0), ) * self._dim)
            self._corrections.append((sp.Integer(-1), rest_state))

    @property
    def order(self):
        return self._order

    @property
    def deviation_only(self):
        return self._deviation_only

    @property
    def compressible(self):
        return self._compressible

    @property
    def density(self):
        return self._rho

    @property
    def background_density(self):
        return self._rho_background

    @property
    def density_deviation(self):
        return self._delta_rho

    @property
    def velocity(self):
        return self._u

    @property
    def continuous_equation(self):
        eq = self._base_equation + sum(f * e for f, e in self._corrections)
        eq = eq.subs(self._c_s_sq_helper, self._c_s_sq)
        return eq

    @property
    def zeroth_order_moment_symbol(self):
        return self._delta_rho if self._deviation_only else self._rho

    @property
    def first_order_moment_symbols(self):
        return self._u

    @property
    def background_distribution(self):
        return ContinuousHydrodynamicMaxwellian(dim=self.dim, compressible=True, deviation_only=False,
                                                order=self._order, rho=self._rho_background,
                                                rho_background=self._rho_background,
                                                delta_rho=0, u=(0,) * self.dim, v=self._v,
                                                c_s_sq=self._c_s_sq)

    @property
    def discrete_populations(self):
        return None

    def central_moment(self, exponent_tuple_or_polynomial, velocity=None):
        if velocity is None:
            velocity = self._u

        return super().central_moment(exponent_tuple_or_polynomial, velocity)

    def central_moments(self, exponent_tuples_or_polynomials, velocity=None):
        if velocity is None:
            velocity = self._u

        return super().central_moments(exponent_tuples_or_polynomials, velocity)

    def cumulant(self, exponent_tuple_or_polynomial, rescale=True):
        if not self._compressible or self._deviation_only:
            raise Exception("Cumulants can only be computed for the compressible, "
                            "non-deviation maxwellian equilibrium!")

        return super().cumulant(exponent_tuple_or_polynomial, rescale=rescale)

    #   ------------------ Overridden Moment Computation ------------------------------------------

    def _monomial_raw_moment(self, exponents):
        moment_value = continuous_moment(self._base_equation, exponents, self._v)

        for coeff, corr in self._corrections:
            moment_value += coeff * continuous_moment(corr, exponents, self._v)

        moment_value = self._correct_order_and_cssq(moment_value)
        moment_value = self._simplify_moment(moment_value)
        return moment_value

    def _monomial_central_moment(self, cm_exponents, velocity):
        #   Setting up the central moment-generating function using SymPy integration
        #   will take unfeasibly long at times
        #   So we compute the central moments by binomial expansion in raw moments

        cm_order = sum(cm_exponents)
        contained_raw_moments = contained_moments(cm_exponents, exclude_original=False)
        moment_value = sp.Integer(0)

        for rm_exponents in contained_raw_moments:
            rm_order = sum(rm_exponents)
            factor = (-1)**(cm_order - rm_order)
            factor *= sp.Mul(*(u**(c - i) * sp.binomial(c, i)
                               for u, c, i in zip(velocity, cm_exponents, rm_exponents)))
            rm_value = self._cached_monomial_raw_moment(rm_exponents)
            moment_value += factor * rm_value

        moment_value = self._correct_order_and_cssq(moment_value)
        moment_value = self._simplify_moment(moment_value)
        return moment_value

    def _monomial_cumulant(self, c_exponents, rescale):
        #   this implementation works only for the compressible, non-deviation equilibrium
        cumulant_value = continuous_cumulant(self._base_equation, c_exponents, self._v)
        cumulant_value = self._correct_order_and_cssq(cumulant_value)
        if rescale:
            cumulant_value = self._rho * cumulant_value
        return cumulant_value

    def _correct_order_and_cssq(self, term):
        term = term.subs(self._c_s_sq_helper, self._c_s_sq)
        term = term.expand()
        if self._order is not None:
            return remove_higher_order_terms(term, order=self._order, symbols=self._u)
        else:
            return term

    def _simplify_moment(self, moment_value):
        if (self.deviation_only or not self.compressible) \
                and isinstance(self.density, sp.Symbol) and isinstance(self.density_deviation, sp.Symbol):
            moment_value = simplify_by_equality(moment_value, self.density,
                                                self.density_deviation, self.background_density)
        return moment_value

    #   ------------------ Utility ----------------------------------------------------------------

    def __repr__(self):
        return f"ContinuousHydrodynamicMaxwellian({self.dim}D, " \
               f"compressible={self.compressible}, deviation_only:{self.deviation_only}" \
               f"order={self.order})"

    def _repr_html_(self):
        def stylized_bool(b):
            return "&#10003;" if b else "&#10007;"

        html = f"""
        <table style="border:none; width: 100%">
            <tr>
                <th colspan="3" style="text-align: left">
                    Continuous Hydrodynamic Maxwellian Equilibrium
                </th>
                <td rowspan="2" style="width: 50%; text-align: center">
                    $f ({sp.latex(self._rho)}, {sp.latex(self._u)}, {sp.latex(self._v)}) 
                        = {sp.latex(self.continuous_equation)}$
                </td>
            </tr>
            <tr>
                <td>Compressible: {stylized_bool(self._compressible)}</td>
                <td>Deviation Only: {stylized_bool(self._deviation_only)}</td>
                <td>Order: {"&#8734;" if self._order is None else self._order}</td>
            </tr>
        </table>
        """

        return html

#   end class ContinuousHydrodynamicMaxwellian
