import sympy as sp

from pystencils.sympyextensions import simplify_by_equality

from lbmpy.maxwellian_equilibrium import discrete_maxwellian_equilibrium

from .generic_discrete_equilibrium import GenericDiscreteEquilibrium


class DiscreteHydrodynamicMaxwellian(GenericDiscreteEquilibrium):
    r"""
        The textbook discretization of the Maxwellian equilibrium distribution of hydrodynamics.

        This class represents the default discretization of the Maxwellian in velocity space,
        computed from the distribution's expansion in Hermite polynomials (cf. :cite:`lbm_book`).
        In :math:`d` dimensions, its populations :math:`f_i` on a given stencil 
        :math:`(\mathbf{c}_i)_{i=0,\dots,q-1}` are given by

        .. math::

            f_i (\rho, \mathbf{u}) 
                = w_i \rho \left( 
                    1 + \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2}
                    + \frac{(\mathbf{c}_i \cdot \mathbf{u})^2}{2 c_s^4}
                    - \frac{\mathbf{u} \cdot \mathbf{u}}{2 c_s^2}
                  \right).

        Here :math:`w_i` denote the Hermite integration weights, also called lattice weights.
        The incompressible variant of this distribution :cite:`HeIncompressible` can be written as

        .. math::
            f_i^{\mathrm{incomp}} (\rho, \mathbf{u})
                = w_i \rho + w_i \rho_0 \left(
                    \frac{\mathbf{c}_i \cdot \mathbf{u}}{c_s^2}
                    + \frac{(\mathbf{c}_i \cdot \mathbf{u})^2}{2 c_s^4}
                    - \frac{\mathbf{u} \cdot \mathbf{u}}{2 c_s^2}
                  \right).

        Again, for usage with zero-centered PDF storage, both distributions may be expressed in a delta-form
        by subtracting their values at the background rest state at :math:`\rho = \rho_0`,
        :math:`\mathbf{u} = \mathbf{0}`, which are exactly the lattice weights:

        .. math::
            \delta f_i &= f_i - w_i \\
            \delta f_i^{\mathrm{incomp}} &= f_i^{\mathrm{incomp}} - w_i \\

        Parameters:
            stencil: Discrete velocity set for the discretization, see :class:`lbmpy.stencils.LBStencil`
            compressible: If `False`, the incompressible equilibrium is created
            deviation_only: If `True`, the delta-equilibrium is created
            order: The discretization order in velocity to which computed statistical modes should be truncated
            rho: Symbol or value for the density
            delta_rho: Symbol or value for the density deviation
            u: Sequence of symbols for the macroscopic velocity
            c_s_sq: Symbol or value for the squared speed of sound
    """

    def __init__(self, stencil, compressible=True, deviation_only=False,
                 order=2,
                 rho=sp.Symbol("rho"),
                 delta_rho=sp.Symbol("delta_rho"),
                 u=sp.symbols("u_:3"),
                 c_s_sq=sp.Symbol("c_s") ** 2):
        dim = stencil.D

        if order is None:
            order = 4

        self._order = order
        self._compressible = compressible
        self._deviation_only = deviation_only
        self._rho = rho
        self._rho_background = sp.Integer(1)
        self._delta_rho = delta_rho
        self._u = u[:dim]
        self._c_s_sq = c_s_sq

        pdfs = discrete_maxwellian_equilibrium(stencil, rho=rho, u=u,
                                               order=order, c_s_sq=c_s_sq,
                                               compressible=compressible)
        if deviation_only:
            shift = discrete_maxwellian_equilibrium(stencil, rho=self._rho_background, u=(0,) * dim,
                                                    order=0, c_s_sq=c_s_sq, compressible=False)
            pdfs = tuple(simplify_by_equality(f - s, rho, delta_rho, self._rho_background) for f, s in zip(pdfs, shift))

        zeroth_order_moment = delta_rho if deviation_only else rho
        super().__init__(stencil, pdfs, zeroth_order_moment, u, deviation_only)

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
    def background_distribution(self):
        """Returns the discrete Maxwellian background distribution, which amounts exactly to the
        lattice weights."""
        return DiscreteHydrodynamicMaxwellian(self._stencil, compressible=True, deviation_only=False,
                                              order=self._order, rho=self._rho_background,
                                              delta_rho=0, u=(0,) * self.dim, c_s_sq=self._c_s_sq)

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

    #   ------------------ Utility ----------------------------------------------------------------

    def __repr__(self):
        return f"DiscreteHydrodynamicMaxwellian({self.stencil}, " \
               f"compressible={self.compressible}, deviation_only:{self.deviation_only}" \
               f"order={self.order})"

    def _repr_html_(self):
        def stylized_bool(b):
            return "&#10003;" if b else "&#10007;"

        html = f"""
        <div style="max-height: 150pt; overflow-y: auto;">
        <table style="border:none; width: 100%">
            <tr>
                <th colspan="3" style="text-align: left">
                    Discrete Hydrodynamic Maxwellian Equilibrium
                </th>
                <td>Compressible: {stylized_bool(self._compressible)}</td>
                <td>Deviation Only: {stylized_bool(self._deviation_only)}</td>
                <td>Order: {"&#8734;" if self._order is None else self._order}</td>
            </tr>
        """

        pdfs = self.discrete_populations
        for f, eq in zip(sp.symbols(f"f_:{len(pdfs)}"), pdfs):
            html += f'<tr><td colspan="6" style="text-align: left;"> ${f} = {sp.latex(eq)}$ </td></tr>'
        
        html += "</table></div>"

        return html

# end class DiscreteHydrodynamicMaxwellian
