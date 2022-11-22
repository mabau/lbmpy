r"""
.. module:: forcemodels
    :synopsis: Collection of forcing terms for hydrodynamic LBM simulations


Get started:
------------

This module offers different models to introduce a body force in the lattice Boltzmann scheme.
If you don't know which model to choose, use :class:`lbmpy.forcemodels.Guo`.


Detailed information:
---------------------

Force models add a term :math:`C_F` to the collision equation:

.. math ::

    f(\mathbf{x} + c_q \Delta t, t + \Delta t) - f(\mathbf{x},t) = \Omega(f, f^{(\mathrm{eq})})
                                                            + \underbrace{F_q}_{\mbox{forcing term}}

The form of this term depends on the concrete force model: the first moment of this forcing term is equal
to the acceleration :math:`\mathbf{a}` for all force models. Here :math:`\mathbf{F}` is the D dimensional force vector,
which defines the force for each spatial dircetion.

.. math ::

    \sum_q \mathbf{c}_q \mathbf{F} = \mathbf{a}


The second order moment is different for the forcing models - if it is zero the model is suited for
incompressible flows. For weakly compressible collision operators a force model with a corrected second order moment
should be chosen.

.. math ::

    \sum_q c_{qi} c_{qj} f_q &= F_i u_j + F_j u_i  &\qquad \mbox{for Guo, Luo models}
    
    \sum_q c_{qi} c_{qj} f_q &= 0  &\qquad \mbox{for Simple, Buick}
    
Models with zero second order moment have:

.. math ::
    
    F_q = \frac{w_q}{c_s^2} c_{qi} \; a_i

Models with nonzero second moment have:

.. math ::
    
    F_q = \frac{w_q}{c_s^2} c_{qi} \; a_i + \frac{w_q}{c_s^4} (c_{qi} c_{qj} - c_s^2 \delta_{ij} ) u_j \, a_i


For all force models the computation of the macroscopic velocity has to be adapted (shifted) by adding a term
:math:`S_{macro}` that we call "macroscopic velocity shift"
    
    .. math ::
    
        \mathbf{u} &= \sum_q \mathbf{c}_q f_q + S_{\mathrm{macro}}
        
        S_{\mathrm{macro}} &= \frac{\Delta t}{2 \cdot \rho} \sum_q F_q


Some models also shift the velocity entering the equilibrium distribution.

Comparison
----------
 
Force models can be distinguished by 2 options:

**Option 1**:
    :math:`C_F = 1` and equilibrium velocity is not shifted, or :math:`C_F=(1 - \frac{\omega}{2})`
    and equilibrum is shifted.
 
**Option 2**: 
    second velocity moment is zero or :math:`F_i u_j + F_j u_i`


=====================  ====================  =================
 Option2 \\ Option1    no equilibrium shift  equilibrium shift
=====================  ====================  =================
second moment zero      :class:`Simple`      :class:`Buick`
second moment nonzero   :class:`Luo`         :class:`Guo`         
=====================  ====================  =================
  
"""
from warnings import warn
import abc
import sympy as sp

from pystencils.sympyextensions import scalar_product

from lbmpy.maxwellian_equilibrium import (
    discrete_maxwellian_equilibrium, get_equilibrium_values_of_maxwell_boltzmann_function
)
from lbmpy.moments import (
    MOMENT_SYMBOLS, exponent_tuple_sort_key, exponents_to_polynomial_representations,
    extract_monomials, moment_sort_key, moment_matrix,
    monomial_to_polynomial_transformation_matrix, set_up_shift_matrix)

FORCE_SYMBOLS = sp.symbols("F_x, F_y, F_z")


class AbstractForceModel(abc.ABC):
    r"""
    Abstract base class for all force models. All force models have to implement the __call__, which should return a
    q dimensional vector added to the PDFs in the population space. If an MRT method is used, it is also possible
    to apply the force directly in the moment space. This is done by additionally providing the function
    moment_space_forcing. The MRT method will check if it is available and apply the force directly in the moment
    space. For MRT methods in the central moment space the central_moment_space_forcing function can be provided,
    which shifts the force vector to the central moment space. Applying the force in the collision space has the
    advantage of saving FLOPs. Furthermore, it is sometimes easier to apply the correct force vector in the
    collision space because often, the relaxation matrix has to be taken into account.

    Args:
        force: force vector of size dim which contains the force for each spatial dimension.
    """

    def __init__(self, force):
        self._force = force

        # All force models work internaly with a pure symbolic list of the forcing vector.
        # Each entry of the original force vector which is not a symbol is mapped to a symbol and a subs dict is
        # created. The subs dict should be used inside the LB method for the creation of the collision rule.
        self._symbolic_force = [x if isinstance(x, sp.Symbol) else y for x, y in zip(force, FORCE_SYMBOLS)]
        self._subs_dict_force = {x: y for (x, y) in zip(self._symbolic_force, force) if x != y}

        # The following booleans should indicate if a force model is has the function moment_space_forcing which
        # transfers the forcing terms to the moment space or central_moment_space_forcing which transfers them to the
        # central moment space
        self.has_moment_space_forcing = hasattr(self, "moment_space_forcing")
        self.has_central_moment_space_forcing = hasattr(self, "central_moment_space_forcing")
        self.has_symmetric_central_moment_forcing = hasattr(self, "symmetric_central_moment_forcing")

    def __call__(self, lb_method):
        r"""
        This function returns a vector of size q which is added to the PDFs in the PDF space. It has to be implemented
        by all forcing models and returns a sympy Matrix containing the q dimensional force vector.

        Args:
            lb_method: LB method, see lbmpy.creationfunctions.create_lb_method
        """
        raise NotImplementedError("Force model class has to overwrite __call__")

    def macroscopic_velocity_shift(self, density):
        r"""
        macroscopic velocity shift by :math:`\frac{\Delta t}{2 \cdot \rho}`

        Args:
            density: Density symbol which is needed for the shift
        """
        return default_velocity_shift(density, self.symbolic_force_vector)

    def macroscopic_momentum_density_shift(self, *_):
        r"""
        macroscopic momentum density shift by :math:`\frac{\Delta t}{2}`
        """
        return default_momentum_density_shift(self.symbolic_force_vector)

    def equilibrium_velocity_shift(self, density):
        r"""
        Some models also shift the velocity entering the equilibrium distribution. By default the shift is zero

        Args:
            density: Density symbol which is needed for the shift
        """
        return [0] * len(self.symbolic_force_vector)

    @property
    def symbolic_force_vector(self):
        return self._symbolic_force

    @property
    def subs_dict_force(self):
        return self._subs_dict_force


class Simple(AbstractForceModel):
    r"""
    A simple force model which introduces the following additional force term in the
    collision process :math:`\frac{w_q}{c_s^2} \mathbf{c_{q}} \cdot \mathbf{F}` (often: force = rho * acceleration
    where rho is the zeroth moment to be consistent with the above definition)
    Should only be used with constant forces!
    Shifts the macroscopic velocity by :math:`\frac{\mathbf{F}}{2}`, but does not change the equilibrium velocity.
    """

    def __call__(self, lb_method):
        force = self.symbolic_force_vector
        assert len(force) == lb_method.dim, "Force vectore must match with the dimensions of the lb method"
        cs_sq = sp.Rational(1, 3)  # squared speed of sound

        result = [(w_i / cs_sq) * scalar_product(force, direction)
                  for direction, w_i in zip(lb_method.stencil, lb_method.weights)]

        return sp.Matrix(result)

    def moment_space_forcing(self, lb_method):
        return (lb_method.moment_matrix * self(lb_method)).expand()

    def central_moment_space_forcing(self, lb_method):
        moments = (lb_method.moment_matrix * sp.Matrix(self(lb_method))).expand()
        return lb_method.shift_matrix * moments

    def symmetric_central_moment_forcing(self, lb_method, central_moments):
        u = lb_method.first_order_equilibrium_moment_symbols
        cm_matrix = moment_matrix(central_moments, lb_method.stencil, shift_velocity=u)
        before = sp.Matrix([0] * lb_method.stencil.Q)
        after = cm_matrix @ sp.Matrix(self(lb_method))
        return before, after


class Luo(AbstractForceModel):
    r"""Force model by Luo :cite:`luo1993lattice`.

    Shifts the macroscopic velocity by :math:`\frac{\mathbf{F}}{2}`, but does not change the equilibrium velocity.
    """

    def __call__(self, lb_method):
        u = sp.Matrix(lb_method.first_order_equilibrium_moment_symbols)
        force = sp.Matrix(self.symbolic_force_vector)

        cs_sq = sp.Rational(1, 3)  # squared speed of sound

        result = []
        for direction, w_i in zip(lb_method.stencil, lb_method.weights):
            direction = sp.Matrix(direction)
            first_summand = (direction - u) / cs_sq
            second_summand = (direction * direction.dot(u)) / cs_sq ** 2

            fq = w_i * force.dot(first_summand + second_summand)

            result.append(fq.simplify())
        return sp.Matrix(result)

    def moment_space_forcing(self, lb_method):
        return (lb_method.moment_matrix * self(lb_method)).expand()

    def central_moment_space_forcing(self, lb_method):
        moments = lb_method.moment_matrix * self(lb_method)
        return (lb_method.shift_matrix * moments).expand()

    def symmetric_central_moment_forcing(self, lb_method, central_moments):
        u = lb_method.first_order_equilibrium_moment_symbols
        cm_matrix = moment_matrix(central_moments, lb_method.stencil, shift_velocity=u)
        before = sp.Matrix([0] * lb_method.stencil.Q)
        after = (cm_matrix @ sp.Matrix(self(lb_method))).expand()
        return before, after


class Guo(AbstractForceModel):
    r"""
    Force model by Guo  :cite:`guo2002discrete`, generalized to MRT,
    which makes it equivalent to :cite:`schiller2008thermal`, equation 4.67
    Adapts the calculation of the macroscopic velocity as well as the equilibrium velocity
    (both shifted by :math:`\frac{\mathbf{F}}{2}`)!
    """

    def __call__(self, lb_method):
        if len(set(lb_method.relaxation_rates)) == 1:
            #   It's an SRT method!
            rr = lb_method.symbolic_relaxation_matrix[0]
            force_terms = Luo(self.symbolic_force_vector)(lb_method)
            correction_factor = (1 - rr / 2)
            result = correction_factor * force_terms
        else:
            force_terms = self.moment_space_forcing(lb_method)
            result = (lb_method.moment_matrix.inv() * force_terms).expand()
        return result

    def moment_space_forcing(self, lb_method):
        luo = Luo(self.symbolic_force_vector)
        q = len(lb_method.stencil)
        correction_factor = sp.eye(q) - sp.Rational(1, 2) * lb_method.symbolic_relaxation_matrix
        moments = correction_factor * (lb_method.moment_matrix * sp.Matrix(luo(lb_method))).expand()
        return moments

    def central_moment_space_forcing(self, lb_method):
        luo = Luo(self.symbolic_force_vector)
        q = len(lb_method.stencil)
        correction_factor = sp.eye(q) - sp.Rational(1, 2) * lb_method.symbolic_relaxation_matrix
        moments = (lb_method.moment_matrix * sp.Matrix(luo(lb_method)))
        central_moments = correction_factor * (lb_method.shift_matrix * moments).expand()

        return central_moments

    def symmetric_central_moment_forcing(self, lb_method, central_moments):
        luo = Luo(self.symbolic_force_vector)
        _, force_cms = luo.symmetric_central_moment_forcing(lb_method, central_moments)
        force_cms = sp.Rational(1, 2) * force_cms
        return force_cms, force_cms

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self.symbolic_force_vector)


class He(AbstractForceModel):
    r"""
    Force model by He  :cite:`HeForce`
    Adapts the calculation of the macroscopic velocity as well as the equilibrium velocity
    (both shifted by :math:`\frac{\mathbf{F}}{2}`)!

    Force moments are derived from the continuous maxwellian equilibrium. From the
    moment integrals of the continuous force term

    .. math::

        F (\mathbf{c}) 
        = \frac{1}{\rho c_s^2} 
          \mathbf{F} \cdot ( \mathbf{c} - \mathbf{u} ) 
          f^{\mathrm{eq}} (\mathbf{c})

    the following analytical expresson for the monomial raw moments of the force is found:

    .. math::

        m_{\alpha\beta\gamma}^{F, \mathrm{He}} 
            = \frac{1}{\rho c_s^2} \left( 
                F_x m^{\mathrm{eq}}_{\alpha+1,\beta,\gamma} 
                + F_y m^{\mathrm{eq}}_{\alpha,\beta+1,\gamma} 
                + F_z m^{\mathrm{eq}}_{\alpha,\beta,\gamma+1} 
                - m^{eq}_{\alpha\beta\gamma} ( \mathbf{F} \cdot \mathbf{u} )
            \right)
    """

    def __init__(self, force):
        super(He, self).__init__(force)

    def forcing_terms(self, lb_method):
        u = sp.Matrix(lb_method.first_order_equilibrium_moment_symbols)
        force = sp.Matrix(self.symbolic_force_vector)

        cs_sq = sp.Rational(1, 3)  # squared speed of sound
        # eq. 6.31 in the LB book by Krüger et al. shows that the equilibrium terms are devided by rho.
        # This is done here with subs({rho: 1}) to be consistent with compressible and incompressible force models
        cqc = lb_method.conserved_quantity_computation
        eq_terms = discrete_maxwellian_equilibrium(lb_method.stencil, rho=sp.Integer(1),
                                                   u=cqc.velocity_symbols, c_s_sq=sp.Rational(1, 3))

        result = []
        for direction, eq in zip(lb_method.stencil, eq_terms):
            direction = sp.Matrix(direction)
            eu_dot_f = (direction - u).dot(force)
            result.append(eq * eu_dot_f / cs_sq)

        return sp.Matrix(result)

    def continuous_force_raw_moments(self, lb_method, moments=None):
        rho = lb_method.zeroth_order_equilibrium_moment_symbol
        u = lb_method.first_order_equilibrium_moment_symbols
        dim = lb_method.dim
        c_s_sq = sp.Rational(1, 3)
        force = sp.Matrix(self.symbolic_force_vector)

        moment_polynomials = lb_method.moments if moments is None else moments
        moment_exponents = sorted(extract_monomials(moment_polynomials), key=exponent_tuple_sort_key)
        moment_monomials = exponents_to_polynomial_representations(moment_exponents)
        extended_monomials = set()
        for m in moment_monomials:
            extended_monomials |= {m} | {m * x for x in MOMENT_SYMBOLS[:dim]}

        extended_monomials = sorted(extended_monomials, key=moment_sort_key)
        moment_eq_values = get_equilibrium_values_of_maxwell_boltzmann_function(extended_monomials, dim, rho=rho,
                                                                                u=u, c_s_sq=c_s_sq)
        moment_to_eq_dict = {m: v for m, v in zip(extended_monomials, moment_eq_values)}

        monomial_force_moments = []
        for moment in moment_monomials:
            m_base = moment_to_eq_dict[moment]
            m_shifted = sp.Matrix([moment_to_eq_dict[moment * x] for x in MOMENT_SYMBOLS[:dim]])
            m_force = (c_s_sq * rho)**(-1) * (force.dot(m_shifted) - m_base * force.dot(u))
            monomial_force_moments.append(m_force.expand())

        mono_to_poly_matrix = monomial_to_polynomial_transformation_matrix(moment_exponents, moment_polynomials)
        polynomial_force_moments = mono_to_poly_matrix * sp.Matrix(monomial_force_moments)
        return polynomial_force_moments

    def continuous_force_central_moments(self, lb_method, moments=None):
        if moments is None:
            moments = lb_method.moments
        raw_moments = self.continuous_force_raw_moments(lb_method, moments=moments)
        u = lb_method.first_order_equilibrium_moment_symbols
        shift_matrix = set_up_shift_matrix(moments, lb_method.stencil, velocity_symbols=u)
        return (shift_matrix * raw_moments).expand()

    def __call__(self, lb_method):
        if len(set(lb_method.relaxation_rates)) == 1:
            #   It's an SRT method!
            rr = lb_method.symbolic_relaxation_matrix[0]
            force_terms = self.forcing_terms(lb_method)
            correction_factor = (1 - rr / 2)
            result = correction_factor * force_terms
        else:
            force_terms = self.moment_space_forcing(lb_method)
            result = (lb_method.moment_matrix.inv() * force_terms).expand()
        return result

    def moment_space_forcing(self, lb_method):
        correction_factor = sp.eye(len(lb_method.stencil)) - sp.Rational(1, 2) * lb_method.symbolic_relaxation_matrix
        moments = self.continuous_force_raw_moments(lb_method)
        moments = (correction_factor * moments).expand()
        return moments

    def central_moment_space_forcing(self, lb_method):
        correction_factor = sp.eye(len(lb_method.stencil)) - sp.Rational(1, 2) * lb_method.symbolic_relaxation_matrix
        central_moments = self.continuous_force_central_moments(lb_method)
        central_moments = (correction_factor * central_moments).expand()
        return central_moments

    def symmetric_central_moment_forcing(self, lb_method, central_moments):
        central_moments = exponents_to_polynomial_representations(central_moments)
        force_cms = sp.Rational(1, 2) * self.continuous_force_central_moments(lb_method, moments=central_moments)
        return force_cms, force_cms

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self.symbolic_force_vector)


class Schiller(Guo):
    r"""
    Force model by Schiller  :cite:`schiller2008thermal`, equation 4.67
    Equivalent to the generalized Guo model.
    """

    def __init__(self, force):
        warn("The Schiller force model is deprecated, please use the Guo model, which is equivalent",
             DeprecationWarning)
        super(Schiller, self).__init__(force)


class Buick(AbstractForceModel):
    r"""
    This force model :cite:`buick2000gravity` has a force term with zero second moment. 
    It is suited for incompressible lattice models. However it should be used with care because such a LB body form
    model is only consistent when applied to the solution of steady - state hydrodynamic problems. More information
    on an analysis of the Buick force model can be found in :cite:`silva2010` and in :cite:`silva2020`
    """

    def __call__(self, lb_method, **kwargs):
        if len(set(lb_method.relaxation_rates)) == 1:
            #   It's an SRT method!
            rr = lb_method.symbolic_relaxation_matrix[0]
            force_terms = Simple(self.symbolic_force_vector)(lb_method)
            correction_factor = (1 - rr / 2)
            result = correction_factor * force_terms
        else:
            force_terms = self.moment_space_forcing(lb_method)
            result = (lb_method.moment_matrix.inv() * force_terms).expand()
        return result

    def moment_space_forcing(self, lb_method):
        simple = Simple(self.symbolic_force_vector)
        q = len(lb_method.stencil)
        correction_factor = sp.eye(q) - sp.Rational(1, 2) * lb_method.symbolic_relaxation_matrix
        moments = correction_factor * (lb_method.moment_matrix * sp.Matrix(simple(lb_method)))
        return moments.expand()

    def central_moment_space_forcing(self, lb_method):
        simple = Simple(self.symbolic_force_vector)
        q = len(lb_method.stencil)
        correction_factor = sp.eye(q) - sp.Rational(1, 2) * lb_method.symbolic_relaxation_matrix
        moments = (lb_method.moment_matrix * sp.Matrix(simple(lb_method)))
        central_moments = correction_factor * (lb_method.shift_matrix * moments)

        return central_moments.expand()

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self.symbolic_force_vector)


class EDM(AbstractForceModel):
    r"""Exact differencing force model as shown in :cite:`lbm_book` in eq. 6.32"""

    def __call__(self, lb_method):
        cqc = lb_method.conserved_quantity_computation
        reference_density = cqc.density_symbol if cqc.compressible else 1
        rho = cqc.density_symbol
        delta_rho = cqc.density_deviation_symbol
        rho_0 = cqc.background_density
        u = cqc.velocity_symbols

        equilibrium_terms = lb_method.get_equilibrium_terms()
        equilibrium_terms = equilibrium_terms.subs({delta_rho: rho - rho_0})

        shifted_u = (u_i + f_i / reference_density for u_i, f_i in zip(u, self._force))
        shifted_eq = equilibrium_terms.subs({u_i: su_i for u_i, su_i in zip(u, shifted_u)})
        return shifted_eq - equilibrium_terms

    def moment_space_forcing(self, lb_method):
        moments = lb_method.moment_matrix * self(lb_method)
        return moments.expand()

    def central_moment_space_forcing(self, lb_method):
        moments = lb_method.moment_matrix * self(lb_method)
        central_moments = lb_method.shift_matrix * moments.expand()
        return central_moments.expand()


class ShanChen(AbstractForceModel):
    r"""Shan and Chen force model. The implementation is done according to :cite:`silva2020`.
        For reference compare table 1 which is the Shan and Chen model for an SRT collision operator. These terms are
        transfered to the moment space and then all representations for the different collision operators are derived
        from that.
    """

    def forcing_terms(self, lb_method):
        q = len(lb_method.stencil)
        cqc = lb_method.conserved_quantity_computation
        rho = cqc.density_symbol if cqc.compressible else 1
        u = cqc.velocity_symbols

        F = sp.Matrix(self.symbolic_force_vector)
        uf = sp.Matrix(u).dot(F)
        F2 = sp.Matrix(F).dot(sp.Matrix(F))
        Fq = sp.zeros(q, 1)
        uq = sp.zeros(q, 1)
        for i, cq in enumerate(lb_method.stencil):
            Fq[i] = sp.Matrix(cq).dot(sp.Matrix(F))
            uq[i] = sp.Matrix(cq).dot(u)

        linear_term = sp.zeros(q, 1)
        non_linear_term = sp.zeros(q, 1)
        for i, w_i in enumerate(lb_method.weights):
            linear_term[i] = w_i * (Fq[i] + 3 * uq[i] * Fq[i] - uf)
            non_linear_term[i] = ((w_i / (2 * rho)) * (3 * Fq[i] ** 2 - F2))

        return linear_term, non_linear_term

    def __call__(self, lb_method):
        force_terms = self.moment_space_forcing(lb_method)
        result = lb_method.moment_matrix.inv() * force_terms
        return result.expand()

    def moment_space_forcing(self, lb_method):
        linear_term, non_linear_term = self.forcing_terms(lb_method)
        q = len(lb_method.stencil)

        rel = lb_method.symbolic_relaxation_matrix
        cs_sq = sp.Rational(1, 3)

        correction_factor = 1 / cs_sq * (sp.eye(q) - sp.Rational(1, 2) * rel)
        M = lb_method.moment_matrix
        moments = correction_factor * (M * linear_term) + correction_factor ** 2 * (M * non_linear_term)
        return moments.expand()

    def central_moment_space_forcing(self, lb_method):
        linear_term, non_linear_term = self.forcing_terms(lb_method)
        q = len(lb_method.stencil)

        rel = lb_method.symbolic_relaxation_matrix
        cs_sq = sp.Rational(1, 3)

        correction_factor = 1 / cs_sq * (sp.eye(q) - sp.Rational(1, 2) * rel)
        M = lb_method.moment_matrix
        N = lb_method.shift_matrix
        moments_linear_term = (M * linear_term)
        moments_non_linear_term = (M * non_linear_term)
        central_moments_linear_term = correction_factor * (N * moments_linear_term)
        central_moments_non_linear_term = correction_factor ** 2 * (N * moments_non_linear_term)
        central_moments = central_moments_linear_term + central_moments_non_linear_term
        return central_moments.expand()

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self.symbolic_force_vector)


# --------------------------------  Helper functions  ------------------------------------------------------------------


def default_velocity_shift(density, force):
    return [f_i / (2 * density) for f_i in force]


def default_momentum_density_shift(force):
    return [f_i / 2 for f_i in force]
