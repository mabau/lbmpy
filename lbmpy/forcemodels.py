r"""
.. module:: forcemodels
    :synopsis: Collection of forcing terms for hydrodynamic LBM simulations


Get started:
------------

This module offers different models to introduce a body force in the lattice Boltzmann scheme.
If you don't know which model to choose, use :class:`lbmpy.forcemodels.Schiller`.
For incompressible collision models the :class:`lbmpy.forcemodels.Buick` model can be better.


Detailed information:
---------------------

Force models add a term :math:`C_F` to the collision equation:

.. math ::

    f(\pmb{x} + c_q \Delta t, t + \Delta t) - f(\pmb{x},t) = \Omega(f, f^{(eq)})
                                                            + \underbrace{F_q}_{\mbox{forcing term}}

The form of this term depends on the concrete force model: the first moment of this forcing term is equal
to the acceleration :math:`\pmb{a}` for all force models.

.. math ::

    \sum_q \pmb{c}_q F_q = \pmb{a}


The second order moment is different for the forcing models - if it is zero the model is suited for
incompressible flows. For weakly compressible collision operators a force model with a corrected second order moment
should be chosen.

.. math ::

    \sum_q c_{qi} c_{qj} f_q = F_i u_j + F_j u_i  \hspace{1cm} \mbox{for Guo, Luo models}
    
    \sum_q c_{qi} c_{qj} f_q = 0  \hspace{1cm} \mbox{for Simple, Buick}
    
Models with zero second order moment have:

.. math ::
    
    F_q = \frac{w_q}{c_s^2} c_{qi} \; a_i

Models with nonzero second moment have:

.. math ::
    
    F_q = \frac{w_q}{c_s^2} c_{qi} \; a_i + \frac{w_q}{c_s^4} (c_{qi} c_{qj} - c_s^2 \delta_{ij} ) u_j \, a_i


For all force models the computation of the macroscopic velocity has to be adapted (shifted) by adding a term
:math:`S_{macro}` that we call "macroscopic velocity shift"
    
    .. math ::
    
        \pmb{u} = \sum_q \pmb{c}_q f_q + S_{macro}
        
        S_{macro} = \frac{\Delta t}{2} \sum_q F_q


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

import sympy as sp

from lbmpy.relaxationrates import get_bulk_relaxation_rate, get_shear_relaxation_rate


class Simple:
    r"""
    A simple force model which introduces the following additional force term in the
    collision process :math:`\frac{w_q}{c_s^2} c_{qi} \; a_i` (often: force = rho * acceleration)
    Should only be used with constant forces!
    Shifts the macroscopic velocity by F/2, but does not change the equilibrium velocity.
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, lb_method, **kwargs):
        assert len(self._force) == lb_method.dim

        def scalar_product(a, b):
            return sum(a_i * b_i for a_i, b_i in zip(a, b))

        return [3 * w_i * scalar_product(self._force, direction)
                for direction, w_i in zip(lb_method.stencil, lb_method.weights)]

    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)

    def macroscopic_momentum_density_shift(self, density):
        return default_momentum_density_shift(self._force)


class Luo:
    r"""Force model by Luo :cite:`luo1993lattice`.

    Shifts the macroscopic velocity by F/2, but does not change the equilibrium velocity.
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, lb_method):
        u = sp.Matrix(lb_method.first_order_equilibrium_moment_symbols)
        force = sp.Matrix(self._force)

        result = []
        for direction, w_i in zip(lb_method.stencil, lb_method.weights):
            direction = sp.Matrix(direction)
            result.append(3 * w_i * force.dot(direction - u + 3 * direction * direction.dot(u)))
        return result

    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)

    def macroscopic_momentum_density_shift(self, density):
        return default_momentum_density_shift(self._force)


class Guo:
    r"""
    Force model by Guo  :cite:`guo2002discrete`
    Adapts the calculation of the macroscopic velocity as well as the equilibrium velocity (both shifted by F/2)!
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, lb_method):
        luo = Luo(self._force)

        shear_relaxation_rate = get_shear_relaxation_rate(lb_method)
        assert len(set(lb_method.relaxation_rates)) == 1, "Guo only works for SRT, use Schiller instead"
        correction_factor = (1 - sp.Rational(1, 2) * shear_relaxation_rate)
        return [correction_factor * t for t in luo(lb_method)]

    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)

    def macroscopic_momentum_density_shift(self, density):
        return default_momentum_density_shift(self._force)

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)


class Schiller:
    r"""
    Force model by Schiller  :cite:`schiller2008thermal`, equation 4.67
    Equivalent to Guo but not restricted to SRT.
    """
    def __init__(self, force):
        self._force = force

    def __call__(self, lb_method):
        u = sp.Matrix(lb_method.first_order_equilibrium_moment_symbols)
        force = sp.Matrix(self._force)
        
        uf = u.dot(force) * sp.eye(len(force))
        omega = get_shear_relaxation_rate(lb_method)
        omega_bulk = get_bulk_relaxation_rate(lb_method)
        G = (u * force.transpose() + force * u.transpose() - uf * sp.Rational(2, lb_method.dim)) * sp.Rational(1, 2) * \
            (2 - omega) + uf * sp.Rational(1, lb_method.dim) * (2 - omega_bulk)

        result = []
        for direction, w_i in zip(lb_method.stencil, lb_method.weights):
            direction = sp.Matrix(direction)
            tr = sp.trace(G * (direction * direction.transpose() - sp.Rational(1, 3) * sp.eye(len(force))))
            result.append(3 * w_i * (force.dot(direction) + sp.Rational(3, 2) * tr))
        return result
    
    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)

    def macroscopic_momentum_density_shift(self, density):
        return default_momentum_density_shift(self._force)


class Buick:
    r"""
    This force model :cite:`buick2000gravity` has a force term with zero second moment. 
    It is suited for incompressible lattice models.
    """

    def __init__(self, force):
        self._force = force

    def __call__(self, lb_method, **kwargs):
        simple = Simple(self._force)

        shear_relaxation_rate = get_shear_relaxation_rate(lb_method)
        assert len(set(lb_method.relaxation_rates)) == 1, "Buick only works for SRT"
        correction_factor = (1 - sp.Rational(1, 2) * shear_relaxation_rate)
        return [correction_factor * t for t in simple(lb_method)]

    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)

    def macroscopic_momentum_density_shift(self, density):
        return default_momentum_density_shift(self._force)

    def equilibrium_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)


class EDM:
    r"""Exact differencing force model"""

    def __init__(self, force):
        self._force = force

    def __call__(self, lb_method, **kwargs):
        cqc = lb_method.conserved_quantity_computation
        rho = cqc.zeroth_order_moment_symbol if cqc.compressible else 1
        u = cqc.first_order_moment_symbols

        shifted_u = (u_i + f_i / rho for u_i, f_i in zip(u, self._force))
        eq_terms = lb_method.get_equilibrium_terms()
        shifted_eq = eq_terms.subs({u_i: su_i for u_i, su_i in zip(u, shifted_u)})
        return shifted_eq - eq_terms

    def macroscopic_velocity_shift(self, density):
        return default_velocity_shift(density, self._force)

    def macroscopic_momentum_density_shift(self, density):
        return default_momentum_density_shift(self._force)


# --------------------------------  Helper functions  ------------------------------------------------------------------


def default_velocity_shift(density, force):
    return [f_i / (2 * density) for f_i in force]


def default_momentum_density_shift(force):
    return [f_i / 2 for f_i in force]
