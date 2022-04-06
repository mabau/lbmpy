*********************************************
Equilibrium Distributions (lbmpy.equilibrium)
*********************************************

.. automodule:: lbmpy.equilibrium


Abstract Base Class
===================

.. autoclass:: lbmpy.equilibrium.AbstractEquilibrium
    :members:
    :private-members: _monomial_raw_moment, _monomial_central_moment, _monomial_cumulant

Generic Discrete Equilibria
===========================

Use the following class for custom discrete equilibria.

.. autoclass:: lbmpy.equilibrium.GenericDiscreteEquilibrium
    :members:

Maxwellian Equilibria for Hydrodynamics
=======================================

The following classes represent the continuous and the discrete variant of the Maxwellian equilibrium for
hydrodynamics.

.. autoclass:: lbmpy.equilibrium.ContinuousHydrodynamicMaxwellian
    :members:

.. autoclass:: lbmpy.equilibrium.DiscreteHydrodynamicMaxwellian
    :members:
