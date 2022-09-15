*******************************************
Methods and Method Creation (lbmpy.methods)
*******************************************

This module defines the classes defining all types of lattice Boltzmann methods available in *lbmpy*,
together with a set of factory functions used to create their instances. The factory functions are
organized in three levels of abstraction. Objects of the method classes should be created only through
these factory functions, never manually.

Methods in *lbmpy* can be distinguished into three categories:
 - :ref:`methods_rawmomentbased`, including the classical single relaxation-time (SRT, BGK), two relaxation-time (TRT) 
   and multiple relaxation-time (MRT) methods, as well as entropic variants of the TRT method.
 - :ref:`methods_centralmomentbased`, which are multiple relaxation-time methods using relaxation in central moment space.
 - :ref:`methods_cumulantbased`, multiple relaxation-time methods using relaxation in cumulant space.

Abstract LB Method Base Class
=============================

.. autoclass:: lbmpy.methods.LbmCollisionRule
    :members:

.. autoclass:: lbmpy.methods.AbstractLbMethod
    :members:


Conserved Quantity Computation
==============================

The classes of the conserved quantity computation (CQC) submodule define an LB Method's conserved quantities and
the equations to compute them. For hydrodynamic methods, :class:`lbmpy.methods.DensityVelocityComputation` is
the typical choice. For custom methods, however, a custom CQC class might have to be created.

.. autoclass:: lbmpy.methods.AbstractConservedQuantityComputation
    :members:

.. autoclass:: lbmpy.methods.DensityVelocityComputation
    :members:


.. _methods_rawmomentbased:

Raw Moment-based methods
========================

These methods are represented by instances of :class:`lbmpy.methods.momentbased.MomentBasedLbMethod` and will derive
collision equations either in population space (SRT, TRT, entropic TRT), or in raw moment space (MRT variants).

Creation Functions
------------------

The following factory functions create raw moment-based methods using variants of the regular hydrodynamic maxwellian
equilibrium.

.. autofunction:: lbmpy.methods.create_srt

.. autofunction:: lbmpy.methods.create_trt

.. autofunction:: lbmpy.methods.create_trt_with_magic_number

.. autofunction:: lbmpy.methods.create_mrt_orthogonal

.. autofunction:: lbmpy.methods.create_trt_kbc


Class
-----

.. autoclass:: lbmpy.methods.momentbased.MomentBasedLbMethod
    :members:


.. _methods_centralmomentbased:

Central Moment-based methods
============================

These methods are represented by instances of :class:`lbmpy.methods.momentbased.CentralMomentBasedLbMethod` and will derive
collision equations in central moment space.

Creation Functions
------------------

The following factory functions create central moment-based methods using variants of the regular hydrodynamic maxwellian
equilibrium.

.. autofunction:: lbmpy.methods.create_central_moment

Class
-----

.. autoclass:: lbmpy.methods.momentbased.CentralMomentBasedLbMethod
    :members:


.. _methods_cumulantbased:

Cumulant-based methods
======================

These methods are represented by instances of :class:`lbmpy.methods.cumulantbased.CumulantBasedLbMethod` and will derive
collision equations in cumulant space.

Creation Functions
------------------

The following factory functions create cumulant-based methods using the regular continuous hydrodynamic maxwellian equilibrium.

.. autofunction:: lbmpy.methods.create_cumulant

.. autofunction:: lbmpy.methods.create_with_monomial_cumulants

.. autofunction:: lbmpy.methods.create_with_default_polynomial_cumulants


Class
-----

.. autoclass:: lbmpy.methods.cumulantbased.CumulantBasedLbMethod
    :members:



Default Moment sets
===================

The following functions provide default sets of polynomial raw moments, central moments and cumulants
to be used in MRT-type methods.

.. autofunction:: lbmpy.methods.default_moment_sets.cascaded_moment_sets_literature

.. autofunction:: lbmpy.methods.default_moment_sets.mrt_orthogonal_modes_literature



Low-Level Method Creation Interface
===================================

The following classes and factory functions constitute the lower levels of abstraction in method creation.
They are called from the higher-level functions described above, or, in special cases, by the user directly.

Custom method variants in population space, raw and central moment space based on the hydrodynamic Maxwellian
equilibrium may be created using either 
:func:`lbmpy.methods.creationfunctions.create_with_discrete_maxwellian_equilibrium` or 
:func:`create_with_continuous_maxwellian_equilibrium`.

Methods may also be created using custom equilibrium distributions using
:func:`lbmpy.methods.creationfunctions.create_from_equilibrium`.

The desired collision space, and the transform classes defining the manner of transforming populations to that
space and back, are defined using :class:`lbmpy.enums.CollisionSpace` and :class:`lbmpy.methods.CollisionSpaceInfo`.

Collision Space Info
--------------------

.. autoclass lbmpy.methods.CollisionSpaceInfo
    :members:

Low-Level Creation Functions
----------------------------

.. autofunction:: lbmpy.methods.creationfunctions.create_with_discrete_maxwellian_equilibrium

.. autofunction:: lbmpy.methods.creationfunctions.create_with_continuous_maxwellian_equilibrium

.. autofunction:: lbmpy.methods.creationfunctions.create_from_equilibrium
