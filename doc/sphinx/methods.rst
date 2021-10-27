***********************
Methods (lbmpy.methods)
***********************


LBM Method Interfaces
=====================

.. autoclass:: lbmpy.methods.LbmCollisionRule
    :members:

.. autoclass:: lbmpy.methods.AbstractLbMethod
    :members:

.. autoclass:: lbmpy.methods.AbstractConservedQuantityComputation
    :members:




LBM with conserved zeroth and first order
=========================================

.. autoclass:: lbmpy.methods.DensityVelocityComputation
    :members:




Moment-based methods
====================

Creation Functions
------------------

.. autofunction:: lbmpy.methods.create_srt

.. autofunction:: lbmpy.methods.create_trt

.. autofunction:: lbmpy.methods.create_trt_with_magic_number

.. autofunction:: lbmpy.methods.create_mrt_orthogonal

.. autofunction:: lbmpy.methods.create_with_continuous_maxwellian_eq_moments

.. autofunction:: lbmpy.methods.create_with_discrete_maxwellian_eq_moments


Class
-----

.. autoclass:: lbmpy.methods.momentbased.MomentBasedLbMethod
    :members:


Cumulant-based methods
======================

Creation Functions
------------------

.. autofunction:: lbmpy.methods.create_with_polynomial_cumulants

.. autofunction:: lbmpy.methods.create_with_monomial_cumulants

.. autofunction:: lbmpy.methods.create_with_default_polynomial_cumulants

.. autofunction:: lbmpy.methods.create_centered_cumulant_model


Utility
-------

.. autoclass:: lbmpy.methods.centeredcumulant.CenteredCumulantForceModel
    :members:

Class
-----

.. autoclass:: lbmpy.methods.centeredcumulant.CenteredCumulantBasedLbMethod
    :members:

Default Moment sets
-------------------

.. autofunction:: lbmpy.methods.default_moment_sets.cascaded_moment_sets_literature

.. autofunction:: lbmpy.methods.default_moment_sets.mrt_orthogonal_modes_literature
