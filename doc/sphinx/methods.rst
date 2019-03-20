***********************
Methods (lbmpy.methods)
***********************


LBM Method Interfaces
=====================

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

.. autoclass:: lbmpy.methods.MomentBasedLbMethod
    :members:

