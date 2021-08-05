*******************************************
Moment Transforms (lbmpy.moment_transforms)
*******************************************

Abstract Base Class
===================

.. autoclass:: lbmpy.moment_transforms.AbstractMomentTransform
    :members:


Moment Space Transforms
===========================

By Matrix-Vector Multiplication
-------------------------------

.. autoclass:: lbmpy.moment_transforms.PdfsToMomentsByMatrixTransform
    :members:

By Chimera-Transform
--------------------

.. autoclass:: lbmpy.moment_transforms.PdfsToMomentsByChimeraTransform
    :members:


Central Moment Space Transforms
===============================

.. autoclass:: lbmpy.moment_transforms.PdfsToCentralMomentsByMatrix
    :members:

.. autoclass:: lbmpy.moment_transforms.FastCentralMomentTransform
    :members:

.. autoclass:: lbmpy.moment_transforms.PdfsToCentralMomentsByShiftMatrix
    :members:

Cumulant Space Transforms
=========================

.. autoclass:: lbmpy.methods.centeredcumulant.cumulant_transform.CentralMomentsToCumulantsByGeneratingFunc
    :members:

