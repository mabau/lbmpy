*******************************************
Moment Transforms (lbmpy.moment_transforms)
*******************************************

The ``moment_transforms`` module provides an ecosystem for transformation of quantities between
lattice Boltzmann population space and other spaces of observable quantities. Currently, transforms
to raw and central moment space are available.

The common base class `lbmpy.moment_transforms.AbstractMomentTransform` defines the interface all transform classes share.
This interface, together with the fundamental principles all transforms adhere to, is explained 
in the following.

Principles of Collision Space Transforms
========================================

Each class of this module implements a bijective map :math:`\mathcal{M}` from population space 
to raw moment or central moment space, capable of transforming the particle distribution 
function :math:`\mathbf{f}` to (almost) arbitrary user-defined moment sets.

Monomial and Polynomial Moments
-------------------------------

We discriminate *monomial* and *polynomial* moments. The monomial moments are the canonical basis of their
corresponding space. Polynomial moments are defined as linear combinations of this basis. Monomial moments are
typically expressed by exponent tuples :math:`(\alpha, \beta, \gamma)`. The monomial raw moments are defined as

.. math::
    
    m_{\alpha \beta \gamma} 
        = \sum_{i = 0}^{q - 1} f_i c_{i,x}^{\alpha} c_{i,y}^{\beta} c_{i,z}^{\gamma}

and the monomial central moments are given by

.. math::
    
    \kappa_{\alpha \beta \gamma} 
        = \sum_{i = 0}^{q - 1} 
            f_i 
            (c_{i,x} - u_x)^{\alpha} 
            (c_{i,y} - u_y)^{\beta} 
            (c_{i,z} - u_z)^{\gamma}.

Polynomial moments are, in turn, expressed by
polynomial expressions :math:`p` in the coordinate symbols :math:`x`, :math:`y` and :math:`z`.
An exponent tuple :math:`(\alpha, \beta, \gamma)` corresponds directly 
to the mixed monomial expression :math:`x^{\alpha} y^{\beta} z^{\gamma}`. Polynomial moments are then
constructed as linear combinations of these monomials. For example, the polynomial

.. math::

    p(x,y,z) = x^2 + y^2 + z^2 + 1.

defines both the polynomial raw moment

.. math::

    M = m_{200} + m_{020} + m_{002}

and central moment

.. math::

    K = \kappa_{200} + \kappa_{020} + \kappa_{002}.


Transformation Matrices
-----------------------

The collision space basis for an MRT LB method on a :math:`DdQq` lattice is spanned by a set of :math:`q`
polynomial quantities, given by polynomials :math:`\left( p_i \right)_{i=0, \dots, q-1}`.
Through the polynomials, a polynomialization matrix :math:`P` is defined such that

.. math::

    \mathbf{M} &= P \mathbf{m} \\
    \mathbf{K} &= P \mathbf{\kappa}

Both rules can also be written using matrix multiplication, such that

.. math::
    \mathbf{m} = M \mathbf{f} 
    \qquad 
    \mathbf{\kappa} = K \mathbf{f}.

Further, there exists a mapping from raw to central moment space using (monomial or polynomial)
shift matrices (see `set_up_shift_matrix`) such that

.. math::
    \mathbf{\kappa} = N \mathbf{m}
    \qquad
    \mathbf{K} = N^P \mathbf{M}.

Working with the transformation matrices, those mappings can easily be inverted.
This module provides classes that derive efficient implementations of these transformations.

Moment Aliasing
---------------

For a collision space transform to be invertible, exactly :math:`q` independent polynomial quantities of
the collision space must be chosen. In that case, since all transforms discussed here are linear, the space of
populations is isomorphic to the chosen moment space. The important word here is 'independent'. Even if :math:`q`
distinct moment polynomials are chosen, due to discrete lattice artifacts, they might not span the entire collision
space if chosen wrongly. The discrete lattice structure gives rise to *moment aliasing* effects. The most simple such
effect occurs in the monomial raw moments, where are all nonzero even and all odd exponents are essentially the same.
For example, we have :math:`m_{400} = m_{200}` or :math:`m_{305} = m_{101}`. This rule holds on every direct-neighborhood
stencil. Sparse stencils, like :math:`D3Q15`, may introduce additional aliases. On the :math:`D3Q15` stencil, due to its
structure, we have also :math:`m_{112} = m_{110}` and even :math:`m_{202} = m_{220} = m_{022} = m_{222}`.

Including aliases in a set of monomial moments will lead to a non-invertible transform and is hence not possible.
In polynomials, however, aliases may occur without breaking the transform. Several established sets of polynomial
moments, like in orthogonal raw moment space MRT methods, will comprise :math:`q` polynomials :math:`\mathbf{M}` that in turn are built
out of :math:`r > q` monomial moments :math:`\mathbf{m}`. In that set of monomials, non-independent moments have to be included by definition.
Since the full transformation matrix :math:`M^P = PM` is still invertible, this is not a problem in general. If, however, the
two steps of the transform are computed separately, it becomes problematic, as the matrices :math:`M \in \mathbb{R}^{r \times q}`
and :math:`P \in \mathbb{R}^{q \times r}` are not invertible on their own. 

But there is a remedy. By eliminating aliases from the moment polynomials, they can be reduced to a new set of polynomials based
on a non-aliased reduced vector of monomial moments :math:`\tilde{\mathbf{m}} \in \mathbb{R}^{q}`, expressed through the reduced
matrix :math:`\tilde{P}`.


Interfaces and Usage
====================

Construction
------------

All moment transform classes expect either a sequence of exponent tuples or a sequence of polynomials that define
the set of quantities spanning the destination space. If polynomials are given, the exponent tuples of the underlying
set of monomials are extracted automatically. Depending on the implementation, moment aliases may be eliminated in the
process, altering both the polynomials and the set of monomials.


Forward and Backward Transform
------------------------------

The core functionality of the transform classes is given by the methods ``forward_transform`` and ``backward_transform``.
They return assignment collections containing the equations to compute moments from populations, and vice versa.

Symbols Used
^^^^^^^^^^^^

Unless otherwise specified by the user, monomial and polynomial quantities occur in the transformation equations according
to the naming conventions listed in the following table:

+--------------------------------+---------------------------------------------+----------------------+
|                                |              Monomial                       |    Polynomial        |
+--------------------------------+---------------------------------------------+----------------------+
| Pre-Collision Raw Moment       | :math:`m_{\alpha \beta \gamma}`             | :math:`M_{i}`        |
+--------------------------------+---------------------------------------------+----------------------+
| Post-Collision Raw Moment      | :math:`m_{post, \alpha \beta \gamma}`       | :math:`M_{post, i}`  |
+--------------------------------+---------------------------------------------+----------------------+
| Pre-Collision Central Moment   | :math:`\kappa_{\alpha \beta \gamma}`        | :math:`K_{i}`        |
+--------------------------------+---------------------------------------------+----------------------+
| Post-Collision Central Moment  | :math:`\kappa_{post, \alpha \beta \gamma}`  | :math:`K_{post, i}`  |
+--------------------------------+---------------------------------------------+----------------------+

These symbols are also exposed by the member properties `pre_collision_symbols`, `post_collision_symbols`, 
`pre_collision_monomial_symbols` and `post_collision_monomial_symbols`.

Forward Transform
^^^^^^^^^^^^^^^^^

Implementations of the `lbmpy.moment_transforms.AbstractMomentTransform.forward_transform` method 
derive equations for the transform from population to moment space, to compute pre-collision moments
from pre-collision populations. The returned ``AssignmentCollection`` has the `pre_collision_symbols` 
as left-hand sides of its main assignments, computed from the given ``pdf_symbols`` and, potentially,
the macroscopic density and velocity. Depending on the implementation, the `pre_collision_monomial_symbols`
may be part of the assignment collection in the form of subexpressions. 

Backward Transform
^^^^^^^^^^^^^^^^^^

Implementations of `lbmpy.moment_transforms.AbstractMomentTransform.backward_transform` contain the post-collision
polynomial quantities as free symbols of their equation right-hand sides, and compute the post-collision populations
from those. The PDF symbols are the left-hand sides of the main assignments.


Absorption of Conserved Quantity Equations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transformations from the population space to any space of observable quantities may *absorb* the equations 
defining the macroscopic quantities entering the equilibrium (typically the density :math:`\rho` and the
velocity :math:`\mathbf{u}`). This means that the ``forward_transform`` will possibly rewrite the
assignments given in the constructor argument ``conserved_quantity_equations`` to reduce
the total operation count. For example, in the transformation step from populations to
raw moments (see `lbmpy.moment_transforms.PdfsToMomentsByChimeraTransform`), :math:`\rho` can be aliased as the zeroth-order moment
:math:`m_{000}`. Assignments to the conserved quantities will then be part of the AssignmentCollection
returned by ``forward_transform`` and need not be added to the collision rule separately. 

Simplification
^^^^^^^^^^^^^^

Both ``forward_transform`` and ``backward_transform`` expect a keyword argument ``simplification``
which can be used to direct simplification steps applied during the derivation of the transformation
equations. Possible values are:

- `False` or ``'none'``: No simplification is to be applied
- `True` or ``'default'``: A default simplification strategy specific to the implementation is applied.
    The actual simplification steps depend strongly on the nature of the equations. They are defined by
    the implementation. It is the responsibility of the implementation to select the most effective
    simplification strategy.
- ``'default_with_cse'``: Same as ``'default'``, but with an additional pass of common subexpression elimination.


Working With Monomials
^^^^^^^^^^^^^^^^^^^^^^

In certain situations, we want the ``forward_transform`` to yield equations for the monomial symbols :math:`m_{\alpha \beta \gamma}`
and :math:`\kappa_{\alpha \beta \gamma}` only, and the ``backward_transform`` to return equations that also expect these symbols as input. 
In this case, it is not sufficient to pass a set of monomials or exponent tuples to the constructor, as those are still treated as 
polynomials internally. Instead, both transform methods expose keyword arguments ``return_monomials`` and ``start_from_monomials``, respectively.
If set to true, equations in the monomial moments are returned. They are best used *only* together with the ``exponent_tuples`` constructor argument
to have full control over the monomials. If polynomials are passed to the constructor, the behaviour of these flags is generally not well-defined,
especially in the presence of aliases.


The Transform Classes
=====================

Abstract Base Class
-------------------

.. autoclass:: lbmpy.moment_transforms.AbstractMomentTransform
    :members:


Moment Space Transforms
-----------------------

.. autoclass:: lbmpy.moment_transforms.PdfsToMomentsByMatrixTransform
    :members:

.. autoclass:: lbmpy.moment_transforms.PdfsToMomentsByChimeraTransform
    :members:


Central Moment Space Transforms
-------------------------------

.. autoclass:: lbmpy.moment_transforms.PdfsToCentralMomentsByMatrix
    :members:

.. autoclass:: lbmpy.moment_transforms.FastCentralMomentTransform
    :members:

.. autoclass:: lbmpy.moment_transforms.PdfsToCentralMomentsByShiftMatrix
    :members:

Cumulant Space Transforms
-------------------------

.. autoclass:: lbmpy.methods.centeredcumulant.cumulant_transform.CentralMomentsToCumulantsByGeneratingFunc
    :members:

