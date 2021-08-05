from abc import abstractmethod

from pystencils.simp import (SimplificationStrategy, sympy_cse)

from lbmpy.moments import (
    exponents_to_polynomial_representations, extract_monomials, exponent_tuple_sort_key)


PRE_COLLISION_RAW_MOMENT = 'm'
POST_COLLISION_RAW_MOMENT = 'm_post'

PRE_COLLISION_MOMENT = 'M'
POST_COLLISION_MOMENT = 'M_post'

PRE_COLLISION_CENTRAL_MOMENT = 'kappa'
POST_COLLISION_CENTRAL_MOMENT = 'kappa_post'


class AbstractMomentTransform:
    r"""Abstract Base Class for classes providing transformations between moment spaces. 
    
    These transformations are bijective maps between two spaces :math:`\mathcal{S}` 
    and :math:`\mathcal{D}` (i.e. populations and moments, or central moments and cumulants). 
    The forward map  :math:`F : \mathcal{S} \mapsto \mathcal{D}` is given by :func:`forward_transform`, 
    and the backward (inverse) map :math:`F^{-1} : \mathcal{D} \mapsto \mathcal{S}`
    is provided by :func:`backward_transform`. The transformations are intended for use within lattice 
    Boltzmann collision operators: The :func:`forward_transform` to map pre-collision populations to the 
    required collision space (possibly by several consecutive transformations), and the 
    :func:`backward_transform` to map post-collision quantities back to populations.

    **Transformations**

    Transformation equations must be returned by implementations of :func:`forward_transform` and 
    :func:`backward_transform` as an :class:`pystencils.AssignmentCollection`.

    - :func:`forward_transform` returns an AssignmentCollection which depends on quantities of the domain 
      :math:`\mathcal{S}` and contains the equations to map them to the codomain :math:`\mathcal{D}`.
    - :func:`backward_transform` is the inverse of :func:`forward_transform` and returns an AssignmentCollection 
      which maps quantities of the codomain :math:`\mathcal{D}` back to the domain :math:`\mathcal{S}`.

    **Absorption of Conserved Quantity Equations**

    Transformations from the population space to any space of observable quantities may *absorb* the equations 
    defining the macroscopic quantities entering the equilibrium (typically the density :math:`\rho` and the
    velocity :math:`\mathbf{u}`). This means that the :func:`forward_transform` will possibly rewrite the
    assignments given in the constructor argument ``conserved_quantity_equations`` to reduce
    the total operation count. For example, in the transformation step from populations to
    raw moments (see `PdfsToMomentsByChimeraTransform`), :math:`\rho` can be aliased as the zeroth-order moment
    :math:`m_{000}`. Assignments to the conserved quantities will then be part of the AssignmentCollection
    returned by :func:`forward_transform` and need not be added to the collision rule separately. 

    **Simplification**

    Both :func:`forward_transform` and :func:`backward_transform` expect a keyword argument ``simplification``
    which can be used to direct simplification steps applied during the derivation of the transformation
    equations. Possible values are:
    
    - `False` or ``'none'``: No simplification is to be applied
    - `True` or ``'default'``: A default simplification strategy specific to the implementation is applied.
      The actual simplification steps depend strongly on the nature of the equations. They are defined by
      the implementation. It is the responsibility of the implementation to select the most effective
      simplification strategy.
    - ``'default_with_cse'``: Same as ``'default'``, but with an additional pass of common subexpression elimination.
    """

    def __init__(self, stencil,
                 equilibrium_density,
                 equilibrium_velocity,
                 moment_exponents=None,
                 moment_polynomials=None,
                 conserved_quantity_equations=None,
                 **kwargs):
        """Abstract Base Class constructor.
        
        Args:
            stencil: Nested tuple defining the velocity set
            equilibrium_density: Symbol of the equilibrium density used in the collision rule
            equilibrium_velocity: Tuple of symbols of the equilibrium velocity used in the collision rule
            moment_exponents=None: Exponent tuples of the monomial basis of the collision space
            moment_polynomials=None: Polynomial basis of the collision space
            conserved_quantity_equations: Optionally, an assignment collection computing the conserved quantities
                                          (typically density and velocity) from pre-collision populations
        """
        if moment_exponents is not None and moment_polynomials is not None:
            raise ValueError("Both moment_exponents and moment_polynomials were given. Pass only one of them!")
        
        self.stencil = stencil
        self.dim = len(stencil[0])
        self.q = len(stencil)

        if moment_exponents is not None:
            self.moment_exponents = list(moment_exponents)
            self.moment_polynomials = exponents_to_polynomial_representations(self.moment_exponents)
        elif moment_polynomials is not None:
            self.moment_polynomials = moment_polynomials
            moment_exponents = list(extract_monomials(moment_polynomials, dim=self.dim))
            self.moment_exponents = sorted(moment_exponents, key=exponent_tuple_sort_key)
        else:
            raise ValueError("You must provide either moment_exponents or moment_polynomials!")

        self.cqe = conserved_quantity_equations
        self.equilibrium_density = equilibrium_density
        self.equilibrium_velocity = equilibrium_velocity

    @abstractmethod
    def forward_transform(self, *args, **kwargs):
        """Implemented in a subclass, will return the forward transform equations."""
        raise NotImplementedError("forward_transform must be implemented in a subclass")

    @abstractmethod
    def backward_transform(self, *args, **kwargs):
        """Implemented in a subclass, will return the backward transform equations."""
        raise NotImplementedError("backward_transform must be implemented in a subclass")

    @property
    def absorbs_conserved_quantity_equations(self):
        """Whether or not the given conserved quantity equations will be included in
        the assignment collection returned by :func:`forward_transform`, possibly in simplified
        form."""
        return False

    @property
    def _default_simplification(self):
        return SimplificationStrategy()

    def _get_simp_strategy(self, simplification, direction=None):
        if isinstance(simplification, bool):
            simplification = 'default' if simplification else 'none'

        if simplification == 'default' or simplification == 'default_with_cse':
            simp = self._default_simplification if direction is None else self._default_simplification[direction]
            if simplification == 'default_with_cse':
                simp.add(sympy_cse)
            return simp
        else:
            return None
