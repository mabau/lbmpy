from abc import abstractmethod
import sympy as sp

from pystencils.simp import (SimplificationStrategy, sympy_cse)

from lbmpy.moments import (
    exponents_to_polynomial_representations, extract_monomials, exponent_tuple_sort_key)

from lbmpy.moments import statistical_quantity_symbol as sq_sym


PRE_COLLISION_MONOMIAL_RAW_MOMENT = 'm'
POST_COLLISION_MONOMIAL_RAW_MOMENT = 'm_post'

PRE_COLLISION_RAW_MOMENT = 'M'
POST_COLLISION_RAW_MOMENT = 'M_post'

PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT = 'kappa'
POST_COLLISION_MONOMIAL_CENTRAL_MOMENT = 'kappa_post'

PRE_COLLISION_CENTRAL_MOMENT = 'K'
POST_COLLISION_CENTRAL_MOMENT = 'K_post'

PRE_COLLISION_MONOMIAL_CUMULANT = 'c'
POST_COLLISION_MONOMIAL_CUMULANT = 'c_post'

PRE_COLLISION_CUMULANT = 'C'
POST_COLLISION_CUMULANT = 'C_post'


class AbstractMomentTransform:
    r"""Abstract Base Class for classes providing transformations between moment spaces."""

    def __init__(self, stencil,
                 equilibrium_density,
                 equilibrium_velocity,
                 moment_exponents=None,
                 moment_polynomials=None,
                 conserved_quantity_equations=None,
                 background_distribution=None,
                 pre_collision_symbol_base=None,
                 post_collision_symbol_base=None,
                 pre_collision_monomial_symbol_base=None,
                 post_collision_monomial_symbol_base=None):
        """Abstract Base Class constructor.

        Args:
            stencil: Nested tuple defining the velocity set
            equilibrium_density: Symbol of the equilibrium density used in the collision rule
            equilibrium_velocity: Tuple of symbols of the equilibrium velocity used in the collision rule
            moment_exponents=None: Exponent tuples of the monomial basis of the collision space
            moment_polynomials=None: Polynomial basis of the collision space
            conserved_quantity_equations: Optionally, an assignment collection computing the conserved quantities
                                          (typically density and velocity) from pre-collision populations
            background_distribution: If not `None`, zero-centered storage of the populations is assumed and the 
                                     moments of the passed distribution (instance of 
                                     `lbmpy.equilibrium.AbstractEquilibrium`) are included in the transform equations.

        """
        if moment_exponents is not None and moment_polynomials is not None:
            raise ValueError("Both moment_exponents and moment_polynomials were given. Pass only one of them!")

        self.stencil = stencil
        self.dim = stencil.D
        self.q = stencil.Q

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

        self.background_distribution = background_distribution

        self.base_pre = pre_collision_symbol_base
        self.base_post = post_collision_symbol_base
        self.mono_base_pre = pre_collision_monomial_symbol_base
        self.mono_base_post = post_collision_monomial_symbol_base

    @property
    def pre_collision_symbols(self):
        """List of symbols corresponding to the pre-collision quantities
        that will be the left-hand sides of assignments returned by :func:`forward_transform`."""
        return sp.symbols(f'{self.base_pre}_:{self.q}')

    @property
    def post_collision_symbols(self):
        """List of symbols corresponding to the post-collision quantities
        that are input to the right-hand sides of assignments returned by:func:`backward_transform`."""
        return sp.symbols(f'{self.base_post}_:{self.q}')

    @property
    def pre_collision_monomial_symbols(self):
        """List of symbols corresponding to the pre-collision monomial quantities
        that might exist as left-hand sides of subexpressions in the assignment collection 
        returned by :func:`forward_transform`."""
        return tuple(sq_sym(self.mono_base_pre, e) for e in self.moment_exponents)

    @property
    def post_collision_monomial_symbols(self):
        """List of symbols corresponding to the post-collision monomial quantities
        that might exist as left-hand sides of subexpressions in the assignment collection
        returned by :func:`backward_transform`."""
        return tuple(sq_sym(self.mono_base_post, e) for e in self.moment_exponents)

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
