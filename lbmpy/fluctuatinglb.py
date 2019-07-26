"""Functions for implementation of fluctuating (randomized) lattice Boltzmann

to generate a fluctuating LBM the equilibrium moment values have to be scaled and an additive (random)
correction term is added to the collision rule


Usage example:

    >>> from lbmpy.session import *
    >>> from pystencils.rng import random_symbol
    >>> method = create_lb_method(stencil='D2Q9', method='MRT')
    >>> rescaled_method = method_with_rescaled_equilibrium_values(method)
    >>> cr = create_lb_collision_rule(lb_method=rescaled_method)
    >>> correction = fluctuation_correction(rescaled_method,
    ...                                     rng_generator=random_symbol(cr.subexpressions, dim=method.dim))
    >>> for i, corr in enumerate(correction):
    ...     cr.main_assignments[i] = ps.Assignment(cr.main_assignments[i].lhs,
    ...                                            cr.main_assignments[i].rhs + corr)
    >>>

"""
import sympy as sp

from lbmpy.moments import MOMENT_SYMBOLS
from pystencils.simp.assignment_collection import SymbolGen


def method_with_rescaled_equilibrium_values(base_method):
    """Re-scales the equilibrium moments by 1 / sqrt(M*w) with moment matrix M and weights w"""
    from lbmpy.creationfunctions import create_lb_method_from_existing

    sig_k = abs(base_method.moment_matrix) * sp.Matrix(base_method.weights)

    def modification_rule(moment, eq, rr):
        i = base_method.moments.index(moment)
        return moment, eq / sp.sqrt(sig_k[i]), rr

    return create_lb_method_from_existing(base_method, modification_rule)


def fluctuation_correction(method, rng_generator, variances=SymbolGen("variance")):
    """Returns additive correction terms to be added to the the collided pdfs"""
    conserved_moments = {sp.sympify(1), *MOMENT_SYMBOLS}

    # A diagonal matrix containing the random fluctuations
    random_matrix = sp.Matrix([0 if m in conserved_moments else next(rng_generator) for m in method.moments])
    random_variance = sp.diag(*[v for v, _ in zip(iter(variances), method.moments)])

    # corrections are applied in real space hence we need to convert to real space here
    return method.moment_matrix.inv() * random_variance * random_matrix
