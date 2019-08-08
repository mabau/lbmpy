"""Functions for implementation of fluctuating (randomized) lattice Boltzmann

to generate a fluctuating LBM the equilibrium moment values have to be scaled and an additive (random)
correction term is added to the collision rule
"""
import numpy as np
import sympy as sp

from lbmpy.moments import MOMENT_SYMBOLS
from pystencils import Assignment, TypedSymbol
from pystencils.rng import PhiloxFourFloats, random_symbol
from pystencils.simp.assignment_collection import SymbolGen


def add_fluctuations_to_collision_rule(collision_rule, temperature=None, variances=(),
                                       block_offsets=(0, 0, 0), seed=TypedSymbol("seed", np.uint32),
                                       rng_node=PhiloxFourFloats, c_s_sq=sp.Rational(1, 3)):
    """"""
    if not (temperature and not variances) or (temperature and variances):
        raise ValueError("Fluctuating LBM: Pass either 'temperature' or 'variances'.")

    method = collision_rule.method
    if not variances:
        variances = fluctuating_variance_from_temperature(method, temperature, c_s_sq)

    rng_symbol_gen = random_symbol(collision_rule.subexpressions, seed,
                                   rng_node=rng_node, dim=method.dim, offsets=block_offsets)
    correction = fluctuation_correction(method, rng_symbol_gen, variances)

    for i, corr in enumerate(correction):
        collision_rule.main_assignments[i] = Assignment(collision_rule.main_assignments[i].lhs,
                                                        collision_rule.main_assignments[i].rhs + corr)


def fluctuating_variance_from_temperature(method, temperature, c_s_sq=sp.Symbol("c_s") ** 2):
    """Produces variance equations according to (3.54) in Schiller08"""
    normalization_factors = abs(method.moment_matrix) * sp.Matrix(method.weights)
    density = method.zeroth_order_equilibrium_moment_symbol
    if method.conserved_quantity_computation.zero_centered_pdfs:
        density += 1
    mu = temperature * density / c_s_sq
    return [sp.sqrt(mu * norm * (1 - (1 - rr) ** 2))
            for norm, rr in zip(normalization_factors, method.relaxation_rates)]


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
