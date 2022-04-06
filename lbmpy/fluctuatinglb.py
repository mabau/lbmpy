"""Functions for implementation of fluctuating (randomized) lattice Boltzmann

to generate a fluctuating LBM the equilibrium moment values have to be scaled and an additive (random)
correction term is added to the collision rule
"""
import numpy as np
import sympy as sp

from lbmpy.moments import MOMENT_SYMBOLS, is_shear_moment, get_order
from lbmpy.equilibrium import ContinuousHydrodynamicMaxwellian
from pystencils import Assignment, TypedSymbol
from pystencils.rng import PhiloxFourFloats, random_symbol
from pystencils.simp.assignment_collection import SymbolGen


def add_fluctuations_to_collision_rule(collision_rule, temperature=None, amplitudes=(),
                                       block_offsets=(0, 0, 0), seed=TypedSymbol("seed", np.uint32),
                                       rng_node=PhiloxFourFloats, c_s_sq=sp.Rational(1, 3)):
    """"""
    if not (temperature and not amplitudes) or (temperature and amplitudes):
        raise ValueError("Fluctuating LBM: Pass either 'temperature' or 'amplitudes'.")
    if collision_rule.method.conserved_quantity_computation.zero_centered_pdfs:
        raise ValueError("The fluctuating LBM is not implemented for zero-centered PDF storage.")

    method = collision_rule.method
    if not amplitudes:
        amplitudes = fluctuation_amplitude_from_temperature(method, temperature, c_s_sq)
    if block_offsets == 'walberla':
        block_offsets = tuple(TypedSymbol("block_offset_{}".format(i), np.uint32) for i in range(3))

    if not method.is_weighted_orthogonal:
        raise ValueError("Fluctuations can only be added to weighted-orthogonal methods")

    rng_symbol_gen = random_symbol(collision_rule.subexpressions, seed=seed,
                                   rng_node=rng_node, dim=method.dim, offsets=block_offsets)
    correction = fluctuation_correction(method, rng_symbol_gen, amplitudes)

    for i, corr in enumerate(correction):
        collision_rule.main_assignments[i] = Assignment(collision_rule.main_assignments[i].lhs,
                                                        collision_rule.main_assignments[i].rhs + corr)


def fluctuation_amplitude_from_temperature(method, temperature, c_s_sq=sp.Symbol("c_s") ** 2):
    """Produces amplitude equations according to (2.60) and (3.54) in Schiller08"""
    normalization_factors = sp.matrix_multiply_elementwise(method.moment_matrix, method.moment_matrix) * \
        sp.Matrix(method.weights)
    density = method.zeroth_order_equilibrium_moment_symbol
    if method.conserved_quantity_computation.zero_centered_pdfs:
        density += 1
    mu = temperature * density / c_s_sq
    return [sp.sqrt(mu * norm * (1 - (1 - rr) ** 2))
            for norm, rr in zip(normalization_factors, method.relaxation_rates)]


def fluctuation_correction(method, rng_generator, amplitudes=SymbolGen("phi")):
    """Returns additive correction terms to be added to the the collided pdfs"""
    conserved_moments = {sp.sympify(1), *MOMENT_SYMBOLS}

    # A diagonal matrix containing the random fluctuations
    random_matrix = sp.Matrix([0 if m in conserved_moments else (next(rng_generator) - 0.5) * sp.sqrt(12)
                               for m in method.moments])
    amplitude_matrix = sp.diag(*[v for v, _ in zip(iter(amplitudes), method.moments)])

    # corrections are applied in real space hence we need to convert to real space here
    return method.moment_matrix.inv() * amplitude_matrix * random_matrix


class ThermalizedEquilibrium(ContinuousHydrodynamicMaxwellian):
    """TODO: Remove Again! 
        
    This class is currently only used in the tutorial notebook `demo_thermalized_lbm.ipynb`
    and has been added only temporarily, until the thermalized LBM is updated to our new
    equilibrium framework."""
    def __init__(self, random_number_symbols, **kwargs):
        super().__init__(**kwargs)
        self.random_number_symbols = random_number_symbols

    def moment(self, exponent_tuple_or_polynomial):
        value = super().moment(exponent_tuple_or_polynomial)
        if is_shear_moment(exponent_tuple_or_polynomial, dim=self.dim):
            value += self.random_number_symbols[0] * 0.001
        elif get_order(exponent_tuple_or_polynomial) > 2:
            value += self.random_number_symbols[1] * 0.001
        return value
