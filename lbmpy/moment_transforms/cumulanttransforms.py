import numpy as np
import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.simp import SimplificationStrategy
from pystencils.simp.assignment_collection import SymbolGen

from lbmpy.moments import (
    moments_up_to_order, statistical_quantity_symbol, exponent_tuple_sort_key,
    monomial_to_polynomial_transformation_matrix
)

from itertools import product, chain

from .abstractmomenttransform import (
    AbstractMomentTransform,
    PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, POST_COLLISION_MONOMIAL_CENTRAL_MOMENT,
    PRE_COLLISION_CUMULANT, POST_COLLISION_CUMULANT,
    PRE_COLLISION_MONOMIAL_CUMULANT, POST_COLLISION_MONOMIAL_CUMULANT
)

#   ======================= Central Moments <-> Cumulants ==============================================================

WAVE_NUMBER_SYMBOLS = sp.symbols('Xi_x, Xi_y, Xi_z')


def moment_index_from_derivative(d, variables):
    diffs = d.args[1:]
    indices = [0] * len(variables)
    for var, count in diffs:
        indices[variables.index(var)] = count
    return tuple(indices)


def derivative_as_statistical_quantity(d, variables, quantity_name):
    indices = moment_index_from_derivative(d, variables)
    return statistical_quantity_symbol(quantity_name, indices)


def count_derivatives(derivative):
    return np.sum(np.fromiter((d[1] for d in derivative.args[1:]), int))


#   ============= Transformation through cumulant-generating function =============

class CentralMomentsToCumulantsByGeneratingFunc(AbstractMomentTransform):

    def __init__(self, stencil, cumulant_polynomials,
                 equilibrium_density,
                 equilibrium_velocity,
                 cumulant_exponents=None,
                 pre_collision_symbol_base=PRE_COLLISION_CUMULANT,
                 post_collision_symbol_base=POST_COLLISION_CUMULANT,
                 pre_collision_monomial_symbol_base=PRE_COLLISION_MONOMIAL_CUMULANT,
                 post_collision_monomial_symbol_base=POST_COLLISION_MONOMIAL_CUMULANT,
                 **kwargs):
        super(CentralMomentsToCumulantsByGeneratingFunc, self).__init__(
            stencil, equilibrium_density, equilibrium_velocity,
            moment_polynomials=cumulant_polynomials,
            moment_exponents=cumulant_exponents,
            pre_collision_symbol_base=pre_collision_symbol_base,
            post_collision_symbol_base=post_collision_symbol_base,
            pre_collision_monomial_symbol_base=pre_collision_monomial_symbol_base,
            post_collision_monomial_symbol_base=post_collision_monomial_symbol_base,
            **kwargs)

        self.cumulant_exponents = self.moment_exponents
        self.cumulant_polynomials = self.moment_polynomials

        if(len(self.cumulant_exponents) != stencil.Q):
            raise ValueError("Number of cumulant exponent tuples must match stencil size.")

        if(len(self.cumulant_polynomials) != stencil.Q):
            raise ValueError("Number of cumulant polynomials must match stencil size.")

        self.central_moment_exponents = self.compute_required_central_moments()

        self.mono_to_poly_matrix = monomial_to_polynomial_transformation_matrix(self.cumulant_exponents,
                                                                                self.cumulant_polynomials)
        self.poly_to_mono_matrix = self.mono_to_poly_matrix.inv()

    @property
    def required_central_moments(self):
        """The required central moments as a sorted list of exponent tuples"""
        return self.central_moment_exponents

    def compute_required_central_moments(self):
        def _contained_moments(m):
            ranges = (range(i + 1) for i in m)
            return product(*ranges)

        #   Always require zeroth and first moments
        required_moments = set(moments_up_to_order(1, dim=self.dim))
        #   In differentiating the generating function, all derivatives contained in c will occur
        #   --> all of these moments are required
        for c in self.cumulant_exponents:
            required_moments |= set(_contained_moments(c))

        assert len(required_moments) == self.stencil.Q, 'Number of required central moments must match stencil size.'

        return sorted(list(required_moments), key=exponent_tuple_sort_key)

    def forward_transform(self,
                          central_moment_base=PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT,
                          simplification=True,
                          subexpression_base='sub_k_to_C',
                          return_monomials=False):
        simplification = self._get_simp_strategy(simplification)

        monomial_equations = []
        for c_symbol, exp in zip(self.pre_collision_monomial_symbols, self.cumulant_exponents):
            eq = self.cumulant_from_central_moments(exp, central_moment_base)
            monomial_equations.append(Assignment(c_symbol, eq))

        if return_monomials:
            subexpressions = []
            main_assignments = monomial_equations
        else:
            subexpressions = monomial_equations
            poly_eqs = self.mono_to_poly_matrix @ sp.Matrix(self.pre_collision_monomial_symbols)
            main_assignments = [Assignment(c, v) for c, v in zip(self.pre_collision_symbols, poly_eqs)]

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self,
                           central_moment_base=POST_COLLISION_MONOMIAL_CENTRAL_MOMENT,
                           simplification=True,
                           subexpression_base='sub_C_to_k',
                           start_from_monomials=False):
        simplification = self._get_simp_strategy(simplification)

        subexpressions = []
        if not start_from_monomials:
            mono_eqs = self.poly_to_mono_matrix @ sp.Matrix(self.post_collision_symbols)
            subexpressions = [Assignment(c, v) for c, v in zip(self.post_collision_monomial_symbols, mono_eqs)]

        main_assignments = []
        for exp in self.central_moment_exponents:
            eq = self.central_moment_from_cumulants(exp, self.mono_base_post)
            k_symbol = statistical_quantity_symbol(central_moment_base, exp)
            main_assignments.append(Assignment(k_symbol, eq))

        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpressions=subexpressions,
                                  subexpression_symbol_generator=symbol_gen)

        if simplification:
            ac = simplification.apply(ac)
        return ac

    def cumulant_from_central_moments(self, cumulant_exponents, moment_symbol_base):
        dim = self.dim
        wave_numbers = WAVE_NUMBER_SYMBOLS[:dim]
        K = sp.Function('K')

        u_symbols = self.equilibrium_velocity
        rho = self.equilibrium_density

        C = sum(w * u for w, u in zip(wave_numbers, u_symbols)) + sp.log(K(*wave_numbers))

        diff_args = chain.from_iterable([var, i] for var, i in zip(wave_numbers, cumulant_exponents))
        cumulant = C.diff(*diff_args)

        derivatives = cumulant.atoms(sp.Derivative)
        derivative_subs = [(d, derivative_as_statistical_quantity(d, wave_numbers, moment_symbol_base))
                           for d in derivatives]
        derivative_subs = sorted(derivative_subs, key=lambda x: count_derivatives(x[0]), reverse=True)
        derivative_subs.append((K(*wave_numbers), statistical_quantity_symbol(moment_symbol_base, (0,) * dim)))

        cumulant = cumulant.subs(derivative_subs)

        value_subs = {x: 0 for x in wave_numbers}
        cumulant = cumulant.subs(value_subs)

        return (rho * cumulant).collect(rho)

    def central_moment_from_cumulants(self, moment_exponents, cumulant_symbol_base):
        dim = len(moment_exponents)
        wave_numbers = WAVE_NUMBER_SYMBOLS[:dim]
        C = sp.Function('C')

        u_symbols = self.equilibrium_velocity
        rho = self.equilibrium_density

        K = sp.exp(C(*wave_numbers) - sum(w * u for w,
                                          u in zip(wave_numbers, u_symbols)))

        diff_args = chain.from_iterable([var, i] for var, i in zip(wave_numbers, moment_exponents))
        moment = K.diff(*diff_args)

        derivatives = moment.atoms(sp.Derivative)

        derivative_subs = [(d, derivative_as_statistical_quantity(d, wave_numbers, 'c')) for d in derivatives]
        derivative_subs = sorted(derivative_subs, key=lambda x: count_derivatives(x[0]), reverse=True)
        derivative_subs.append((C(*wave_numbers), statistical_quantity_symbol('c', (0,) * dim)))

        moment = moment.subs(derivative_subs)

        value_subs = [(x, 0) for x in wave_numbers]

        moment = moment.subs(value_subs)

        c_indices = [(0,) * dim] + [moment_index_from_derivative(d, wave_numbers) for d in derivatives]
        moment = moment.subs([(statistical_quantity_symbol('c', idx),
                               statistical_quantity_symbol(cumulant_symbol_base, idx) / rho)
                              for idx in c_indices])

        return moment.expand().collect(rho)

    @property
    def _default_simplification(self):
        simplification = SimplificationStrategy()
        return simplification
# end class CentralMomentsToCumulantsByGeneratingFunc
