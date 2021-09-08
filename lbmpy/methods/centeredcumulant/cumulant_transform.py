import numpy as np
import sympy as sp

from pystencils import Assignment, AssignmentCollection
from pystencils.simp import SimplificationStrategy, add_subexpressions_for_divisions
from pystencils.simp.assignment_collection import SymbolGen

from lbmpy.moments import (
    moments_up_to_order, get_order, statistical_quantity_symbol, exponent_tuple_sort_key
)

from itertools import product, chain

from lbmpy.moment_transforms import (
    AbstractMomentTransform, PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT, POST_COLLISION_MONOMIAL_CENTRAL_MOMENT
)

#   ======================= Central Moments <-> Cumulants ==============================================================

WAVE_NUMBER_SYMBOLS = sp.symbols('Xi_x, Xi_y, Xi_z')

PRE_COLLISION_CUMULANT = 'C'
POST_COLLISION_CUMULANT = 'C_post'


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

    def __init__(self, stencil, cumulant_exponents, equilibrium_density, equilibrium_velocity, **kwargs):
        super(CentralMomentsToCumulantsByGeneratingFunc, self).__init__(
            stencil, equilibrium_density, equilibrium_velocity,  
            moment_exponents=cumulant_exponents, **kwargs)

        self.cumulant_exponents = self.moment_exponents
        self.central_moment_exponents = self.compute_required_central_moments()

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
        return sorted(list(required_moments), key=exponent_tuple_sort_key)

    def forward_transform(self,
                          cumulant_base=PRE_COLLISION_CUMULANT,
                          central_moment_base=PRE_COLLISION_MONOMIAL_CENTRAL_MOMENT,
                          simplification=True,
                          subexpression_base='sub_k_to_C'):
        simplification = self._get_simp_strategy(simplification)

        main_assignments = []
        for exp in self.cumulant_exponents:
            eq = self.cumulant_from_central_moments(exp, central_moment_base)
            c_symbol = statistical_quantity_symbol(cumulant_base, exp)
            main_assignments.append(Assignment(c_symbol, eq))
        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(
            main_assignments, subexpression_symbol_generator=symbol_gen)
        
        if simplification:
            ac = simplification.apply(ac)
        return ac

    def backward_transform(self,
                           cumulant_base=POST_COLLISION_CUMULANT,
                           central_moment_base=POST_COLLISION_MONOMIAL_CENTRAL_MOMENT,
                           simplification=True,
                           omit_conserved_moments=False,
                           subexpression_base='sub_C_to_k'):
        simplification = self._get_simp_strategy(simplification)

        main_assignments = []
        for exp in self.central_moment_exponents:
            if omit_conserved_moments and get_order(exp) <= 1:
                continue
            eq = self.central_moment_from_cumulants(exp, cumulant_base)
            k_symbol = statistical_quantity_symbol(central_moment_base, exp)
            main_assignments.append(Assignment(k_symbol, eq))
        symbol_gen = SymbolGen(subexpression_base)
        ac = AssignmentCollection(main_assignments, subexpression_symbol_generator=symbol_gen)
        
        if simplification:
            ac = simplification.apply(ac)
        return ac

    def cumulant_from_central_moments(self, cumulant_exponents, moment_symbol_base):
        dim = self.dim
        assert len(cumulant_exponents) == dim
        wave_numbers = WAVE_NUMBER_SYMBOLS[:dim]
        K = sp.Function('K')

        u_symbols = self.equilibrium_velocity
        rho = self.equilibrium_density

        C = sum(w * u for w, u in zip(wave_numbers, u_symbols)) + sp.log(K(*wave_numbers))

        diff_args = chain.from_iterable([var, i] for var, i in zip(wave_numbers, cumulant_exponents))
        cumulant = C.diff(*diff_args)
        required_central_moments = set()

        derivatives = cumulant.atoms(sp.Derivative)
        derivative_subs = []
        for d in derivatives:
            moment_index = moment_index_from_derivative(d, wave_numbers)
            if sum(moment_index) > 1:  # lower order moments are replaced anyway
                required_central_moments.add(moment_index)
            derivative_subs.append((d, statistical_quantity_symbol(moment_symbol_base, moment_index)))
        derivative_subs = sorted(derivative_subs, key=lambda x: count_derivatives(x[0]), reverse=True)

        # K(0,0,0) = rho
        cumulant = cumulant.subs(derivative_subs)

        # First central moments equal zero
        value_subs = {x: 0 for x in wave_numbers}
        for i in range(dim):
            indices = [0] * dim
            indices[i] = 1
            value_subs[statistical_quantity_symbol(
                moment_symbol_base, indices)] = 0

        cumulant = cumulant.subs(value_subs)
        cumulant = cumulant.subs(K(*((0,) * dim)), rho)  # K(0,0,0) = rho

        return (rho * cumulant).collect(rho)

    def central_moment_from_cumulants(self, moment_exponents, cumulant_symbol_base):
        dim = len(moment_exponents)
        wave_numbers = WAVE_NUMBER_SYMBOLS[:dim]
        C = sp.Function('C')

        u_symbols = self.equilibrium_velocity
        rho = self.equilibrium_density

        K = sp.exp(C(*wave_numbers) - sum(w * u for w,
                                          u in zip(wave_numbers, u_symbols)))

        diff_args = chain.from_iterable(
            [var, i] for var, i in zip(wave_numbers, moment_exponents))
        moment = K.diff(*diff_args)

        derivatives = moment.atoms(sp.Derivative)
        c_indices = [moment_index_from_derivative(d, wave_numbers) for d in derivatives]

        derivative_subs = [(d, derivative_as_statistical_quantity(d, wave_numbers, 'c')) for d in derivatives]
        derivative_subs = sorted(derivative_subs, key=lambda x: count_derivatives(x[0]), reverse=True)

        moment = moment.subs(derivative_subs)

        # C(0,0,0) = log(rho), c_100 = u_x, etc.
        value_subs = [(x, 0) for x in wave_numbers]
        for i, u in enumerate(u_symbols):
            c_idx = [0] * dim
            c_idx[i] = 1
            value_subs.append((statistical_quantity_symbol('c', c_idx), u))

        moment = moment.subs(value_subs)
        moment = moment.subs(C(*((0,) * dim)), sp.log(rho))
        moment = moment.subs([(statistical_quantity_symbol('c', idx),
                               statistical_quantity_symbol(cumulant_symbol_base, idx) / rho)
                              for idx in c_indices])

        return moment.expand().collect(rho)

    @property
    def _default_simplification(self):
        simplification = SimplificationStrategy()
        simplification.add(add_subexpressions_for_divisions)

        return simplification
# end class CentralMomentsToCumulantsByGeneratingFunc
