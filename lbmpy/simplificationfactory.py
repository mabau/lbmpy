import sympy as sp

from lbmpy.innerloopsplit import create_lbm_split_groups
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from lbmpy.methods.momentbased import MomentBasedLbMethod
from lbmpy.methods.momentbasedsimplifications import (
    factor_density_after_factoring_relaxation_times, factor_relaxation_rates,
    replace_common_quadratic_and_constant_term, replace_density_and_velocity, replace_second_order_velocity_products)
from pystencils.simp import (
    SimplificationStrategy, add_subexpressions_for_divisions, apply_to_all_assignments,
    subexpression_substitution_in_main_assignments)


def create_simplification_strategy(lb_method, split_inner_loop=False):
    s = SimplificationStrategy()
    expand = apply_to_all_assignments(sp.expand)

    if isinstance(lb_method, MomentBasedLbMethod):
        if len(set(lb_method.relaxation_rates)) <= 2:
            s.add(expand)
            s.add(replace_second_order_velocity_products)
            s.add(expand)
            s.add(factor_relaxation_rates)
            s.add(replace_density_and_velocity)
            s.add(replace_common_quadratic_and_constant_term)
            s.add(factor_density_after_factoring_relaxation_times)
            s.add(subexpression_substitution_in_main_assignments)
            if split_inner_loop:
                s.add(create_lbm_split_groups)
            s.add(add_subexpressions_for_divisions)
        else:
            s.add(subexpression_substitution_in_main_assignments)
            if split_inner_loop:
                s.add(create_lbm_split_groups)
    elif isinstance(lb_method, CumulantBasedLbMethod):
        s.add(expand)
        s.add(factor_relaxation_rates)
        s.add(add_subexpressions_for_divisions)

    return s
