from functools import partial
import sympy as sp

from lbmpy.innerloopsplit import create_lbm_split_groups
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from pystencils.assignment_collection.simplifications import apply_to_all_assignments, \
    subexpression_substitution_in_main_assignments, sympy_cse, add_subexpressions_for_divisions


def create_simplification_strategy(lb_method, cse_pdfs=False, cse_global=False, split_inner_loop=False):
    from pystencils.assignment_collection import SimplificationStrategy
    from lbmpy.methods import MomentBasedLbMethod
    from lbmpy.methods.momentbasedsimplifications import replace_second_order_velocity_products, \
        factor_density_after_factoring_relaxation_times, factor_relaxation_rates, cse_in_opposing_directions, \
        replace_common_quadratic_and_constant_term, replace_density_and_velocity

    s = SimplificationStrategy()

    expand = partial(apply_to_all_assignments, operation=sp.expand)
    expand.__name__ = "expand"

    if isinstance(lb_method, MomentBasedLbMethod):
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
    elif isinstance(lb_method, CumulantBasedLbMethod):
        s.add(expand)
        s.add(factor_relaxation_rates)

    s.add(add_subexpressions_for_divisions)

    if cse_pdfs:
        s.add(cse_in_opposing_directions)
    if cse_global:
        s.add(sympy_cse)

    return s
