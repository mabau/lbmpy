import sympy as sp

from lbmpy.innerloopsplit import create_lbm_split_groups
from lbmpy.methods.momentbased.momentbasedmethod import MomentBasedLbMethod
from lbmpy.methods.momentbased.centralmomentbasedmethod import CentralMomentBasedLbMethod
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod

from lbmpy.methods.cumulantbased.cumulant_simplifications import (
    insert_log_products, expand_post_collision_central_moments)
from lbmpy.methods.momentbased.momentbasedsimplifications import (
    factor_density_after_factoring_relaxation_times, factor_relaxation_rates,
    replace_common_quadratic_and_constant_term, replace_density_and_velocity, replace_second_order_velocity_products,
    insert_half_force, insert_conserved_quantity_products)
from pystencils.simp import (
    SimplificationStrategy, add_subexpressions_for_divisions, apply_to_all_assignments,
    subexpression_substitution_in_main_assignments, insert_aliases, insert_constants,
    add_subexpressions_for_constants)
# add_subexpressions_for_constants)


def create_simplification_strategy(lb_method, split_inner_loop=False):
    if isinstance(lb_method, MomentBasedLbMethod):
        if lb_method.moment_space_collision:
            return _moment_space_simplification(split_inner_loop)
        else:
            if len(set(lb_method.relaxation_rates)) <= 2:
                # SRT and TRT methods with population-space collision
                return _srt_trt_population_space_simplification(split_inner_loop)
            else:
                # General MRT methods with population-space collision
                return _mrt_population_space_simplification(split_inner_loop)
    elif isinstance(lb_method, CentralMomentBasedLbMethod):
        return _moment_space_simplification(split_inner_loop)
    elif isinstance(lb_method, CumulantBasedLbMethod):
        return _cumulant_space_simplification(split_inner_loop)
    else:
        return SimplificationStrategy()

#   --------------- Internal ----------------------------------------------------------------------------


def _srt_trt_population_space_simplification(split_inner_loop):
    s = SimplificationStrategy()
    expand = apply_to_all_assignments(sp.expand)
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
    s.add(insert_constants)
    s.add(insert_aliases)
    s.add(lambda ac: ac.new_without_unused_subexpressions())
    return s


def _mrt_population_space_simplification(split_inner_loop):
    s = SimplificationStrategy()
    s.add(subexpression_substitution_in_main_assignments)
    s.add(add_subexpressions_for_divisions)
    if split_inner_loop:
        s.add(create_lbm_split_groups)
    s.add(lambda ac: ac.new_without_unused_subexpressions())
    return s


def _moment_space_simplification(split_inner_loop):
    s = SimplificationStrategy()
    s.add(insert_constants)
    s.add(insert_half_force)
    s.add(insert_aliases)
    s.add(add_subexpressions_for_divisions)
    s.add(add_subexpressions_for_constants)
    if split_inner_loop:
        s.add(create_lbm_split_groups)
    s.add(lambda ac: ac.new_without_unused_subexpressions())
    return s


def _cumulant_space_simplification(split_inner_loop):
    s = SimplificationStrategy()
    s.add(insert_constants)
    s.add(insert_aliases)
    s.add(insert_log_products)
    s.add(insert_conserved_quantity_products)
    s.add(insert_half_force)
    s.add(expand_post_collision_central_moments)
    s.add(insert_aliases)
    s.add(insert_constants)
    s.add(add_subexpressions_for_divisions)
    s.add(add_subexpressions_for_constants)
    if split_inner_loop:
        s.add(create_lbm_split_groups)
    s.add(lambda ac: ac.new_without_unused_subexpressions())
    return s
