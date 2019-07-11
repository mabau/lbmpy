"""
This module holds special transformations for simplifying the collision equations of moment-based methods.
All of these transformations operate on :class:`pystencils.AssignmentCollection` and need special
simplification hints, which are set by the MomentBasedLbMethod.
"""
import sympy as sp

from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from pystencils import Assignment
from pystencils.sympyextensions import (
    extract_most_common_factor, replace_second_order_products, subs_additive)


def replace_second_order_velocity_products(cr: LbmCollisionRule):
    """
    Replaces mixed quadratic velocity terms like :math:`u_0 * u_1` by :math:`(u_0+u_1)^2 - u_0^2 - u_1^2`
    Required simplification hints:
        - velocity: sequence of velocity symbols
    """
    sh = cr.simplification_hints
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"

    result = []
    substitutions = []
    u = sh['velocity']
    for i, s in enumerate(cr.main_assignments):
        new_rhs = replace_second_order_products(s.rhs, u, positive=None, replace_mixed=substitutions)
        result.append(Assignment(s.lhs, new_rhs))
    res = cr.copy(result)
    res.subexpressions += substitutions
    return res


def factor_relaxation_rates(cr: LbmCollisionRule):
    """
    Factors collision equations by relaxation rates.
    Required simplification hints:
        - relaxation_rates: Sequence of relaxation rates
    """
    sh = cr.simplification_hints
    assert 'relaxation_rates' in sh, "Needs simplification hint 'relaxation_rates': Sequence of relaxation rates"
    if len(sh['relaxation_rates']) > 19:  # heuristics, works well if there is a small number of relaxation rates
        return cr

    relaxation_rates = sp.Matrix(sh['relaxation_rates']).atoms(sp.Symbol)

    result = []
    for s in cr.main_assignments:
        new_rhs = s.rhs
        for rp in relaxation_rates:
            new_rhs = new_rhs.collect(rp)
        result.append(Assignment(s.lhs, new_rhs))
    return cr.copy(result)


def factor_density_after_factoring_relaxation_times(cr: LbmCollisionRule):
    """
    Tries to factor out the density. This only works if previously
    :func:`lbmpy.methods.momentbasedsimplifications.factor_relaxation_times` was run.

    This transformations makes only sense for compressible models - for incompressible models this does nothing

    Required simplification hints:
        - density: density symbol which is factored out
        - relaxation_rates: set of symbolic relaxation rates in which the terms are assumed to be already factored
    """
    sh = cr.simplification_hints
    assert 'density' in sh, "Needs simplification hint 'density': Symbol for density"
    assert 'relaxation_rates' in sh, "Needs simplification hint 'relaxation_rates': Set of symbolic relaxation rates"

    relaxation_rates = sp.Matrix(sh['relaxation_rates']).atoms(sp.Symbol)
    result = []
    rho = sh['density']
    for s in cr.main_assignments:
        new_rhs = s.rhs
        for rp in relaxation_rates:
            coeff = new_rhs.coeff(rp)
            new_rhs = new_rhs.subs(coeff, coeff.collect(rho))
        result.append(Assignment(s.lhs, new_rhs))
    return cr.copy(result)


def replace_density_and_velocity(cr: LbmCollisionRule):
    """
    Looks for terms that can be replaced by the density or by one of the velocities
        Required simplification hints:
        - density: density symbol
        - velocity: sequence of velocity symbols
    """
    sh = cr.simplification_hints
    assert 'density' in sh, "Needs simplification hint 'density': Symbol for density"
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"
    rho = sh['density']
    u = sh['velocity']

    substitutions = cr.new_filtered([rho] + list(u)).new_without_subexpressions().main_assignments
    result = []
    for s in cr.main_assignments:
        new_rhs = s.rhs
        for replacement in substitutions:
            new_rhs = subs_additive(new_rhs, replacement.lhs, replacement.rhs, required_match_replacement=0.5)
        result.append(Assignment(s.lhs, new_rhs))
    return cr.copy(result)


def replace_common_quadratic_and_constant_term(cr: LbmCollisionRule):
    """
    A common quadratic term (f_eq_common) is extracted from the collision equation for center
    and substituted in all equations

    Required simplification hints:
        - density: density symbol
        - velocity: sequence of velocity symbols
        - relaxation_rates: Sequence of relaxation rates
        - stencil:
    """
    sh = cr.simplification_hints
    assert 'density' in sh, "Needs simplification hint 'density': Symbol for density"
    assert 'velocity' in sh, "Needs simplification hint 'velocity': Sequence of velocity symbols"
    assert 'relaxation_rates' in sh, "Needs simplification hint 'relaxation_rates': Sequence of relaxation rates"

    stencil = cr.method.stencil
    assert sum([abs(e) for e in stencil[0]]) == 0, "Works only if first stencil entry is the center direction"
    f_eq_common = __get_common_quadratic_and_constant_terms(cr)
    if f_eq_common is None:
        return cr

    if len(f_eq_common.args) > 1:
        f_eq_common = Assignment(sp.Symbol('f_eq_common'), f_eq_common)
        result = []
        for s in cr.main_assignments:
            new_rhs = subs_additive(s.rhs, f_eq_common.lhs, f_eq_common.rhs, required_match_replacement=0.5)
            result.append(Assignment(s.lhs, new_rhs))
        res = cr.copy(result)
        res.subexpressions.append(f_eq_common)
        return res
    else:
        return cr


def cse_in_opposing_directions(cr: LbmCollisionRule):
    """
    Looks for common subexpressions in terms for opposing directions (e.g. north & south, top & bottom )

    Required simplification hints:
        - relaxation_rates: Sequence of relaxation rates
        - post_collision_pdf_symbols: sequence of symbols
    """
    sh = cr.simplification_hints
    assert 'relaxation_rates' in sh, "Needs simplification hint 'relaxation_rates': Sequence of relaxation rates"
    update_rules = cr.main_assignments
    stencil = cr.method.stencil

    if not sh['relaxation_rates']:
        return cr

    relaxation_rates = sp.Matrix(sh['relaxation_rates']).atoms(sp.Symbol)

    replacement_symbol_generator = cr.subexpression_symbol_generator

    direction_to_update_rule = {direction: update_rule for update_rule, direction in zip(update_rules, stencil)}
    result = []
    substitutions = []
    new_coefficient_substitutions = dict()
    for update_rule, direction in zip(update_rules, stencil):
        if direction not in direction_to_update_rule:
            continue  # already handled the inverse direction
        inverse_dir = tuple([-i for i in direction])
        inverse_rule = direction_to_update_rule[inverse_dir]
        if inverse_dir == direction:
            result.append(update_rule)  # center is not modified
            continue
        del direction_to_update_rule[inverse_dir]
        del direction_to_update_rule[direction]

        update_rules = [update_rule, inverse_rule]

        if len(relaxation_rates) == 0:
            found_subexpressions, new_terms = sp.cse(update_rules, symbols=replacement_symbol_generator,
                                                     order='None', optimizations=[])
            substitutions += [Assignment(f[0], f[1]) for f in found_subexpressions]

            update_rules = new_terms
        else:
            for relaxation_rate in relaxation_rates:
                terms = [update_rule.rhs.coeff(relaxation_rate) for update_rule in update_rules]
                result_of_common_factor = [extract_most_common_factor(t) for t in terms]
                common_factors = [r[0] for r in result_of_common_factor]
                terms_without_factor = [r[1] for r in result_of_common_factor]

                if common_factors[0] == common_factors[1] and common_factors[0] != 1:
                    new_coefficient = common_factors[0] * relaxation_rate
                    if new_coefficient not in new_coefficient_substitutions:
                        new_coefficient_substitutions[new_coefficient] = next(replacement_symbol_generator)
                    new_coefficient = new_coefficient_substitutions[new_coefficient]
                    handled_terms = terms_without_factor
                else:
                    new_coefficient = relaxation_rate
                    handled_terms = terms

                found_subexpressions, new_terms = sp.cse(handled_terms, symbols=replacement_symbol_generator,
                                                         order='None', optimizations=[])
                substitutions += [Assignment(f[0], f[1]) for f in found_subexpressions]

                update_rules = [Assignment(ur.lhs, ur.rhs.subs(relaxation_rate * old_term, new_coefficient * new_term))
                                for ur, new_term, old_term in zip(update_rules, new_terms, terms)]

        result += update_rules

    for term, substituted_var in new_coefficient_substitutions.items():
        substitutions.append(Assignment(substituted_var, term))

    result.sort(key=lambda e: cr.method.post_collision_pdf_symbols.index(e.lhs))
    res = cr.copy(result)
    res.subexpressions += substitutions
    return res


# -------------------------------------- Helper Functions --------------------------------------------------------------

def __get_common_quadratic_and_constant_terms(cr: LbmCollisionRule):
    """Determines a common subexpression useful for most LBM model often called f_eq_common.
    It contains the quadratic and constant terms of the center update rule."""
    sh = cr.simplification_hints
    stencil = cr.method.stencil
    relaxation_rates = sp.Matrix(sh['relaxation_rates']).atoms(sp.Symbol)

    dim = len(stencil[0])

    pdf_symbols = cr.free_symbols - relaxation_rates

    center = tuple([0] * dim)
    t = cr.main_assignments[stencil.index(center)].rhs
    for rp in relaxation_rates:
        t = t.subs(rp, 1)

    for fa in pdf_symbols:
        t = t.subs(fa, 0)

    if 'force_terms' in sh:
        t = t.subs({ft: 0 for ft in sh['force_terms']})

    weight = t

    for u in sh['velocity']:
        weight = weight.subs(u, 0)
    weight = weight / sh['density']
    if weight == 0:
        return None
    return t / weight
