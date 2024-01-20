"""
This module holds special transformations for simplifying the collision equations of moment-based methods.
All of these transformations operate on :class:`pystencils.AssignmentCollection` and need special
simplification hints, which are set by the MomentBasedLbMethod.
"""
import sympy as sp
from itertools import product

from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from pystencils import Assignment, AssignmentCollection
from pystencils.stencil import inverse_direction
from pystencils.simp.subexpression_insertion import insert_subexpressions, is_constant
from pystencils.sympyextensions import extract_most_common_factor, replace_second_order_products, subs_additive

from collections import defaultdict


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
    if len(set(sh['relaxation_rates'])) > 19:  # heuristics, works well if there is a small number of relaxation rates
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


def substitute_moments_in_conserved_quantity_equations(ac: AssignmentCollection):
    """
    Applied on an assignment collection containing both equations for raw moments
    and conserved quantities, this simplification attempts to express the conserved
    quantities in terms of the zeroth- and first-order moments.

    For example, :math:`rho =  f_0 + f_1 + ... + f_8` will be replaced by the zeroth-
    order moment: :math:`rho = m_{00}`

    Required simplification hints:
        - cq_symbols_to_moments: A dictionary mapping the conserved quantity symbols
          to their corresponding moment symbols (like `{rho : m_00, u_0 : m_10, u_1 : m_01}`).
    """
    sh = ac.simplification_hints
    if 'cq_symbols_to_moments' not in sh:
        raise ValueError("Simplification hint 'cq_symbols_to_moments' missing.")

    cq_symbols_to_moments = sh['cq_symbols_to_moments']
    if len(cq_symbols_to_moments) == 0:
        return ac

    required_symbols = list(cq_symbols_to_moments.keys()) + list(cq_symbols_to_moments.values())
    reduced_ac = ac.new_filtered(required_symbols).new_without_subexpressions()
    main_asm_dict = ac.main_assignments_dict
    subexp_dict = ac.subexpressions_dict
    reduced_assignments = reduced_ac.main_assignments_dict

    for cq_sym, moment_sym in cq_symbols_to_moments.items():
        moment_eq = reduced_assignments[moment_sym]
        assert moment_eq.count(cq_sym) == 0, f"Expressing conserved quantity {cq_sym} using moment {moment_sym} " \
                                             "would introduce a circular dependency."
        cq_eq = subs_additive(reduced_assignments[cq_sym], moment_sym, moment_eq)
        if cq_sym in main_asm_dict:
            main_asm_dict[cq_sym] = cq_eq
        else:
            assert moment_sym in subexp_dict, f"Cannot express subexpression {cq_sym}" \
                                              f" using main assignment {moment_sym}!"
            subexp_dict[cq_sym] = cq_eq

    main_assignments = [Assignment(lhs, rhs) for lhs, rhs in main_asm_dict.items()]
    subexpressions = [Assignment(lhs, rhs) for lhs, rhs in subexp_dict.items()]
    ac = ac.copy(main_assignments=main_assignments, subexpressions=subexpressions)
    ac.topological_sort()
    return ac.new_without_unused_subexpressions()


def split_pdf_main_assignments_by_symmetry(ac: AssignmentCollection):
    """
    Splits assignments to post-collision PDFs streaming in opposite directions
    into their symmetric and asymetric parts, which are extracted as subexpressions.
    Useful especially when computing PDF values from post-collision raw moments, where
    symmetric splitting can reduce the number of required additions by one half.

    Required simplification hints:
        - stencil: Velocity set of the LB method as a nested tuple of directions
        - post_collision_pdf_symbols: Sequence of symbols corresponding to the stencil velocities
    """
    sh = ac.simplification_hints
    if 'stencil' not in sh:
        raise ValueError("Symmetric splitting requires the stencil as a simplification hint.")
    if 'post_collision_pdf_symbols' not in sh:
        raise ValueError("Symmetric splitting requires the post-collision pdf symbols as a simplification hint.")

    stencil = sh['stencil']
    pdf_symbols = sh['post_collision_pdf_symbols']

    asm_dict = ac.main_assignments_dict
    subexpressions = ac.subexpressions
    done = set()
    subexp_to_symbol_dict = defaultdict(lambda: next(ac.subexpression_symbol_generator))
    half = sp.Rational(1, 2)
    for i, f in enumerate(pdf_symbols):
        if i in done:
            continue
        c = stencil[i]
        if all(cj == 0 for cj in c):
            continue
        c_inv = inverse_direction(c)
        i_inv = stencil.index(c_inv)
        f_inv = pdf_symbols[i_inv]
        done |= {i, i_inv}

        f_eq = asm_dict[f]
        f_inv_eq = asm_dict[f_inv]

        symmetric_part = half * (f_eq + f_inv_eq)
        asymmetric_part = half * (f_eq - f_inv_eq)

        symmetric_symb = subexp_to_symbol_dict[symmetric_part]
        asymmetric_symb = subexp_to_symbol_dict[asymmetric_part]

        asm_dict[f] = symmetric_symb + asymmetric_symb
        asm_dict[f_inv] = symmetric_symb - asymmetric_symb
    for subexp, sym in subexp_to_symbol_dict.items():
        subexpressions.append(Assignment(sym, subexp))
    main_assignments = [Assignment(lhs, rhs) for lhs, rhs in asm_dict.items()]
    return ac.copy(main_assignments=main_assignments, subexpressions=subexpressions)


def insert_pure_products(ac, symbols, **kwargs):
    """Inserts any subexpression whose RHS is a product containing exclusively factors
    from the given sequence of symbols."""
    def callback(exp):
        rhs = exp.rhs
        if isinstance(rhs, sp.Symbol) and rhs in symbols:
            return True
        elif isinstance(rhs, sp.Mul):
            if all((is_constant(arg) or (arg in symbols)) for arg in rhs.args):
                return True
        return False

    return insert_subexpressions(ac, callback, **kwargs)


def insert_conserved_quantity_products(cr, **kwargs):
    from lbmpy.moments import statistical_quantity_symbol as sq_sym
    from lbmpy.moment_transforms import PRE_COLLISION_MONOMIAL_RAW_MOMENT as m
    
    rho = cr.method.zeroth_order_equilibrium_moment_symbol
    u = cr.method.first_order_equilibrium_moment_symbols
    m000 = sq_sym(m, (0,) * cr.method.dim)
    symbols = (rho, m000) + u

    return insert_pure_products(cr, symbols)


def insert_half_force(cr, **kwargs):
    fmodel = cr.method.force_model
    if not fmodel:
        return cr
    force = fmodel.symbolic_force_vector
    force_exprs = set(c * f / 2 for c, f in product((1, -1), force))

    def callback(expr):
        return expr.rhs in force_exprs

    return insert_subexpressions(cr, callback, **kwargs)

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
    weight = weight.subs(sh['density_deviation'], 1)
    weight = weight.subs(sh['density'], 1)

    for u in sh['velocity']:
        weight = weight.subs(u, 0)
    # weight = weight / sh['density']
    if weight == 0:
        return None

    # t = t.subs(sh['density_deviation'], 0)

    return t / weight
