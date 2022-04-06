from collections import namedtuple

import sympy as sp
from sympy.core.cache import cacheit

from lbmpy.chapman_enskog.derivative import (
    chapman_enskog_derivative_expansion, chapman_enskog_derivative_recombination)
from lbmpy.moments import (
    discrete_moment, get_moment_indices, moment_matrix, non_aliased_moment,
    polynomial_to_exponent_representation)
from pystencils.cache import disk_cache
from pystencils.fd import (
    Diff, DiffOperator, expand_diff_full, expand_diff_linear, expand_diff_products,
    normalize_diff_order)
from pystencils.sympyextensions import normalize_product, symmetric_product


class ChapmanEnskogAnalysis:

    def __init__(self, method, constants=None):
        cqc = method.conserved_quantity_computation
        self._method = method
        self._moment_cache = LbMethodEqMoments(method)
        self.rho = cqc.density_symbol
        self.u = cqc.velocity_symbols
        self.t = sp.Symbol("t")
        self.epsilon = sp.Symbol("epsilon")

        taylored_lb_eq = get_taylor_expanded_lb_equation(dim=self._method.dim)
        self.equations_by_order = chapman_enskog_ansatz(taylored_lb_eq)

        # Taking moments
        c = sp.Matrix([expanded_symbol("c", subscript=i) for i in range(self._method.dim)])
        moments_until_order1 = [1] + list(c)
        moments_order2 = [c_i * c_j for c_i, c_j in symmetric_product(c, c)]

        symbolic_relaxation_rates = [rr for rr in method.relaxation_rates if isinstance(rr, sp.Symbol)]
        if constants is None:
            constants = set(symbolic_relaxation_rates)
        else:
            constants.update(symbolic_relaxation_rates)

        self.constants = constants

        o_eps_moments1 = [expand_diff_linear(self._take_and_insert_moments(self.equations_by_order[1] * moment),
                                             constants=constants)
                          for moment in moments_until_order1]
        o_eps_moments2 = [expand_diff_linear(self._take_and_insert_moments(self.equations_by_order[1] * moment),
                                             constants=constants)
                          for moment in moments_order2]
        o_eps_sq_moments1 = [expand_diff_linear(self._take_and_insert_moments(self.equations_by_order[2] * moment),
                                                constants=constants)
                             for moment in moments_until_order1]

        self._equationsWithHigherOrderMoments = [self._ce_recombine(ord1 * self.epsilon + ord2 * self.epsilon ** 2)
                                                 for ord1, ord2 in zip(o_eps_moments1, o_eps_sq_moments1)]

        self.higher_order_moments = compute_higher_order_moment_subs_dict(tuple(o_eps_moments1 + o_eps_moments2))

        # Match to Navier stokes
        compressible, pressure, sigma = match_to_navier_stokes(self._equationsWithHigherOrderMoments)
        self.compressible = compressible
        self.pressure_equation = pressure
        self._sigmaWithHigherOrderMoments = sigma
        self._sigma = sigma.subs(self.higher_order_moments).expand().applyfunc(self._ce_recombine)
        self._sigmaWithoutErrorTerms = remove_error_terms(self._sigma)

    def get_macroscopic_equations(self, substitute_higher_order_moments=False):
        if substitute_higher_order_moments:
            return [expand_diff_full(e.subs(self.higher_order_moments), constants=self.constants)
                    for e in self._equationsWithHigherOrderMoments]
        else:
            return self._equationsWithHigherOrderMoments

    def get_viscous_stress_tensor(self, substitute_higher_order_moments=True):
        if substitute_higher_order_moments:
            return self._sigma
        else:
            return self._sigmaWithHigherOrderMoments

    def _take_and_insert_moments(self, eq):
        eq = take_moments(eq)
        eq = substitute_collision_operator_moments(eq, self._moment_cache)
        return insert_moments(eq, self._moment_cache).expand()

    def _ce_recombine(self, expr):
        expr = chapman_enskog_derivative_recombination(expr, self.t, stop_order=3)
        for l in range(self._method.dim):
            expr = chapman_enskog_derivative_recombination(expr, l, stop_order=2)
        return expr

    def get_dynamic_viscosity(self):
        candidates = self.get_shear_viscosity_candidates()
        if len(candidates) != 1:
            raise ValueError("Could not find expression for kinematic viscosity. "
                             "Probably method does not approximate Navier Stokes.")
        return candidates.pop()

    def get_kinematic_viscosity(self):
        if self.compressible:
            return (self.get_dynamic_viscosity() / self.rho).expand()
        else:
            return self.get_dynamic_viscosity()

    def get_shear_viscosity_candidates(self):
        result = set()
        dim = self._method.dim
        for i, j in symmetric_product(range(dim), range(dim), with_diagonal=False):
            result.add(-sp.cancel(self._sigmaWithoutErrorTerms[i, j] / (Diff(self.u[i], j) + Diff(self.u[j], i))))
        return result

    def does_approximate_navier_stokes(self):
        """Returns a set of equations that are required in order for the method to approximate Navier Stokes equations
        up to second order"""
        conditions = {0}
        dim = self._method.dim
        assert dim > 1
        # Check that shear viscosity does not depend on any u derivatives - create conditions (equations) that
        # have to be fulfilled for this to be the case
        viscosity_reference = self._sigmaWithoutErrorTerms[0, 1].expand().coeff(Diff(self.u[0], 1))
        for i, j in symmetric_product(range(dim), range(dim), with_diagonal=False):
            term = self._sigmaWithoutErrorTerms[i, j]
            equal_cross_term_condition = sp.expand(term.coeff(Diff(self.u[i], j)) - viscosity_reference)
            term = term.subs({Diff(self.u[i], j): 0,
                              Diff(self.u[j], i): 0})

            conditions.add(equal_cross_term_condition)
            for k in range(dim):
                symmetric_term_condition = term.coeff(Diff(self.u[k], k))
                conditions.add(symmetric_term_condition)
            term = term.subs({Diff(self.u[k], k): 0 for k in range(dim)})
            conditions.add(term)

        bulk_candidates = list(self.get_bulk_viscosity_candidates(-viscosity_reference))
        if len(bulk_candidates) > 0:
            for i in range(1, len(bulk_candidates)):
                conditions.add(bulk_candidates[0] - bulk_candidates[i])

        return conditions

    def get_bulk_viscosity_candidates(self, viscosity=None):
        sigma = self._sigmaWithoutErrorTerms
        assert self._sigmaWithHigherOrderMoments.is_square
        result = set()
        if viscosity is None:
            viscosity = self.get_dynamic_viscosity()
        for i in range(sigma.shape[0]):
            bulk_term = sigma[i, i] + 2 * viscosity * Diff(self.u[i], i)
            bulk_term = bulk_term.expand()
            for d in bulk_term.atoms(Diff):
                bulk_term = bulk_term.collect(d)
                result.add(bulk_term.coeff(d))
                bulk_term = bulk_term.subs(d, 0)
            if bulk_term != 0:
                return set()
        if len(result) == 0:
            result.add(0)
        return result

    def get_bulk_viscosity(self):
        candidates = self.get_bulk_viscosity_candidates()
        if len(candidates) != 1:
            raise ValueError("Could not find expression for bulk viscosity. "
                             "Probably method does not approximate Navier Stokes.")

        viscosity = self.get_dynamic_viscosity()
        return (candidates.pop() + 2 * viscosity / 3).expand()

    def relaxation_rate_from_kinematic_viscosity(self, nu):
        kinematic_viscosity = self.get_kinematic_viscosity()
        solve_res = sp.solve(kinematic_viscosity - nu, kinematic_viscosity.atoms(sp.Symbol), dict=True)
        return solve_res[0]


# --------------------------------------------- Helper Functions -------------------------------------------------------


def expanded_symbol(name, subscript=None, superscript=None, **kwargs):
    if subscript is not None:
        name += "_{%s}" % (subscript,)
    if superscript is not None:
        name += "^{(%s)}" % (superscript,)
    return sp.Symbol(name, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------


# noinspection PyMethodOverriding,PyUnresolvedReferences
class CeMoment(sp.Symbol):
    def __new__(cls, name, *args, **kwargs):
        obj = CeMoment.__xnew_cached_(cls, name, *args, **kwargs)
        return obj

    def __new_stage2__(self, name, moment_tuple, superscript=-1):
        obj = super(CeMoment, self).__xnew__(self, name)
        obj._moment_tuple = moment_tuple
        while len(obj._moment_tuple) < 3:
            obj._moment_tuple = obj._moment_tuple + (0,)
        obj.superscript = superscript
        return obj

    __xnew__ = staticmethod(__new_stage2__)
    __xnew_cached_ = staticmethod(cacheit(__new_stage2__))

    def _hashable_content(self):
        super_class_contents = list(super(CeMoment, self)._hashable_content())
        return tuple(super_class_contents + [hash(repr(self.moment_tuple)), hash(repr(self.superscript))])

    @property
    def indices(self):
        return get_moment_indices(self.moment_tuple)

    @property
    def moment_tuple(self):
        return self._moment_tuple

    def __getnewargs__(self):
        return self.name, self.moment_tuple, self.superscript

    def _latex(self, *_):
        coord_str = []
        for i, comp in enumerate(self.moment_tuple):
            coord_str += [str(i)] * comp
        coord_str = "".join(coord_str)
        result = "{%s_{%s}" % (self.name, coord_str)
        if self.superscript >= 0:
            result += "^{(%d)}}" % (self.superscript,)
        else:
            result += "}"
        return result

    def __repr__(self):
        return "%s_(%d)_%s" % (self.name, self.superscript, self.moment_tuple)

    def __str__(self):
        return "%s_(%d)_%s" % (self.name, self.superscript, self.moment_tuple)


class LbMethodEqMoments:
    def __init__(self, lb_method):
        self._eq = tuple(e.rhs for e in lb_method.get_equilibrium().main_assignments)
        self._momentCache = dict()
        self._postCollisionMomentCache = dict()
        self._stencil = lb_method.stencil
        self._inverseMomentMatrix = moment_matrix(lb_method.moments, lb_method.stencil).inv()
        self._method = lb_method

    def __call__(self, ce_moment):
        return self.get_pre_collision_moment(ce_moment)

    def get_pre_collision_moment(self, ce_moment):
        assert ce_moment.superscript == 0, "Only equilibrium moments can be obtained with this function"
        if ce_moment not in self._momentCache:
            self._momentCache[ce_moment] = discrete_moment(self._eq, ce_moment.moment_tuple, self._stencil)
        return self._momentCache[ce_moment]

    def get_post_collision_moment(self, ce_moment, exponent=1, pre_collision_moment_name="\\Pi"):
        if (ce_moment, exponent) in self._postCollisionMomentCache:
            return self._postCollisionMomentCache[(ce_moment, exponent)]

        stencil = self._method.stencil
        moment2pdf = self._inverseMomentMatrix

        moment_tuple = ce_moment.moment_tuple

        moment_symbols = []
        for moment, (eq_value, rr) in self._method.relaxation_info_dict.items():
            if isinstance(moment, tuple):
                moment_symbols.append(-rr**exponent
                                      * CeMoment(pre_collision_moment_name, moment, ce_moment.superscript))
            else:
                exponent_repr = polynomial_to_exponent_representation(moment)
                moment_symbols.append(-rr**exponent * sum(coeff * CeMoment(pre_collision_moment_name, moment_tuple,
                                                                           ce_moment.superscript)
                                                          for coeff, moment_tuple in exponent_repr))
        moment_symbols = sp.Matrix(moment_symbols)
        post_collision_value = discrete_moment(tuple(moment2pdf * moment_symbols), moment_tuple, stencil)
        self._postCollisionMomentCache[(ce_moment, exponent)] = post_collision_value

        return post_collision_value

    def substitute_pre_collision_moments(self, expr, pre_collision_moment_name="\\Pi"):
        substitutions = {m: self.get_pre_collision_moment(m) for m in expr.atoms(CeMoment)
                         if m.superscript == 0 and m.name == pre_collision_moment_name}
        return expr.subs(substitutions)

    def substitute_post_collision_moments(self, expr,
                                          pre_collision_moment_name="\\Pi", post_collision_moment_name="\\Upsilon"):
        """Substitutes post-collision equilibrium moments.

        Args:
            expr: expression with fully expanded derivatives
            pre_collision_moment_name: post-collision moments are replaced by CeMoments with this name
            post_collision_moment_name: name of post-collision CeMoments

        Returns:
            expressions where equilibrium post-collision moments have been replaced
        """
        expr = sp.expand(expr)

        def visit(node, exponent):
            if node.func == sp.Pow:
                base, exp = node.args
                return visit(base, exp)
            elif isinstance(node, CeMoment) and node.name == post_collision_moment_name:
                return self.get_post_collision_moment(node, exponent, pre_collision_moment_name)
            else:
                return node**exponent if not node.args else node.func(*[visit(k, 1) for k in node.args])
        return visit(expr, 1)

    def substitute(self, expr, pre_collision_moment_name="\\Pi", post_collision_moment_name="\\Upsilon"):
        result = self.substitute_post_collision_moments(expr, pre_collision_moment_name, post_collision_moment_name)
        result = self.substitute_pre_collision_moments(result, pre_collision_moment_name)
        return result


def insert_moments(eqn, lb_method_moments, moment_name="\\Pi", use_solvability_conditions=True):
    substitutions = {}
    if use_solvability_conditions:
        substitutions.update({m: 0 for m in eqn.atoms(CeMoment)
                              if m.superscript > 0 and sum(m.moment_tuple) <= 1 and m.name == moment_name})

    substitutions.update({m: lb_method_moments(m) for m in eqn.atoms(CeMoment)
                          if m.superscript == 0 and m.name == moment_name})
    return eqn.subs(substitutions)


def substitute_collision_operator_moments(expr, lb_moment_computation, collision_op_moment_name='\\Upsilon',
                                          pre_collision_moment_name="\\Pi"):
    moments_to_replace = [m for m in expr.atoms(CeMoment) if m.name == collision_op_moment_name]
    subs_dict = {}
    for ce_moment in moments_to_replace:
        subs_dict[ce_moment] = lb_moment_computation.get_post_collision_moment(ce_moment, 1, pre_collision_moment_name)

    return expr.subs(subs_dict)


def take_moments(eqn, pdf_to_moment_name=(('f', '\\Pi'), ('\\Omega f', '\\Upsilon')), velocity_name='c',
                 max_expansion=5, use_one_neighborhood_aliasing=False):

    pdf_symbols = [tuple(expanded_symbol(name, superscript=i) for i in range(max_expansion))
                   for name, _ in pdf_to_moment_name]

    velocity_terms = tuple(expanded_symbol(velocity_name, subscript=i) for i in range(3))

    def determine_f_index(factor):
        FIndex = namedtuple("FIndex", ['moment_name', 'superscript'])
        for symbol_list_id, pdf_symbols_element in enumerate(pdf_symbols):
            try:
                return FIndex(pdf_to_moment_name[symbol_list_id][1], pdf_symbols_element.index(factor))
            except ValueError:
                pass
        return None

    def handle_product(product_term):
        f_index = None
        derivative_term = None
        c_indices = []
        rest = 1
        for factor in normalize_product(product_term):
            if isinstance(factor, Diff):
                assert f_index is None
                f_index = determine_f_index(factor.get_arg_recursive())
                derivative_term = factor
            elif factor in velocity_terms:
                c_indices += [velocity_terms.index(factor)]
            else:
                new_f_index = determine_f_index(factor)
                if new_f_index is None:
                    rest *= factor
                else:
                    assert not(new_f_index and f_index)
                    f_index = new_f_index

        moment_tuple = [0] * len(velocity_terms)
        for c_idx in c_indices:
            moment_tuple[c_idx] += 1
        moment_tuple = tuple(moment_tuple)

        if use_one_neighborhood_aliasing:
            moment_tuple = non_aliased_moment(moment_tuple)
        result = CeMoment(f_index.moment_name, moment_tuple, f_index.superscript)
        if derivative_term is not None:
            result = derivative_term.change_arg_recursive(result)
        result *= rest
        return result

    functions = sum(pdf_symbols, ())
    eqn = expand_diff_linear(eqn, functions).expand()

    if eqn.func == sp.Mul:
        return handle_product(eqn)
    else:
        assert eqn.func == sp.Add
        return sum(handle_product(t) for t in eqn.args)


def time_diff_selector(eq):
    return [d for d in eq.atoms(Diff) if d.target == sp.Symbol("t")]


def moment_selector(eq):
    return list(eq.atoms(CeMoment))


def diff_expand_normalizer(eq):
    return expand_diff_products(eq).expand()


def chain_solve_and_substitute(assignments, unknown_selector, normalizing_func=diff_expand_normalizer):
    """Takes a list (hierarchy) of equations and does the following:
       Loops over given equations and for every equation:
        - normalizes the equation with the provided normalizing_func
        - substitute symbols that have been already solved for
        - calls the unknown_selector function with an equation. This function should return a list of unknown symbols,
          and has to have length 0 or 1
        - if unknown was returned, the equation is solved for, and the pair (unknown-> solution)
          is entered into the dict
    """
    result_assignments = []
    subs_dict = {}
    for i, eq in enumerate(assignments):
        eq = normalizing_func(eq)
        eq = eq.subs(subs_dict)
        eq = normalizing_func(eq)
        result_assignments.append(eq)

        symbols_to_solve_for = unknown_selector(eq)
        if len(symbols_to_solve_for) == 0:
            continue
        assert len(symbols_to_solve_for) <= 1, "Unknown Selector return multiple unknowns - expected <=1\n" + str(
            symbols_to_solve_for)
        symbol_to_solve_for = symbols_to_solve_for[0]
        solve_res = sp.solve(eq, symbol_to_solve_for)
        assert len(solve_res) == 1, "Could not solve uniquely for unknown" + str(symbol_to_solve_for)
        subs_dict[symbol_to_solve_for] = normalizing_func(solve_res[0])
    return result_assignments, subs_dict


def count_vars(expr, variables):
    factor_list = normalize_product(expr)
    diffs_to_unpack = [e for e in factor_list if isinstance(e, Diff)]
    factor_list = [e for e in factor_list if not isinstance(e, Diff)]

    while diffs_to_unpack:
        d = diffs_to_unpack.pop()
        args = normalize_product(d.arg)
        for a in args:
            if isinstance(a, Diff):
                diffs_to_unpack.append(a)
            else:
                factor_list.append(a)

    result = 0
    for v in variables:
        result += factor_list.count(v)
    return result


def remove_higher_order_u(expr, order=1, u=sp.symbols("u_:3")):
    return sum(a for a in expr.args if count_vars(a, u) <= order)


def remove_error_terms(expr):
    rho_diffs_to_zero = {Diff(sp.Symbol("rho"), i): 0 for i in range(3)}
    expr = expr.subs(rho_diffs_to_zero)
    if isinstance(expr, sp.Matrix):
        expr = expr.applyfunc(remove_higher_order_u)
    else:
        expr = remove_higher_order_u(expr.expand())
    return sp.cancel(expr.expand())

# ----------------------------------------------------------------------------------------------------------------------


def get_taylor_expanded_lb_equation(pdf_symbol_name="f", pdfs_after_collision_operator="\\Omega f", velocity_name="c",
                                    dim=3, taylor_order=2, shift=True):
    dim_labels = [sp.Rational(i, 1) for i in range(dim)]

    c = sp.Matrix([expanded_symbol(velocity_name, subscript=label) for label in dim_labels])
    dt, t = sp.symbols("Delta_t t")
    pdf = sp.Symbol(pdf_symbol_name)
    collided_pdf = sp.Symbol(pdfs_after_collision_operator)

    dt_operator = DiffOperator(target=t)
    dx_operator = sp.Matrix([DiffOperator(target=l) for l in dim_labels])

    taylor_operator = sum(dt ** k * (dt_operator + c.dot(dx_operator)) ** k / sp.functions.factorial(k)
                          for k in range(1, taylor_order + 1))

    functions = [pdf, collided_pdf]
    eq_4_5 = taylor_operator - dt * collided_pdf
    applied_eq_4_5 = expand_diff_linear(DiffOperator.apply(eq_4_5, pdf, apply_to_constants=False), functions)

    if shift:
        operator = ((dt / 2) * (dt_operator + c.dot(dx_operator))).expand()
        op_times_eq_4_5 = expand_diff_linear(DiffOperator.apply(operator, applied_eq_4_5, apply_to_constants=False),
                                             functions).expand()
        op_times_eq_4_5 = normalize_diff_order(op_times_eq_4_5, functions)
        eq_4_7 = (applied_eq_4_5 - op_times_eq_4_5).subs(dt ** (taylor_order + 1), 0)
    else:
        eq_4_7 = applied_eq_4_5.subs(dt ** (taylor_order + 1), 0)

    eq_4_7 = eq_4_7.subs(dt, 1)
    return eq_4_7.expand()


def chapman_enskog_ansatz(equation, time_derivative_orders=(1, 3), spatial_derivative_orders=(1, 2),
                          pdfs=(['f', 0, 3], ['\\Omega f', 1, 3]), commutative=True):
    r"""Uses a Chapman Enskog Ansatz to expand given equation.

    Args:
        equation: equation to expand
        time_derivative_orders: tuple describing range for time derivative to expand
        spatial_derivative_orders: tuple describing range for spatial derivatives to expand
        pdfs: symbols to expand: sequence of triples (symbol_name, start_order, end_order)
        commutative: can be set to False to have non-commutative pdf symbols
    Returns:
        tuple mapping epsilon order to equation
    """
    t, eps = sp.symbols("t epsilon")

    # expand time derivatives
    if time_derivative_orders:
        equation = chapman_enskog_derivative_expansion(equation, t, eps, *time_derivative_orders)

    # expand spatial derivatives
    if spatial_derivative_orders:
        spatial_derivatives = [a for a in equation.atoms(Diff) if str(a.target) != 't']
        labels = set(a.target for a in spatial_derivatives)
        for label in labels:
            equation = chapman_enskog_derivative_expansion(equation, label, eps, *spatial_derivative_orders)

    # expand pdfs
    subs_dict = {}
    expanded_pdf_symbols = []

    max_expansion_order = spatial_derivative_orders[1] if spatial_derivative_orders else 10
    for pdf_name, start_order, stop_order in pdfs:
        if isinstance(pdf_name, sp.Symbol):
            pdf_name = pdf_name.name
        expanded_pdf_symbols += [expanded_symbol(pdf_name, superscript=i, commutative=commutative)
                                 for i in range(start_order, stop_order)]
        pdf_symbol = sp.Symbol(pdf_name, commutative=commutative)
        subs_dict[pdf_symbol] = sum(eps ** i * expanded_symbol(pdf_name, superscript=i, commutative=commutative)
                                    for i in range(start_order, stop_order))
        max_expansion_order = max(max_expansion_order, stop_order)
    equation = equation.subs(subs_dict)
    equation = expand_diff_linear(equation, functions=expanded_pdf_symbols).expand().collect(eps)
    result = {eps_order: equation.coeff(eps ** eps_order) for eps_order in range(1, 2 * max_expansion_order)}
    result[0] = equation.subs(eps, 0)
    return result


def match_to_navier_stokes(conservation_equations, rho=sp.Symbol("rho"), u=sp.symbols("u_:3"), t=sp.Symbol("t")):
    dim = len(conservation_equations) - 1
    u = u[:dim]
    funcs = u + (rho,)

    def diff_simplify(eq):
        variables = eq.atoms(CeMoment)
        variables.update(funcs)
        return expand_diff_products(expand_diff_linear(eq, variables)).expand()

    def match_continuity_eq(continuity_eq):
        continuity_eq = diff_simplify(continuity_eq)
        is_compressible = u[0] * Diff(rho, 0) in continuity_eq.args
        factor = rho if is_compressible else 1
        ref_continuity_eq = diff_simplify(Diff(rho, t) + sum(Diff(factor * u[i], i) for i in range(dim)))
        return ref_continuity_eq - continuity_eq, is_compressible

    def match_moment_eqs(moment_eqs, is_compressible):
        shear_and_pressure_eqs = []
        for i, mom_eq in enumerate(moment_eqs):
            factor = rho if is_compressible else 1
            ref = diff_simplify(Diff(factor * u[i], t) + sum(Diff(factor * u[i] * u[j], j) for j in range(dim)))
            shear_and_pressure_eqs.append(diff_simplify(moment_eqs[i]) - ref)

        # new_filtered pressure term
        coefficient_arg_sets = []
        for i, eq in enumerate(shear_and_pressure_eqs):
            coefficient_arg_sets.append(set())
            eq = eq.expand()
            assert eq.func == sp.Add
            for term in eq.args:
                if term.atoms(CeMoment):
                    continue
                candidate_list = [e for e in term.atoms(Diff) if e.target == i]
                if len(candidate_list) != 1:
                    continue
                coefficient_arg_sets[i].add((term / candidate_list[0], candidate_list[0].arg))
        pressure_terms = set.intersection(*coefficient_arg_sets)

        sigma_ = sp.zeros(dim)
        error_terms_ = []
        for i, shear_and_pressure_eq in enumerate(shear_and_pressure_eqs):
            eq_without_pressure = shear_and_pressure_eq - sum(coeff * Diff(arg, i) for coeff, arg in pressure_terms)
            for d in eq_without_pressure.atoms(Diff):
                eq_without_pressure = eq_without_pressure.collect(d)
                sigma_[i, d.target] += eq_without_pressure.coeff(d) * d.arg
                eq_without_pressure = eq_without_pressure.subs(d, 0)

            error_terms_.append(eq_without_pressure)
        pressure_ = [coeff * arg for coeff, arg in pressure_terms]

        return pressure_, sigma_, error_terms_

    continuity_error_terms, compressible = match_continuity_eq(conservation_equations[0])
    pressure, sigma, moment_error_terms = match_moment_eqs(conservation_equations[1:], compressible)

    error_terms = [continuity_error_terms] + moment_error_terms
    for et in error_terms:
        assert et == 0

    return compressible, pressure, sigma


@disk_cache
def compute_higher_order_moment_subs_dict(moment_equations):
    o_eps_without_time_diffs, time_diff_substitutions = chain_solve_and_substitute(moment_equations, time_diff_selector)
    moments_to_solve_for = set()
    pi_ab_equations = []
    for eq in o_eps_without_time_diffs:
        found_moments = moment_selector(eq)
        if found_moments:
            moments_to_solve_for.update(found_moments)
            pi_ab_equations.append(eq)
    return sp.solve(pi_ab_equations, moments_to_solve_for)
