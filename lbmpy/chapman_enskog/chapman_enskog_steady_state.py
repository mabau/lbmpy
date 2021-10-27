import functools

import numpy as np
import sympy as sp

from lbmpy.chapman_enskog.chapman_enskog import (
    CeMoment, LbMethodEqMoments, chapman_enskog_ansatz, expanded_symbol, insert_moments,
    remove_higher_order_u, take_moments)
from pystencils.fd import (
    Diff, DiffOperator, collect_diffs, expand_diff_linear, normalize_diff_order)
from pystencils.sympyextensions import kronecker_delta, multidimensional_sum, normalize_product


class SteadyStateChapmanEnskogAnalysis:

    def __init__(self, method, force_model_class=None, order=4):
        self.method = method
        self.dim = method.dim
        self.order = order
        self.physical_variables = list(sp.Matrix(self.method.moment_equilibrium_values).atoms(sp.Symbol))  # rho, u..
        self.eps = sp.Symbol("epsilon")

        self.f_sym = sp.Symbol("f", commutative=False)
        self.f_syms = [expanded_symbol("f", superscript=i, commutative=False) for i in range(order + 1)]
        self.collision_op_sym = sp.Symbol("A", commutative=False)
        self.force_sym = sp.Symbol("F_q", commutative=False)
        self.velocity_syms = sp.Matrix([expanded_symbol("c", subscript=i, commutative=False) for i in range(self.dim)])

        self.F_q = [0] * len(self.method.stencil)
        self.force_model = None
        if force_model_class:
            acceleration_symbols = sp.symbols("a_:%d" % (self.dim,), commutative=False)
            self.physical_variables += acceleration_symbols
            self.force_model = force_model_class(acceleration_symbols)
            self.F_q = self.force_model(self.method)

        # Perform the analysis
        self.taylored_equation = self._create_taylor_expanded_equation()
        inserted_hierarchy, raw_hierarchy = self._create_pdf_hierarchy(self.taylored_equation)
        self.pdf_hierarchy = inserted_hierarchy
        self.pdf_hierarchy_raw = raw_hierarchy
        self.recombined_eq = self._recombine_pdfs(self.pdf_hierarchy)

        symbols_to_values = self._get_symbols_to_values_dict()
        self.continuity_equation = self._compute_continuity_equation(self.recombined_eq, symbols_to_values)
        self.momentum_equations = [self._compute_momentum_equation(self.recombined_eq, symbols_to_values, h)
                                   for h in range(self.dim)]

    def get_pdf_hierarchy(self, order, collision_operator_symbol=sp.Symbol("omega")):
        def substitute_non_commuting_symbols(eq):
            return eq.subs({a: sp.Symbol(a.name) for a in eq.atoms(sp.Symbol)})
        result = self.pdf_hierarchy[order].subs(self.collision_op_sym, collision_operator_symbol)
        result = normalize_diff_order(result, functions=(self.f_syms[0], self.force_sym))
        return substitute_non_commuting_symbols(result)

    def get_continuity_equation(self, only_order=None):
        return self._extract_order(self.continuity_equation, only_order)

    def get_momentum_equation(self, only_order=None):
        return [self._extract_order(e, only_order) for e in self.momentum_equations]

    def _extract_order(self, eq, order):
        if order is None:
            return eq
        elif order == 0:
            return eq.subs(self.eps, 0)
        else:
            return eq.coeff(self.eps ** order)

    def _create_taylor_expanded_equation(self):
        """
        Creates a generic, Taylor expanded lattice Boltzmann update equation with collision and force term.
        Collision operator and force terms are represented symbolically.
        """
        c = self.velocity_syms
        dx = sp.Matrix([DiffOperator(target=l) for l in range(self.dim)])

        differential_operator = sum((self.eps * c.dot(dx)) ** n / sp.factorial(n)
                                    for n in range(1, self.order + 1))
        taylor_expansion = DiffOperator.apply(differential_operator.expand(), self.f_sym)

        f_non_eq = self.f_sym - self.f_syms[0]
        return taylor_expansion + self.collision_op_sym * f_non_eq - self.eps * self.force_sym

    def _create_pdf_hierarchy(self, taylored_equation):
        """
        Expresses the expanded pdfs f^1, f^2, ..  as functions of the equilibrium f^0.
        Returns a list where element [1] is the equation for f^1 etc.
        """
        chapman_enskog_hierarchy = chapman_enskog_ansatz(taylored_equation, spatial_derivative_orders=None,
                                                         pdfs=(['f', 0, self.order + 1],), commutative=False)
        chapman_enskog_hierarchy = [chapman_enskog_hierarchy[i] for i in range(self.order + 1)]

        inserted_hierarchy = []
        raw_hierarchy = []
        substitution_dict = {}
        for ce_eq, f_i in zip(chapman_enskog_hierarchy, self.f_syms):
            new_eq = -1 / self.collision_op_sym * (ce_eq - self.collision_op_sym * f_i)
            raw_hierarchy.append(new_eq)
            new_eq = expand_diff_linear(new_eq.subs(substitution_dict), functions=self.f_syms + [self.force_sym])
            if new_eq:
                substitution_dict[f_i] = new_eq
            inserted_hierarchy.append(new_eq)

        return inserted_hierarchy, raw_hierarchy

    def _recombine_pdfs(self, pdf_hierarchy):
        return sum(pdf_hierarchy[i] * self.eps ** (i - 1) for i in range(1, self.order + 1))

    def _compute_continuity_equation(self, recombined_eq, symbols_to_values):
        return self._compute_moments(recombined_eq, symbols_to_values)

    def _compute_momentum_equation(self, recombined_eq, symbols_to_values, coordinate):
        eq = sp.expand(self.velocity_syms[coordinate] * recombined_eq)

        result = self._compute_moments(eq, symbols_to_values)
        if self.force_model and hasattr(self.force_model, 'equilibrium_velocity_shift'):
            compressible = self.method.conserved_quantity_computation.compressible
            shift = self.force_model.equilibrium_velocity_shift(sp.Symbol("rho") if compressible else 1)
            result += shift[coordinate]
        return result

    def _get_symbols_to_values_dict(self):
        result = {1 / self.collision_op_sym: self.method.inverse_collision_matrix,
                  self.force_sym: sp.Matrix(self.force_model(self.method)) if self.force_model else 0,
                  self.f_syms[0]: self.method.get_equilibrium_terms()}
        for i, c_i in enumerate(self.velocity_syms):
            result[c_i] = sp.Matrix([d[i] for d in self.method.stencil])

        return result

    def _compute_moments(self, recombined_eq, symbols_to_values):
        eq = recombined_eq.expand()
        assert eq.func is sp.Add

        new_products = []
        for product in eq.args:
            assert product.func is sp.Mul

            derivative = None

            new_prod = 1
            for arg in reversed(normalize_product(product)):
                if isinstance(arg, Diff):
                    assert derivative is None, "More than one derivative term in the product"
                    derivative = arg
                    arg = arg.get_arg_recursive()  # new argument is inner part of derivative

                if arg in symbols_to_values:
                    arg = symbols_to_values[arg]

                have_shape = hasattr(arg, 'shape') and hasattr(new_prod, 'shape')
                if have_shape and arg.shape == new_prod.shape and arg.shape[1] == 1:
                    # since sympy 1.9 sp.matrix_multiply_elementwise does not work anymore in this case
                    new_prod = sp.Matrix(np.multiply(new_prod, arg))
                else:
                    new_prod = arg * new_prod
                if new_prod == 0:
                    break

            if new_prod == 0:
                continue

            new_prod = sp.expand(sum(new_prod))

            if derivative is not None:
                new_prod = derivative.change_arg_recursive(new_prod)

            new_products.append(new_prod)

        return normalize_diff_order(expand_diff_linear(sp.Add(*new_products), functions=self.physical_variables))


# ----------------------------------------------------------------------------------------------------------------------


class SteadyStateChapmanEnskogAnalysisSRT:
    """Less general but simpler, steady state Chapman Enskog analysis for SRT methods"""

    def __init__(self, method, order=4):
        self.method = method
        dim = method.dim
        moment_computation = LbMethodEqMoments(method)

        eps, collision_operator, f, dt = sp.symbols("epsilon B f Delta_t")
        self.dt = dt
        expanded_pdf_symbols = [expanded_symbol("f", superscript=i) for i in range(0, order + 1)]
        feq = expanded_pdf_symbols[0]
        c = sp.Matrix([expanded_symbol("c", subscript=i) for i in range(dim)])
        dx = sp.Matrix([DiffOperator(target=l) for l in range(dim)])
        differential_operator = sum((dt * eps * c.dot(dx)) ** n / sp.factorial(n) for n in range(1, order + 1))
        taylor_expansion = DiffOperator.apply(differential_operator.expand(), f, apply_to_constants=False)
        eps_dict = chapman_enskog_ansatz(taylor_expansion,
                                         spatial_derivative_orders=None,  # do not expand the differential operator
                                         pdfs=(['f', 0, order + 1],))  # expand only the 'f' terms

        self.scale_hierarchy = [-collision_operator * eps_dict[i] for i in range(0, order + 1)]
        self.scale_hierarchy_raw = self.scale_hierarchy.copy()

        expanded_pdfs = [feq, self.scale_hierarchy[1]]
        subs_dict = {expanded_pdf_symbols[1]: self.scale_hierarchy[1]}

        for i in range(2, len(self.scale_hierarchy)):
            eq = self.scale_hierarchy[i].subs(subs_dict)
            eq = expand_diff_linear(eq, functions=expanded_pdf_symbols)
            eq = normalize_diff_order(eq, functions=expanded_pdf_symbols)
            subs_dict[expanded_pdf_symbols[i]] = eq
            expanded_pdfs.append(eq)
        self.scale_hierarchy = expanded_pdfs

        constants = sp.Matrix(method.relaxation_rates).atoms(sp.Symbol)
        recombined = -sum(self.scale_hierarchy[n] for n in range(1, order + 1))  # Eq 18a
        recombined = sp.cancel(recombined / (dt * collision_operator)).expand()  # cancel common factors

        def handle_postcollision_values(expr):
            expr = expr.expand()
            assert isinstance(expr, sp.Add)
            result = 0
            for summand in expr.args:

                moment = summand.atoms(CeMoment)
                moment = moment.pop()
                collision_operator_exponent = normalize_product(summand).count(collision_operator)
                if collision_operator_exponent == 0:
                    result += summand
                else:
                    substitutions = {
                        collision_operator: 1,
                        moment: -moment_computation.get_post_collision_moment(moment, -collision_operator_exponent),
                    }
                    result += summand.subs(substitutions)

            return result

        # Continuity equation (mass transport)
        cont_eq = take_moments(recombined, max_expansion=(order + 1) * 2)
        cont_eq = handle_postcollision_values(cont_eq)
        cont_eq = expand_diff_linear(cont_eq, constants=constants).expand().collect(dt)
        self.continuity_equation_with_moments = cont_eq
        cont_eq = insert_moments(cont_eq, moment_computation, use_solvability_conditions=False)
        cont_eq = expand_diff_linear(cont_eq, constants=constants).expand().collect(dt)
        self.continuity_equation = cont_eq

        # Momentum equation (momentum transport)
        self.momentum_equations_with_moments = []
        self.momentum_equations = []
        for h in range(dim):
            mom_eq = take_moments(recombined * c[h], max_expansion=(order + 1) * 2)
            mom_eq = handle_postcollision_values(mom_eq)
            mom_eq = expand_diff_linear(mom_eq, constants=constants).expand().collect(dt)
            self.momentum_equations_with_moments.append(mom_eq)
            mom_eq = insert_moments(mom_eq, moment_computation, use_solvability_conditions=False)
            mom_eq = expand_diff_linear(mom_eq, constants=constants).expand().collect(dt)
            self.momentum_equations.append(mom_eq)

    def get_continuity_equation(self, order):
        if order == 0:
            result = self.continuity_equation.subs(self.dt, 0)
        else:
            result = self.continuity_equation.coeff(self.dt ** order)
        return collect_diffs(result)

    def get_momentum_equation(self, coordinate, order):
        if order == 0:
            result = self.momentum_equations[coordinate].subs(self.dt, 0)
        else:
            result = self.momentum_equations[coordinate].coeff(self.dt ** order)
        return collect_diffs(result)

    def determine_viscosities(self, coordinate):
        """Matches the first order term of the momentum equation to Navier stokes.

        Automatically neglects higher order velocity terms and rho derivatives

        The bulk viscosity is predicted differently than by the normal Navier Stokes analysis...why??

        Args:
            coordinate: which momentum equation to use i.e. x,y or z, to approximate Navier Stokes
                        all have to return the same result
        """
        dim = self.method.dim

        def d(arg, *args):
            """Shortcut to create nested derivatives"""
            assert arg is not None
            args = sorted(args, reverse=True, key=lambda e: e.name if isinstance(e, sp.Symbol) else e)
            res = arg
            for i in args:
                res = Diff(res, i)
            return res

        s = functools.partial(multidimensional_sum, dim=dim)
        kd = kronecker_delta

        eta, eta_b = sp.symbols("nu nu_B")
        u = sp.symbols("u_:3")[:dim]
        a = coordinate
        navier_stokes_ref = eta * sum(d(u[a], b, b) + d(u[b], b, a)
                                      for b, in s(1)) + (eta_b - 2 * eta / 3) * sum(d(u[g], b, g) * kd(a, b)
                                                                                    for b, g in s(2))
        navier_stokes_ref = -navier_stokes_ref.expand()

        first_order_terms = self.get_momentum_equation(coordinate, order=1)
        first_order_terms = remove_higher_order_u(first_order_terms)
        first_order_terms = expand_diff_linear(first_order_terms, constants=[sp.Symbol("rho")])

        match_coeff_equations = []
        for item in navier_stokes_ref.atoms(Diff):
            match_coeff_equations.append(navier_stokes_ref.coeff(item) - first_order_terms.coeff(item))
        return sp.solve(match_coeff_equations, [eta, eta_b])
