import sympy as sp
from pystencils.sympyextensions import normalize_product
from lbmpy.chapman_enskog.derivative import Diff, DiffOperator, expand_using_linearity, normalize_diff_order
from lbmpy.chapman_enskog.chapman_enskog import expanded_symbol, use_chapman_enskog_ansatz


class SteadyStateChapmanEnskogAnalysis(object):

    def __init__(self, method, force_model_class=None, order=4):
        self.method = method
        self.dim = method.dim
        self.order = order
        self.physicalVariables = list(sp.Matrix(self.method.moment_equilibrium_values).atoms(sp.Symbol))  # rho, u..
        self.eps = sp.Symbol("epsilon")

        self.f_sym = sp.Symbol("f", commutative=False)
        self.f_syms = [expanded_symbol("f", superscript=i, commutative=False) for i in range(order + 1)]
        self.collisionOpSym = sp.Symbol("A", commutative=False)
        self.force_sym = sp.Symbol("F_q", commutative=False)
        self.velocity_syms = sp.Matrix([expanded_symbol("c", subscript=i, commutative=False) for i in range(self.dim)])

        self.F_q = [0] * len(self.method.stencil)
        self.force_model = None
        if force_model_class:
            acceleration_symbols = sp.symbols("a_:%d" % (self.dim,), commutative=False)
            self.physicalVariables += acceleration_symbols
            self.force_model = force_model_class(acceleration_symbols)
            self.F_q = self.force_model(self.method)

        # Perform the analysis
        self.tayloredEquation = self._create_taylor_expanded_equation()
        inserted_hierarchy, raw_hierarchy = self._create_pdf_hierarchy(self.tayloredEquation)
        self.pdfHierarchy = inserted_hierarchy
        self.pdfHierarchyRaw = raw_hierarchy
        self.recombinedEq = self._recombine_pdfs(self.pdfHierarchy)

        symbols_to_values = self._get_symbols_to_values_dict()
        self.continuityEquation = self._compute_continuity_equation(self.recombinedEq, symbols_to_values)
        self.momentumEquations = [self._compute_momentum_equation(self.recombinedEq, symbols_to_values, h)
                                  for h in range(self.dim)]

    def get_pdf_hierarchy(self, order, collision_operator_symbol=sp.Symbol("omega")):
        def substitute_non_commuting_symbols(eq):
            return eq.subs({a: sp.Symbol(a.name) for a in eq.atoms(sp.Symbol)})
        result = self.pdfHierarchy[order].subs(self.collisionOpSym, collision_operator_symbol)
        result = normalize_diff_order(result, functions=(self.f_syms[0], self.force_sym))
        return substitute_non_commuting_symbols(result)

    def get_continuity_equation(self, only_order=None):
        return self._extract_order(self.continuityEquation, only_order)

    def get_momentum_equation(self, only_order=None):
        return [self._extract_order(e, only_order) for e in self.momentumEquations]

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
        return taylor_expansion + self.collisionOpSym * f_non_eq - self.eps * self.force_sym

    def _create_pdf_hierarchy(self, taylored_equation):
        """
        Expresses the expanded pdfs f^1, f^2, ..  as functions of the equilibrium f^0.
        Returns a list where element [1] is the equation for f^1 etc.
        """
        chapman_enskog_hierarchy = use_chapman_enskog_ansatz(taylored_equation, spatial_derivative_orders=None,
                                                             pdfs=(['f', 0, self.order + 1],), commutative=False)
        chapman_enskog_hierarchy = [chapman_enskog_hierarchy[i] for i in range(self.order + 1)]

        inserted_hierarchy = []
        raw_hierarchy = []
        substitution_dict = {}
        for ceEq, f_i in zip(chapman_enskog_hierarchy, self.f_syms):
            new_eq = -1 / self.collisionOpSym * (ceEq - self.collisionOpSym * f_i)
            raw_hierarchy.append(new_eq)
            new_eq = expand_using_linearity(new_eq.subs(substitution_dict), functions=self.f_syms + [self.force_sym])
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
        if self.force_model and hasattr(self.force_model, 'equilibriumVelocityShift'):
            compressible = self.method.conserved_quantity_computation.compressible
            shift = self.force_model.equilibriumVelocityShift(sp.Symbol("rho") if compressible else 1)
            result += shift[coordinate]
        return result

    def _get_symbols_to_values_dict(self):
        result = {1 / self.collisionOpSym: self.method.inverse_collision_matrix,
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
                    new_prod = sp.matrix_multiply_elementwise(new_prod, arg)
                else:
                    new_prod = arg * new_prod

            new_prod = sp.expand(sum(new_prod))

            if derivative is not None:
                new_prod = derivative.change_arg_recursive(new_prod)

            new_products.append(new_prod)

        return normalize_diff_order(expand_using_linearity(sp.Add(*new_products), functions=self.physicalVariables))
