from collections import defaultdict

import sympy as sp

from pystencils.fd import Diff, expand_diff_full, expand_diff_linear, functional_derivative
from pystencils.sympyextensions import multidimensional_sum as multi_sum
from pystencils.sympyextensions import normalize_product, prod

order_parameter_symbol_name = "phi"
surface_tension_symbol_name = "tau"
interface_width_symbol = sp.Symbol("alpha")


def symmetric_symbolic_surface_tension(i, j):
    """Returns symbolic surface tension. The function is symmetric, i.e. interchanging i and j yields the same result.
    If both phase indices i and j are chosen equal, zero is returned"""
    if i == j:
        return 0
    index = (i, j) if i < j else (j, i)
    return sp.Symbol(f"{surface_tension_symbol_name}_{index[0]}_{index[1]}")


def symbolic_order_parameters(num_symbols):
    return sp.symbols(f"{order_parameter_symbol_name}_:{num_symbols}")


def free_energy_functional_3_phases(order_parameters=None, interface_width=interface_width_symbol, transformed=True,
                                    include_bulk=True, include_interface=True, expand_derivatives=True,
                                    kappa=sp.symbols("kappa_:3")):
    """Free Energy of ternary multi-component model :cite:`Semprebon2016`. """
    kappa_prime = tuple(interface_width ** 2 * k for k in kappa)
    c = sp.symbols("C_:3")

    bulk_free_energy = sum(k * C_i ** 2 * (1 - C_i) ** 2 / 2 for k, C_i in zip(kappa, c))
    surface_free_energy = sum(k * Diff(C_i) ** 2 / 2 for k, C_i in zip(kappa_prime, c))

    f = 0
    if include_bulk:
        f += bulk_free_energy
    if include_interface:
        f += surface_free_energy

    if not transformed:
        return f

    if order_parameters:
        rho, phi, psi = order_parameters
    else:
        rho, phi, psi = sp.symbols("rho phi psi")

    transformation_matrix = sp.Matrix([[1, 1, 1],
                                       [1, -1, 0],
                                       [0, 0, 1]])
    rho_def, phi_def, psi_def = transformation_matrix * sp.Matrix(c)
    order_param_to_concentration_relation = sp.solve([rho_def - rho, phi_def - phi, psi_def - psi], c)

    f = f.subs(order_param_to_concentration_relation)
    if expand_derivatives:
        f = expand_diff_linear(f, functions=order_parameters)

    return f, transformation_matrix


def free_energy_functional_n_phases_penalty_term(order_parameters, interface_width=interface_width_symbol, kappa=None,
                                                 penalty_term_factor=0.01):
    num_phases = len(order_parameters)
    if kappa is None:
        kappa = sp.symbols(f"kappa_:{num_phases}")
    if not hasattr(kappa, "__len__"):
        kappa = [kappa] * num_phases

    def f(x):
        return x ** 2 * (1 - x) ** 2

    bulk = sum(f(c) * k / 2 for c, k in zip(order_parameters, kappa))
    interface = sum(Diff(c) ** 2 / 2 * interface_width ** 2 * k
                    for c, k in zip(order_parameters, kappa))

    bulk_penalty_term = (1 - sum(c for c in order_parameters)) ** 2
    return bulk + interface + penalty_term_factor * bulk_penalty_term


def n_phases_correction_function(c, beta, power=2):
    return sp.Piecewise((-beta * c ** power, c < 0),
                        (-beta * (1 - c) ** power, c > 1),
                        (c ** 2 * (1 - c) ** 2, True))


def n_phases_correction_function_wrong(c, beta, power=2):
    return sp.Piecewise((-beta * c ** power, c < 0),
                        (-beta * (1 - c) ** power, c > 1),
                        (c ** 2 * (1 - c) ** power, True))


def n_phases_correction_function_sign_switch(c, beta):
    return sp.Piecewise((-beta * (c ** 2) * (1 - c) ** 2, c < 0),
                        (-beta * (c ** 2) * (1 - c) ** 2, c > 1),
                        (c ** 2 * (1 - c) ** 2, True))


def free_energy_functional_n_phases(num_phases=None, surface_tensions=symmetric_symbolic_surface_tension,
                                    interface_width=interface_width_symbol, order_parameters=None,
                                    include_bulk=True, include_interface=True, symbolic_lambda=False,
                                    symbolic_dependent_variable=False,
                                    f1=lambda c: c ** 2 * (1 - c) ** 2,
                                    f2=lambda c: c ** 2 * (1 - c) ** 2,
                                    triple_point_energy=0):
    r"""
    Returns a symbolic expression for the free energy of a system with N phases and
    specified surface tensions. The total free energy is the sum of a bulk and an interface component.

    .. math ::

        F_{bulk} = \int \frac{3}{\sqrt{2} \eta}
            \sum_{\substack{\alpha,\beta=0 \\ \alpha \neq \beta}}^{N-1}
            \frac{\tau(\alpha,\beta)}{2} \left[ f(\phi_\alpha) + f(\phi_\beta)
            - f(\phi_\alpha + \phi_\beta)  \right] \; d\Omega

        F_{interface} = \int \sum_{\alpha,\beta=0}^{N-2} \frac{\Lambda_{\alpha\beta}}{2}
                        \left( \nabla \phi_\alpha \cdot \nabla \phi_\beta \right)\; d\Omega

        \Lambda_{\alpha \beta} = \frac{3 \eta}{\sqrt{2}}  \left[ \tau(\alpha,N-1) + \tau(\beta,N-1) -
                                 \tau(\alpha,\beta)  \right]

        f(c) = c^2( 1-c)^2

    Args:
        num_phases: number of phases, called N above
        surface_tensions: surface tension function, called with two phase indices (two integers)
        interface_width: called :math:`\eta` above, controls the interface width
        order_parameters: explicitly
        f1: bulk energy is computed as f1(c_i) + f1(c_j) - f2(c_i + c_j)
        f2: see f2
        triple_point_energy: term multiplying c[i]*c[j]*c[k] for i < j < k
      Parameters useful for viewing / debugging the function
        include_bulk: if false no bulk term is added
        include_interface:if false no interface contribution is added
        symbolic_lambda: surface energy coefficient is represented by symbol, not in expanded form
        symbolic_dependent_variable: last phase variable is defined as 1-other_phase_vars, if this is set to True
                                     it is represented by phi_A for better readability
    """
    assert not (num_phases is None and order_parameters is None)
    if order_parameters is None:
        phi = symbolic_order_parameters(num_phases - 1)
    else:
        phi = order_parameters
        num_phases = len(phi) + 1

    if not symbolic_dependent_variable:
        phi = tuple(phi) + (1 - sum(phi),)
    else:
        phi = tuple(phi) + (sp.Symbol("phi_D"), )

    if callable(surface_tensions):
        surface_tensions = surface_tensions
    else:
        t = surface_tensions

        def surface_tensions(i, j):
            return t if i != j else 0

    # Compared to handwritten notes we scale the interface width parameter here to obtain the correct
    # equations for the interface profile and the surface tensions i.e. to pass tests
    # test_analyticInterfaceSolution and test_surfaceTensionDerivation
    interface_width *= sp.sqrt(2)

    def lambda_coeff(k, l):
        if symbolic_lambda:
            symbol_names = (k, l) if k < l else (l, k)
            return sp.Symbol(f"Lambda_{symbol_names[0]}{symbol_names[1]}")
        n = num_phases - 1
        if k == l:
            assert surface_tensions(l, l) == 0
        return 3 / sp.sqrt(2) * interface_width * (surface_tensions(k, n)
                                                   + surface_tensions(l, n) - surface_tensions(k, l))

    def bulk_term(i, j):
        return surface_tensions(i, j) / 2 * (f1(phi[i]) + f1(phi[j]) - f2(phi[i] + phi[j]))

    f_bulk = 3 / sp.sqrt(2) / interface_width * sum(bulk_term(i, j) for i, j in multi_sum(2, num_phases) if i != j)
    f_interface = sum(lambda_coeff(i, j) / 2 * Diff(phi[i]) * Diff(phi[j]) for i, j in multi_sum(2, num_phases - 1))

    for i in range(len(phi)):
        for j in range(i):
            for k in range(j):
                f_bulk += triple_point_energy * phi[i] * phi[j] * phi[k]

    result = 0
    if include_bulk:
        result += f_bulk
    if include_interface:
        result += f_interface
    return result


def separate_into_bulk_and_interface(free_energy):
    """Separates the bulk and interface parts of a free energy

    >>> F = free_energy_functional_n_phases(3)
    >>> bulk, inter = separate_into_bulk_and_interface(F)
    >>> assert sp.expand(bulk - free_energy_functional_n_phases(3, include_interface=False)) == 0
    >>> assert sp.expand(inter - free_energy_functional_n_phases(3, include_bulk=False)) == 0
    """
    free_energy = free_energy.expand()
    bulk_part = free_energy.subs({a: 0 for a in free_energy.atoms(Diff)})
    interface_part = free_energy - bulk_part
    return bulk_part, interface_part


def analytic_interface_profile(x, interface_width=interface_width_symbol):
    r"""Analytic expression for a 1D interface normal to x with given interface width.

    The following doctest shows that the returned analytical solution is indeed a solution of the ODE that we
    get from the condition :math:`\mu_0 = 0` (thermodynamic equilibrium) for a situation with only a single order
    parameter, i.e. at a transition between two phases.

    Examples:
        >>> num_phases = 4
        >>> x, phi = sp.Symbol("x"), symbolic_order_parameters(num_phases-1)
        >>> F = free_energy_functional_n_phases(order_parameters=phi)
        >>> mu = chemical_potentials_from_free_energy(F)
        >>> mu0 = mu[0].subs({p: 0 for p in phi[1:]})  # mu[0] as function of one order parameter only
        >>> solution = analytic_interface_profile(x)
        >>> solution_substitution = {phi[0]: solution, Diff(Diff(phi[0])): sp.diff(solution, x, x) }
        >>> sp.expand(mu0.subs(solution_substitution))  # inserting solution should solve the mu_0=0 equation
        0
    """
    return (1 + sp.tanh(x / (2 * interface_width))) / 2


def chemical_potentials_from_free_energy(free_energy, order_parameters=None):
    """Computes chemical potentials as functional derivative of free energy."""
    symbols = free_energy.atoms(sp.Symbol)
    if order_parameters is None:
        order_parameters = [s for s in symbols if s.name.startswith(order_parameter_symbol_name)]
        order_parameters.sort(key=lambda e: e.name)
        order_parameters = order_parameters[:-1]
    constants = [s for s in symbols if s not in order_parameters]
    return sp.Matrix([expand_diff_linear(functional_derivative(free_energy, op), constants=constants)
                      for op in order_parameters])


def force_from_phi_and_mu(order_parameters, dim, mu=None):
    if mu is None:
        mu = sp.symbols(f"mu_:{len(order_parameters)}")

    return sp.Matrix([sum(- c_i * Diff(mu_i, a) for c_i, mu_i in zip(order_parameters, mu))
                      for a in range(dim)])


def substitute_laplacian_by_sum(eq, dim):
    """Substitutes abstract Laplacian represented by ∂∂ by a sum over all dimensions
    i.e. in case of 3D: ∂∂ is replaced by ∂0∂0 + ∂1∂1 + ∂2∂2

    Args:
        eq: the term where the substitutions should be made
        dim: spatial dimension, in example above, 3
    """
    functions = [d.args[0] for d in eq.atoms(Diff)]
    substitutions = {Diff(Diff(op)): sum(Diff(Diff(op, i), i) for i in range(dim))
                     for op in functions}
    return expand_diff_full(eq.subs(substitutions))


def cosh_integral(f, var):
    """Integrates a function f that has exactly one cosh term, from -oo to oo, by
    substituting a new helper variable for the cosh argument"""
    cosh_term = list(f.atoms(sp.cosh))
    assert len(cosh_term) == 1
    integral = sp.Integral(f, var)
    transformed_int = integral.transform(cosh_term[0].args[0], sp.Symbol("u", real=True))
    return sp.integrate(transformed_int.args[0], (transformed_int.args[1][0], -sp.oo, sp.oo))


def symmetric_tensor_linearization(dim):
    next_idx = 0
    result_map = {}
    for idx in multi_sum(2, dim):
        idx = tuple(sorted(idx))
        if idx in result_map:
            continue
        else:
            result_map[idx] = next_idx
            next_idx += 1
    return result_map

# ----------------------------------------- Pressure Tensor ------------------------------------------------------------


def extract_gamma(free_energy, order_parameters):
    """Extracts parameters before the gradient terms"""
    result = defaultdict(lambda: 0)
    free_energy = free_energy.expand()
    assert free_energy.func == sp.Add
    for product in free_energy.args:
        product = normalize_product(product)
        diff_factors = [e for e in product if e.func == Diff]
        if len(diff_factors) == 0:
            continue

        if len(diff_factors) != 2:
            raise ValueError(f"Could not determine Λ because of term {str(product)}")

        indices = sorted([order_parameters.index(d.args[0]) for d in diff_factors])
        increment = prod(e for e in product if e.func != Diff)
        if diff_factors[0] == diff_factors[1]:
            increment *= 2
        result[tuple(indices)] += increment
    return result


def pressure_tensor_bulk_component(free_energy, order_parameters, bulk_chemical_potential=None):
    """Diagonal component of pressure tensor in bulk"""
    bulk_free_energy, _ = separate_into_bulk_and_interface(free_energy)
    if bulk_chemical_potential is None:
        mu_bulk = chemical_potentials_from_free_energy(bulk_free_energy, order_parameters)
    else:
        mu_bulk = bulk_chemical_potential
    return sum(c_i * mu_i for c_i, mu_i in zip(order_parameters, mu_bulk)) - bulk_free_energy


def pressure_tensor_interface_component(free_energy, order_parameters, dim, a, b):
    gamma = extract_gamma(free_energy, order_parameters)
    d = Diff
    result = 0
    for i, c_i in enumerate(order_parameters):
        for j, c_j in enumerate(order_parameters):
            t = d(c_i, a) * d(c_j, b) + d(c_i, b) * d(c_j, a)
            if a == b:
                t -= sum(d(c_i, g) * d(c_j, g) for g in range(dim))
                t -= sum(c_i * d(d(c_j, g), g) for g in range(dim))
                t -= sum(c_j * d(d(c_i, g), g) for g in range(dim))
            gamma_ij = gamma[(i, j)] if i < j else gamma[(j, i)]
            result += t * gamma_ij / 2
    return result


def pressure_tensor_interface_component_new(free_energy, order_parameters, dim, a, b):
    gamma = extract_gamma(free_energy, order_parameters)
    d = Diff
    result = 0
    for i, c_i in enumerate(order_parameters):
        for j, c_j in enumerate(order_parameters):
            t = 2 * d(c_i, a) * d(c_j, b)
            if a == b:
                t -= sum(d(c_i, g) * d(c_j, g) for g in range(dim))
                t -= 2 * sum(c_i * d(d(c_j, g), g) for g in range(dim))
            gamma_ij = gamma[(i, j)] if i < j else gamma[(j, i)]
            result += t * gamma_ij / 2
    return result


def pressure_tensor_from_free_energy(free_energy, order_parameters, dim, bulk_chemical_potential=None,
                                     include_bulk=True, include_interface=True):
    op = order_parameters

    def get_entry(i, j):
        p_if = pressure_tensor_interface_component(free_energy, op, dim, i, j) if include_interface else 0
        if include_bulk:
            p_b = pressure_tensor_bulk_component(free_energy, op, bulk_chemical_potential) if i == j else 0
        else:
            p_b = 0
        return sp.expand(p_if + p_b)

    result = sp.Matrix(dim, dim, get_entry)
    return result


def force_from_pressure_tensor(pressure_tensor, functions=None, pbs=None):
    assert len(pressure_tensor.shape) == 2 and pressure_tensor.shape[0] == pressure_tensor.shape[1]
    dim = pressure_tensor.shape[0]

    def force_component(b):
        r = -sum(Diff(pressure_tensor[a, b], a) for a in range(dim))
        r = expand_diff_full(r, functions=functions)

        if pbs is not None:
            r += 2 * Diff(pbs, b) * pbs

        return r

    return sp.Matrix([force_component(b) for b in range(dim)])


def pressure_tensor_bulk_sqrt_term(free_energy, order_parameters, density, c_s_sq=sp.Rational(1, 3)):
    pbs = sp.sqrt(sp.Abs(density * c_s_sq - pressure_tensor_bulk_component(free_energy, order_parameters)))
    return pbs
