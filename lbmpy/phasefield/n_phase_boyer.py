"""n-phase model according to
Hierarchy of consistent n-component Cahn-Hilliard systems by Franck Boyer, Sebastian Minjeaud
"""

from itertools import combinations

import sympy as sp

import pystencils.fd as fd
from lbmpy.phasefield.analytical import chemical_potentials_from_free_energy
from pystencils.cache import memorycache
from pystencils.fd import Diff
from pystencils.sympyextensions import prod


@memorycache(maxsize=8)
def diffusion_coefficients(surface_tensions):
    r"""Computes diffusion coefficients labeled ̅α in the paper"""
    assert surface_tensions.rows == surface_tensions.cols
    num_phases = surface_tensions.rows
    assert all(surface_tensions[k, k] == 0 for k in range(num_phases)), "Diagonal of surface tension matrix has to be 0"
    alpha_symbolic = sp.Matrix(num_phases, num_phases,
                               lambda i, j: sp.symbols(f"α_{i}{j}" if i < j else f"α_{j}{i}"))
    for i in range(num_phases):
        alpha_symbolic[i, i] = -sum(alpha_symbolic[i, j]
                                    for j in range(num_phases) if i != j)

    gamma_vector = sp.Matrix(sp.symbols(f"γ_:{num_phases}"))
    unit_vector = sp.Matrix(num_phases, 1, lambda a, b: 1)
    unit_matrix = sp.Matrix(num_phases, num_phases, lambda a, b: 1 if a == b else 0)

    lemma_2_1 = alpha_symbolic * surface_tensions - unit_matrix - gamma_vector * unit_vector.T

    eq_sys = [lemma_2_1[i, j] for i in range(num_phases) for j in range(num_phases)]
    unknowns = list(alpha_symbolic.atoms(sp.Symbol)) + list(gamma_vector.atoms(sp.Symbol))
    solution = sp.solve(eq_sys, unknowns)
    assert solution, "Equation system for diffusion coefficients could not be solved"
    return sp.ImmutableDenseMatrix(alpha_symbolic.subs(solution)), gamma_vector.subs(solution)


def capital_i(k, i):
    """third equation in section 3.1.1"""
    if not hasattr(i, '__len__'):
        i = list(range(i))
    for result in combinations(i, k):
        yield result


def capital_s(k, c, f):
    n = len(c)
    result = 0
    for i in capital_i(k, n):
        result += f(sum(c[index] for index in i))
    return result


def psi(k, c, f):
    """first equation in section 3.1.1"""
    n = len(c)
    assert k <= n
    result = 0
    for i in capital_i(k, n):
        for s in range(1, k + 1):
            for j in capital_i(s, i):
                result += (-1) ** (k - s) * f(sum(c[index] for index in j))
    return result


def h(u, v):
    return sp.Abs(v) * v / (sp.Abs(v) + u ** 2)


def compute_ab(f):
    c = sp.Dummy(positive=True)
    sqrt_f = sp.sqrt(f(c))
    max_sqrt_f = max(sqrt_f.subs(c, 0),
                     sqrt_f.subs(c, sp.Rational(1, 2)),
                     sqrt_f.subs(c, 1))
    sqrt_int = sp.integrate(sqrt_f, (c, 0, 1))
    return max_sqrt_f / sqrt_int, 1 / (2 * max_sqrt_f * sqrt_int)


def l_bar(free_energy, alpha_bar, c):
    result = []
    n = len(c)
    for i in range(n):
        result.append(-sum(alpha_bar[i, j] * sp.diff(free_energy, c[j]) for j in range(n)))
    return sp.Matrix(result)


def free_energy_interfacial(c, surface_tensions, a, epsilon):
    n = len(c)
    # this is different than in the paper: the sum is under the condition i < j, not i != j
    # otherwise the model does not correctly reduce to equation 1.5
    return -epsilon * a / 2 * sum(surface_tensions[i, j] * Diff(c[i]) * Diff(c[j])
                                  for i in range(n) for j in range(i))


def free_energy_bulk(capital_f, b, epsilon):
    return b / epsilon * capital_f


def lambda_equal_surface_tension(k, scalar_sigma, lambda1):
    k += 1
    one = sp.sympify(1)
    return lambda1 - scalar_sigma / 2 * sum(one / s for s in range(1, k))


def capital_f_bulk_equal_surface_tension(c, f, scalar_sigma, lambda1):
    n = len(c)
    return sum(lambda_equal_surface_tension(k, scalar_sigma, lambda1) * psi(k + 1, c, f)
               for k in range(n))


def capital_f0(c, surface_tension, f=lambda c: c ** 2 * (1 - c) ** 2):
    n = len(c)
    result = 0
    for j in range(n):
        for i in range(j):
            result += surface_tension[i, j] / 2 * (f(c[i]) + f(c[j]) - f(c[i] + c[j]))
    return result


def free_energy(c, epsilon, surface_tensions, stabilization_factor):
    alpha, _ = diffusion_coefficients(surface_tensions)

    capital_f = (capital_f0(c, surface_tensions)
                 + correction_g(c, surface_tensions)
                 + stabilization_factor * stabilization_term(c, alpha))

    def f(x):
        return x ** 2 * (1 - x) ** 2

    a, b = compute_ab(f)

    f_bulk = free_energy_bulk(capital_f, b, epsilon)
    f_if = free_energy_interfacial(c, surface_tensions, a, epsilon)
    return f_if + f_bulk


def symbolic_surface_tensions(num_phases):
    def creation_func(i, j):
        if i == j:
            return 0
        if j < i:
            i, j = j, i
        return sp.Symbol("sigma_{}{}".format(i, j))
    return sp.ImmutableDenseMatrix(num_phases, num_phases, creation_func)


def cahn_hilliard_fd(c, mu, alpha, dx=1, dt=1):
    discretize = fd.Discretization2ndOrder(dx=dx, dt=dt)
    result = []
    for c_i in c:
        pde = fd.transient(c_i) + fd.diffusion(alpha * mu, diffusion_coeff=1)
        result.append(discretize(pde))
    return result


# noinspection PyPep8Naming
class capital_h(sp.Function):
    nargs = (2,)

    def fdiff(self, argindex=1):
        u, v = self.args
        av = sp.Abs(v)
        zero_cond = sp.And(sp.Eq(u, 0), sp.Eq(v, 0))
        if argindex == 1:
            val = -2 * u * av * v / (av + u ** 2) ** 2
            return sp.Piecewise((0, zero_cond), (val, True))
        elif argindex == 2:
            val = (2 * av * u ** 2 + v ** 2) / (av + u ** 2) ** 2
            return sp.Piecewise((1, zero_cond), (val, True))
        else:
            raise sp.function.ArgumentIndexError(self, argindex)

    def doit(self, **hints):
        u, v = self.args
        av = sp.Abs(v)
        zero_cond = sp.And(sp.Eq(u, 0), sp.Eq(v, 0))
        return sp.Piecewise((0, zero_cond),
                            (av * v / (av + u**2), True))


def correction_g(c, surface_tensions, symbolic_coefficients=False):
    assert len(c) == surface_tensions.rows
    n = len(c)
    result = 0
    for i in capital_i(4, n):
        for s in i:
            reduced_i = tuple(e for e in i if e != s)
            if symbolic_coefficients:
                coeff = sp.Symbol("Lambda_{}{}{}{}{}".format(s, *i))
            else:
                coeff = capital_lambda(surface_tensions, reduced_i)[s]
            result += coeff * capital_h(c[s], prod(c[j] for j in i))
    return result


def capital_gamma(sigma, i, index_tuple):
    assert tuple(sorted(index_tuple)) == index_tuple
    j, k, m = index_tuple
    alpha, gamma = diffusion_coefficients(sigma)
    return -6 * (alpha[i, j] * (sigma[j, k] + sigma[j, m])
                 + alpha[i, k] * (sigma[j, k] + sigma[k, m])
                 + alpha[i, m] * (sigma[j, m] + sigma[k, m]) - gamma[i])


def capital_lambda(surface_tensions, index_tuple):
    n = surface_tensions.rows
    alpha, _ = diffusion_coefficients(surface_tensions)

    unknowns = [sp.Dummy() for _ in range(n)]
    for i in index_tuple:
        unknowns[i] = sp.oo

    eqs = []
    for i in range(n):
        if i in index_tuple:
            continue
        eq = - capital_gamma(surface_tensions, i, index_tuple)
        for j in range(n):
            if j in index_tuple:
                continue
            eq += alpha[i, j] * unknowns[j]
        eqs.append(eq)

    unknown_symbols = [unknowns[i] for i in range(n) if i not in index_tuple]
    solve_result = sp.solve(eqs, unknown_symbols)
    return sp.Matrix(unknowns).subs(solve_result)


def stabilization_term(c, alpha):
    n = alpha.rows
    result = sum(c[i]**2 * c[j]**2 * c[k]**2
                 for i in range(n) for j in range(i) for k in range(j))

    for j in range(n):
        for i in range(j):
            sub = 0
            for l in range(n):
                if l in (i, j):
                    continue
                for k in range(l):
                    if k in (i, j):
                        continue
                    sub += capital_theta(alpha, (i, j, k, l)) * capital_h(c[k], c[k] * c[l])
                    sub += capital_theta(alpha, (i, j, l, k)) * capital_h(c[l], c[k] * c[l])
            result -= c[i]**2 * c[j]**2 * sub
    return result


def capital_theta(alpha, index):
    assert len(index) == 4
    reduced_index = (index[0], index[1], index[3])
    return capital_theta_helper(alpha, reduced_index)[index[2]]


@memorycache(maxsize=512)
def capital_theta_helper(alpha, index):
    n = alpha.rows
    j0, k0, l0 = index
    unknowns = [sp.Dummy() for _ in range(n)]
    for i in index:
        unknowns[i] = sp.oo

    eqs = []
    for i in range(n):
        if i in index:
            continue
        eq = - 2 * alpha[i, l0]
        for j in range(n):
            if j in index:
                continue
            eq += alpha[i, j] * unknowns[j]
        eqs.append(eq)

    unknown_symbols = [unknowns[i] for i in range(n) if i not in index]
    solve_res = sp.solve(eqs, unknown_symbols)
    return sp.Matrix(unknowns).subs(solve_res)


# ------------------------------------------ Custom Piecewise simplification -------------------------------------------

def condition_variables(expression):
    """Returns set of variables that are used in Piecewise conditions"""
    result = set()

    def visit(expr):
        if isinstance(expr, sp.Piecewise):
            for e in expr.args:
                result.update(e[1].atoms(sp.Symbol))
        else:
            for a in expr.args:
                visit(a)
    visit(expression)
    return result


def simplify_zero_conditions(expression):
    cond_vars = condition_variables(expression)
    if not cond_vars:
        return expression

    result = []
    substitutions = {}
    for cond_var in cond_vars:
        zero_expr = expression.subs(cond_var, 0)
        condition_type = sp.LessThan if cond_var.is_nonnegative else sp.Eq
        result += [(simplify_zero_conditions(zero_expr), condition_type(cond_var, 0))]
        substitutions[cond_var] = sp.Dummy(nonzero=True)

    non_zero_expr = expression.subs(substitutions).subs({v: k for k, v in substitutions.items()})
    result += [(non_zero_expr, True)]
    return sp.Piecewise(*result)


def chemical_potential_n_phase_boyer(order_parameters, interface_width, surface_tensions, correction_factor,
                                     zero_threshold=0, assume_nonnegative=False):
    n = len(order_parameters)
    c = order_parameters
    if hasattr(surface_tensions, '__call__'):
        sigma = sp.ImmutableDenseMatrix(n, n, lambda i, j: surface_tensions(i, j) if i != j else 0)
    else:
        sigma = sp.ImmutableDenseMatrix(n, n, lambda i, j: surface_tensions[i, j] if i != j else 0)

    alpha, _ = diffusion_coefficients(sigma)
    capital_f = capital_f0(c, sigma) + correction_g(c, sigma) + correction_factor * stabilization_term(c, alpha)

    def f(c):
        return c ** 2 * (1 - c) ** 2

    a, b = compute_ab(f)

    fe_bulk = free_energy_bulk(capital_f, b, interface_width)
    fe_if = free_energy_interfacial(c, sigma, a, interface_width)

    mu_bulk = chemical_potentials_from_free_energy(fe_bulk, order_parameters)
    mu_bulk = sp.Matrix([simplify_zero_conditions(e.doit()) for e in mu_bulk])
    if zero_threshold != 0:
        substitutions = {sp.Eq(c_i, 0): sp.StrictLessThan(sp.Abs(c_i), zero_threshold) for c_i in c}
        mu_bulk = mu_bulk.subs(substitutions)

    if assume_nonnegative:
        substitutions = {c_i: sp.Dummy(nonnegative=True) for c_i in c}
        mu_bulk = mu_bulk.subs(substitutions).subs({v: k for k, v in substitutions.items()})

    mu_if = chemical_potentials_from_free_energy(fe_if, order_parameters)
    return fe_bulk, fe_if, mu_bulk, mu_if


# noinspection PyUnresolvedReferences
def plot_h():  # pragma: no cover
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Make data
    u = np.linspace(-1, 1, 100)
    v = np.linspace(0, 1, 100)
    x_grid, y_grid = np.meshgrid(u, v)
    us, vs = sp.symbols("u, v", real=True)
    h_vals = capital_h(us, vs)
    hl = sp.lambdify((us, vs), h_vals, modules=['numpy'])
    z_vals = hl(x_grid, y_grid)
    # Plot the surface
    ax.plot_surface(x_grid, y_grid, z_vals, color='b')
    plt.show()
