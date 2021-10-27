import sympy as sp

from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.methods.creationfunctions import create_from_equilibrium
from pystencils.sympyextensions import kronecker_delta, multidimensional_sum


def cahn_hilliard_lb_method(stencil, mu, relaxation_rate=sp.Symbol("omega"), gamma=1):
    r"""Returns LB equilibrium that solves the Cahn Hilliard equation.

    ..math ::

        \partial_t \phi + \partial_i ( \phi v_i ) = M \nabla^2 \mu

    Args:
        stencil: tuple of discrete directions
        mu: symbolic expression (field access) for the chemical potential
        relaxation_rate: relaxation rate of method
        gamma: tunable parameter affecting the second order equilibrium moment
    """
    weights = get_weights(stencil, c_s_sq=sp.Rational(1, 3))

    kd = kronecker_delta

    def s(*args):
        for r in multidimensional_sum(*args, dim=len(stencil[0])):
            yield r

    op = sp.Symbol("rho")
    v = sp.symbols(f"u_:{stencil.D}")

    equilibrium = []
    for d, w in zip(stencil, weights):
        c_s = sp.sqrt(sp.Rational(1, 3))
        result = gamma * mu / (c_s ** 2)
        result += op * sum(d[i] * v[i] for i, in s(1)) / (c_s ** 2)
        result += op * sum(v[i] * v[j] * (d[i] * d[j] - c_s ** 2 * kd(i, j)) for i, j in s(2)) / (2 * c_s ** 4)
        equilibrium.append(w * result)

    rho = sp.Symbol("rho")
    equilibrium[0] = rho - sp.expand(sum(equilibrium[1:]))
    return create_from_equilibrium(stencil, tuple(equilibrium), relaxation_rate, compressible=True)
