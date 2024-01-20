import sympy as sp

from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.equilibrium import GenericDiscreteEquilibrium
from lbmpy.methods import DensityVelocityComputation
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

    compressible = True
    zero_centered = False
    cqc = DensityVelocityComputation(stencil, compressible, zero_centered)
    rho = cqc.density_symbol
    v = cqc.velocity_symbols

    equilibrium_terms = []
    for d, w in zip(stencil, weights):
        c_s = sp.sqrt(sp.Rational(1, 3))
        result = gamma * mu / (c_s ** 2)
        result += rho * sum(d[i] * v[i] for i, in s(1)) / (c_s ** 2)
        result += rho * sum(v[i] * v[j] * (d[i] * d[j] - c_s ** 2 * kd(i, j)) for i, j in s(2)) / (2 * c_s ** 4)
        equilibrium_terms.append(w * result)

    equilibrium_terms[0] = rho - sp.expand(sum(equilibrium_terms[1:]))
    equilibrium = GenericDiscreteEquilibrium(stencil, equilibrium_terms, rho, v, deviation_only=False)
    
    return create_from_equilibrium(stencil, equilibrium, cqc, relaxation_rate, zero_centered=zero_centered)
