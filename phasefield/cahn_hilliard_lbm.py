import sympy as sp
from lbmpy.creationfunctions import createLatticeBoltzmannFunction, createLatticeBoltzmannUpdateRule, \
    createLatticeBoltzmannAst
from lbmpy.methods.creationfunctions import createFromEquilibrium
from pystencils.sympyextensions import kroneckerDelta, multidimensionalSummation
from lbmpy.moments import getDefaultMomentSetForStencil
from lbmpy.maxwellian_equilibrium import getWeights


def createCahnHilliardEquilibrium(stencil, mu, gamma=1):
    """Returns LB equilibrium that solves the Cahn Hilliard equation

    ..math ::

        \partial_t \phi + \partial_i ( \phi v_i ) = M \nabla^2 \mu
    
    :param stencil: tuple of discrete directions
    :param mu: symbolic expression (field access) for the chemical potential
    :param gamma: tunable parameter affecting the second order equilibrium moment
    """
    weights = getWeights(stencil, c_s_sq=sp.Rational(1, 3))

    kd = kroneckerDelta

    def s(*args):
        for r in multidimensionalSummation(*args, dim=len(stencil[0])):
            yield r

    op = sp.Symbol("rho")
    v = sp.symbols("u_:%d" % (len(stencil[0]),))

    equilibrium = []
    for d, w in zip(stencil, weights):
        c_s = sp.sqrt(sp.Rational(1, 3))
        result = gamma * mu / (c_s ** 2)
        result += op * sum(d[i] * v[i] for i, in s(1)) / (c_s ** 2)
        result += op * sum(v[i] * v[j] * (d[i] * d[j] - c_s ** 2 * kd(i, j)) for i, j in s(2)) / (2 * c_s ** 4)
        equilibrium.append(w * result)

    rho = sp.Symbol("rho")
    equilibrium[0] = rho - sp.expand(sum(equilibrium[1:]))
    return tuple(equilibrium)


def createCahnHilliardLbFunction(stencil, relaxationRate, velocityField, mu, orderParameterOut,
                                 optimizationParams={}, gamma=1, srcFieldName='src', dstFieldName='dst'):
    """
    Update rule for a LB scheme that solves Cahn-Hilliard.

    :param stencil:
    :param relaxationRate: relaxation rate controls the mobility
    :param velocityField: velocity field (output from N-S LBM)
    :param mu: chemical potential field
    :param orderParameterOut: field where order parameter :math:`\phi` is written to
    :param optimizationParams: generic optimization parameters passed to creation functions
    :param gamma: tunable equilibrium parameter
    """
    equilibrium = createCahnHilliardEquilibrium(stencil, mu, gamma)
    rrRates = {m: relaxationRate for m in getDefaultMomentSetForStencil(stencil)}
    method = createFromEquilibrium(stencil, tuple(equilibrium), rrRates, compressible=True)

    updateRule = createLatticeBoltzmannUpdateRule(method, optimizationParams,
                                                  output={'density': orderParameterOut},
                                                  velocityInput=velocityField, fieldName=srcFieldName,
                                                  secondFieldName=dstFieldName)

    ast = createLatticeBoltzmannAst(updateRule=updateRule, optimizationParams=optimizationParams)
    return createLatticeBoltzmannFunction(ast=ast, optimizationParams=optimizationParams)

