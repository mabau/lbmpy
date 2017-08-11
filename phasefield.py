import sympy as sp
from pystencils.equationcollection.equationcollection import EquationCollection
from pystencils.sympyextensions import kroneckerDelta, multidimensionalSummation
from lbmpy.maxwellian_equilibrium import getWeights
from lbmpy.chapman_enskog.derivative import expandUsingLinearity, Diff
from lbmpy.methods.creationfunctions import createFromEquilibrium
from lbmpy.moments import getDefaultMomentSetForStencil
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.updatekernels import createStreamPullCollideKernel


def functionalDerivative(functional, v, constants=None):
    """
    - assumes that gradients are represented by Diff() node (from Chapman Enskog module)    
    - Diff(Diff(r)) represents the divergence of r
    """
    functional = expandUsingLinearity(functional, constants=constants)
    diffs = functional.atoms(Diff)

    diffV = Diff(v)
    assert diffV in diffs  # not necessary in general, but for this use case this should be true

    nonDiffPart = functional.subs({d: 0 for d in diffs})

    partialF_partialV = sp.diff(nonDiffPart, v)

    dummy = sp.Dummy()
    partialF_partialGradV = functional.subs(diffV, dummy).diff(dummy).subs(dummy, diffV)

    result = partialF_partialV - Diff(partialF_partialGradV)
    return expandUsingLinearity(result, constants=constants)


def discreteLaplace(field, index, dx):
    dim = field.spatialDimensions
    count = 0
    result = 0
    for d in range(dim):
        for offset in (-1, 1):
            count += 1
            result += field.neighbor(d, offset)(index)

    result -= count * field.center()(index)
    result /= dx ** 2
    return result


def createCahnHilliardEquilibrium(stencil, mu, gamma=1):
    weights = getWeights(stencil, c_s_sq=sp.Rational(1, 3))

    kd = kroneckerDelta

    def s(*args):
        for r in multidimensionalSummation(*args, dim=len(stencil[0])):
            yield r

    op = sp.Symbol("rho")
    v = sp.symbols("v_:%d" % (len(stencil[0]),))

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


def createCahnHilliardUpdateRule(stencil, relaxationRate, pdfArr, velocityField, mu, orderParameterOut, gamma=1):
    equilibrium = createCahnHilliardEquilibrium(stencil, mu, gamma)
    v = sp.symbols("v_:%d" % (len(stencil[0]),))
    op = sp.Symbol("rho")

    rrRates = {m: relaxationRate for m in getDefaultMomentSetForStencil(stencil)}
    method = createFromEquilibrium(stencil, tuple(equilibrium), rrRates, compressible=True)

    inputEqs = [sp.Eq(op, sum(method.preCollisionPdfSymbols))]
    inputEqs += [sp.Eq(v_i, velocityField(i)) for i, v_i in enumerate(v)]

    collisionRule = method.getCollisionRule(EquationCollection(inputEqs, []))
    simplification = createSimplificationStrategy(method)
    collisionRule = simplification(collisionRule)
    collisionRule.mainEquations.append(sp.Eq(orderParameterOut, op))
    updateRule = createStreamPullCollideKernel(collisionRule, numpyField=pdfArr)
    updateRule.collisionRule = collisionRule
    updateRule.method = method
    return updateRule
