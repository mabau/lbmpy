from collections import OrderedDict
from functools import reduce
import itertools
import operator
import sympy as sp
from lbmpy.methods.creationfunctions import createWithDiscreteMaxwellianEqMoments
from pystencils.transformations import fastSubs
from lbmpy.stencils import getStencil
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.moments import momentsUpToComponentOrder, MOMENT_SYMBOLS, momentsOfOrder, \
    exponentsToPolynomialRepresentations


def discreteEntropy(function, f_eq):
    return -sum([f_i * sp.ln(f_i / w_i) for f_i, w_i in zip(function, f_eq)])


def discreteApproxEntropy(function, f_eq):
    return -sum([f_i * ((f_i / w_i)-1) for f_i, w_i in zip(function, f_eq)])


def splitUpdateEquationsInDeltasPlusRest(updateEqsRhs, relaxationRate):
    deltas = [ue.expand().collect(relaxationRate).coeff(relaxationRate)
              for ue in updateEqsRhs]
    rest = [ue.expand().collect(relaxationRate) - relaxationRate * delta for ue, delta in zip(updateEqsRhs, deltas)]

    return rest, deltas


def findEntropyMaximizingOmega(omega_s, f_eq,  ds, dh):
    dsdh = sum([ds_i * dh_i / f_eq_i for ds_i, dh_i, f_eq_i in zip(ds, dh, f_eq)])
    dhdh = sum([dh_i * dh_i / f_eq_i for dh_i, f_eq_i in zip(dh, f_eq)])
    return 1 - ((omega_s - 1) * dsdh / dhdh)


def determineRelaxationRateByEntropyCondition(updateRule, omega_s, omega_h):
    stencil = updateRule.method.stencil
    Q = len(stencil)
    fSymbols = updateRule.method.preCollisionPdfSymbols

    updateEqsRhs = [e.rhs for e in updateRule.mainEquations]
    _, ds = splitUpdateEquationsInDeltasPlusRest(updateEqsRhs, omega_s)
    _, dh = splitUpdateEquationsInDeltasPlusRest(updateEqsRhs, omega_h)
    dsSymbols = [sp.Symbol("entropicDs_%d" % (i,)) for i in range(Q)]
    dhSymbols = [sp.Symbol("entropicDh_%d" % (i,)) for i in range(Q)]
    feqSymbols = [sp.Symbol("entropicFeq_%d" % (i,)) for i in range(Q)]

    subexprs = [sp.Eq(a, b) for a, b in zip(dsSymbols, ds)] + \
               [sp.Eq(a, b) for a, b in zip(dhSymbols, dh)] + \
               [sp.Eq(a, f_i + ds_i + dh_i) for a, f_i, ds_i, dh_i in zip(feqSymbols, fSymbols, dsSymbols, dhSymbols)]

    optimalOmegaH = findEntropyMaximizingOmega(omega_s, feqSymbols, dsSymbols, dhSymbols)

    subexprs += [sp.Eq(omega_h, optimalOmegaH)]

    newUpdateEquations = []
    for updateEq in updateRule.mainEquations:
        index = updateRule.method.postCollisionPdfSymbols.index(updateEq.lhs)
        newEq = sp.Eq(updateEq.lhs, fSymbols[index] + omega_s * dsSymbols[index] + omega_h * dhSymbols[index])
        newUpdateEquations.append(newEq)
    return updateRule.copy(newUpdateEquations, updateRule.subexpressions + subexprs)


def decompositionByRelaxationRate(updateRule, relaxationRate):
    lm = updateRule.latticeModel
    stencil = lm.stencil

    affineTerms = [0] * len(stencil)
    linearTerms = [0] * len(stencil)
    quadraticTerms = [0] * len(stencil)

    for updateEquation in updateRule.updateEquations:
        index = lm.pdfDestinationSymbols.index(updateEquation.lhs)
        rhs = updateEquation.rhs
        linearTerms[index] = rhs.coeff(relaxationRate)
        quadraticTerms[index] = rhs.coeff(relaxationRate**2)
        affineTerms[index] = rhs - relaxationRate * linearTerms[index] - relaxationRate**2 * quadraticTerms[index]

        if relaxationRate in affineTerms[index].atoms(sp.Symbol):
            raise ValueError("Relaxation Rate decomposition failed (affine part) - run simplification first")
        if relaxationRate in linearTerms[index].atoms(sp.Symbol):
            raise ValueError("Relaxation Rate decomposition failed (linear part) - run simplification first")
        if relaxationRate in quadraticTerms[index].atoms(sp.Symbol):
            raise ValueError("Relaxation Rate decomposition failed (quadratic part) - run simplification first")

    return affineTerms, linearTerms, quadraticTerms


class RelaxationRatePolynomialDecomposition:

    def __init__(self, collisionRule, freeRelaxationRates, fixedRelaxationRates):
        self._collisionRule = collisionRule
        self._freeRelaxationRates = freeRelaxationRates
        self._fixedRelaxationRates = fixedRelaxationRates
        self._allRelaxationRates = fixedRelaxationRates + freeRelaxationRates
        for se in collisionRule.subexpressions:
            for rr in freeRelaxationRates:
                assert rr not in se.rhs.atoms(sp.Symbol), \
                    "Decomposition not possible since free relaxation rates are already in subexpressions"

    def symbolicRelaxationRateFactors(self, relaxationRate, power):
        Q = len(self._collisionRule.method.stencil)
        omegaIdx = self._allRelaxationRates.index(relaxationRate)
        return [sp.Symbol("entFacOmega_%d_%d_%d" % (i, omegaIdx, power)) for i in range(Q)]

    def relaxationRateFactors(self, relaxationRate):
        updateEquations = self._collisionRule.mainEquations

        result = []
        for updateEquation in updateEquations:
            factors = []
            rhs = updateEquation.rhs
            power = 0
            while True:
                power += 1
                factor = rhs.coeff(relaxationRate ** power)
                if factor != 0:
                    if relaxationRate in factor.atoms(sp.Symbol):
                        raise ValueError("Relaxation Rate decomposition failed - run simplification first")
                    factors.append(factor)
                else:
                    break

            result.append(factors)

        return result

    def constantExprs(self):
        subsDict = {rr: 0 for rr in self._freeRelaxationRates}
        subsDict.update({rr: 0 for rr in self._fixedRelaxationRates})
        updateEquations = self._collisionRule.mainEquations
        return [fastSubs(eq.rhs, subsDict) for eq in updateEquations]

    def equilibriumExprs(self):
        subsDict = {rr: 1 for rr in self._freeRelaxationRates}
        subsDict.update({rr: 1 for rr in self._fixedRelaxationRates})
        updateEquations = self._collisionRule.mainEquations
        return [fastSubs(eq.rhs, subsDict) for eq in updateEquations]

    def symbolicEquilibrium(self):
        Q = len(self._collisionRule.method.stencil)
        return [sp.Symbol("entFeq_%d" % (i,)) for i in range(Q)]


def determineRelaxationRateByEntropyConditionIterative(updateRule, omega_s, omega_h,
                                                       newtonIterations=2, initialValue=1):
    method = updateRule.method
    decomp = RelaxationRatePolynomialDecomposition(updateRule, [omega_h], [omega_s])

    # compute and assign relaxation rate factors
    newUpdateEquations = []
    fEqEqs = []
    rrFactorDefinitions = []
    relaxationRates = [omega_s, omega_h]

    for i, constantExpr in enumerate(decomp.constantExprs()):
        updateEqRhs = constantExpr
        fEqRhs = constantExpr
        for rr in relaxationRates:
            factors = decomp.relaxationRateFactors(rr)
            for idx, f in enumerate(factors[i]):
                power = idx + 1
                symbolicFactor = decomp.symbolicRelaxationRateFactors(rr, power)[i]
                rrFactorDefinitions.append(sp.Eq(symbolicFactor, f))
                updateEqRhs += rr ** power * symbolicFactor
                fEqRhs += symbolicFactor
        newUpdateEquations.append(sp.Eq(method.postCollisionPdfSymbols[i], updateEqRhs))
        fEqEqs.append(sp.Eq(decomp.symbolicEquilibrium()[i], fEqRhs))

    # newton iterations to determine free omega
    intermediateOmegas = [sp.Symbol("omega_iter_%i" % (i,)) for i in range(newtonIterations+1)]
    intermediateOmegas[0] = initialValue
    intermediateOmegas[-1] = omega_h

    newtonIterationEquations = []
    for omega_idx in range(len(intermediateOmegas)-1):
        rhsOmega = intermediateOmegas[omega_idx]
        lhsOmega = intermediateOmegas[omega_idx+1]
        updateEqsRhs = [e.rhs for e in newUpdateEquations]
        entropy = discreteApproxEntropy(updateEqsRhs, [e.lhs for e in fEqEqs])
        entropyDiff = sp.diff(entropy, omega_h)
        entropySecondDiff = sp.diff(entropyDiff, omega_h)
        entropyDiff = entropyDiff.subs(omega_h, rhsOmega)
        entropySecondDiff = entropySecondDiff.subs(omega_h, rhsOmega)

        newtonEq = sp.Eq(lhsOmega, rhsOmega - entropyDiff / entropySecondDiff)
        newtonIterationEquations.append(newtonEq)

    # final update equations
    newSubexpressions = updateRule.subexpressions + rrFactorDefinitions + fEqEqs + newtonIterationEquations
    return updateRule.copy(newUpdateEquations, newSubexpressions)


def createKbcEntropicCollisionRule(dim, name='KBC-N4', useNewtonIterations=False, velocityRelaxation=None,
                                   shearRelaxationRate=sp.Symbol("omega"),
                                   higherOrderRelaxationRate=sp.Symbol("omega_h"),
                                   fixedOmega=None, **kwargs):
    def product(iterable):
        return reduce(operator.mul, iterable, 1)

    theMoment = MOMENT_SYMBOLS[:dim]

    rho = [sp.Rational(1, 1)]
    velocity = list(theMoment)

    shearTensorOffDiagonal = [product(t) for t in itertools.combinations(theMoment, 2)]
    shearTensorDiagonal = [m_i * m_i for m_i in theMoment]
    shearTensorTrace = sum(shearTensorDiagonal)
    shearTensorTracefreeDiagonal = [dim * d - shearTensorTrace for d in shearTensorDiagonal]

    energyTransportTensor = list(exponentsToPolynomialRepresentations([a for a in momentsOfOrder(3, dim, True)
                                                                       if 3 not in a]))

    explicitlyDefined = set(rho + velocity + shearTensorOffDiagonal + shearTensorDiagonal + energyTransportTensor)
    rest = list(set(exponentsToPolynomialRepresentations(momentsUpToComponentOrder(2, dim))) - explicitlyDefined)
    assert len(rest) + len(explicitlyDefined) == 3**dim

    # naming according to paper Karlin 2015: Entropic multirelaxation lattice Boltzmann models for turbulent flows
    D = shearTensorOffDiagonal + shearTensorTracefreeDiagonal[:-1]
    T = [shearTensorTrace]
    Q = energyTransportTensor
    if name == 'KBC-N1':
        decomposition = [D, T+Q+rest]
    elif name == 'KBC-N2':
        decomposition = [D+T, Q+rest]
    elif name == 'KBC-N3':
        decomposition = [D+Q, T+rest]
    elif name == 'KBC-N4':
        decomposition = [D+T+Q, rest]
    else:
        raise ValueError("Unknown model. Supported models KBC-Nx")

    omega_s, omega_h = shearRelaxationRate, higherOrderRelaxationRate
    shearPart, restPart = decomposition

    velRelaxation = omega_s if velocityRelaxation is None else velocityRelaxation
    relaxationRates = [omega_s] + \
                      [velRelaxation] * len(velocity) + \
                      [omega_s] * len(shearPart) + \
                      [omega_h] * len(restPart)

    stencil = getStencil("D2Q9") if dim == 2 else getStencil("D3Q27")
    allMoments = rho + velocity + shearPart + restPart
    momentToRr = OrderedDict((m, rr) for m, rr in zip(allMoments, relaxationRates))
    method = createWithDiscreteMaxwellianEqMoments(stencil, momentToRr, cumulant=False, **kwargs)

    simplify = createSimplificationStrategy(method)
    collisionRule = simplify(method.getCollisionRule())

    if useNewtonIterations:
        collisionRule = determineRelaxationRateByEntropyConditionIterative(collisionRule, omega_s, omega_h, 4)
    else:
        collisionRule = determineRelaxationRateByEntropyCondition(collisionRule, omega_s, omega_h)

    if fixedOmega:
        collisionRule = collisionRule.newWithSubstitutions({omega_s: fixedOmega})

    return collisionRule


if __name__ == "__main__":
    ur = createKbcEntropicCollisionRule(2, useNewtonIterations=False)
