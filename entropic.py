import sympy as sp
from lbmpy.equilibria import standardDiscreteEquilibrium, getWeights
from pystencils.transformations import fastSubs


def discreteEntropyFromWeights(function, weights):
    return -sum([f_i * sp.ln(f_i / w_i) for f_i, w_i in zip(function, weights)])


def discreteApproxEntropyFromWeights(function, weights):
    return -sum([f_i * ((f_i / w_i)-1) for f_i, w_i in zip(function, weights)])


def discreteEntropy(function, stencil):
    return discreteEntropyFromWeights(function, getWeights(stencil))


def discreteApproxEntropy(function, stencil):
    weights = getWeights(stencil)
    return discreteApproxEntropyFromWeights(function, weights)


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
    stencil = updateRule.latticeModel.stencil
    Q = len(stencil)
    fSymbols = updateRule.latticeModel.pdfSymbols

    updateEqsRhs = [e.rhs for e in updateRule.updateEquations]
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
    for updateEq in updateRule.updateEquations:
        index = updateRule.latticeModel.pdfDestinationSymbols.index(updateEq.lhs)
        newEq = sp.Eq(updateEq.lhs, fSymbols[index] + omega_s * dsSymbols[index] + omega_h * dhSymbols[index])
        newUpdateEquations.append(newEq)
    return updateRule.newWithSubexpressions(newUpdateEquations, subexprs)


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
        Q = len(self._collisionRule.latticeModel.stencil)
        omegaIdx = self._allRelaxationRates.index(relaxationRate)
        return [sp.Symbol("entFacOmega_%d_%d_%d" % (i, omegaIdx, power)) for i in range(Q)]

    def relaxationRateFactors(self, relaxationRate):
        updateEquations = self._collisionRule.updateEquations

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
        updateEquations = self._collisionRule.updateEquations
        return [fastSubs(eq.rhs, subsDict) for eq in updateEquations]

    def equilibriumExprs(self):
        subsDict = {rr: 1 for rr in self._freeRelaxationRates}
        subsDict.update({rr: 1 for rr in self._fixedRelaxationRates})
        updateEquations = self._collisionRule.updateEquations
        return [fastSubs(eq.rhs, subsDict) for eq in updateEquations]

    def symbolicEquilibrium(self):
        Q = len(self._collisionRule.latticeModel.stencil)
        return [sp.Symbol("entFeq_%d" % (i,)) for i in range(Q)]


def determineRelaxationRateByEntropyConditionIterative(updateRule, omega_s, omega_h,
                                                       newtonIterations=2, initialValue=1):
    lm = updateRule.latticeModel
    decomp = RelaxationRatePolynomialDecomposition(updateRule, [omega_h], [omega_s])

    # compute and assign f_eq
    #fEqEqs = [sp.Eq(a, b) for a, b in zip(decomp.symbolicEquilibrium(), decomp.equilibriumExprs())]

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
        newUpdateEquations.append(sp.Eq(lm.pdfDestinationSymbols[i], updateEqRhs))
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
        entropy = discreteApproxEntropyFromWeights(updateEqsRhs, [e.lhs for e in fEqEqs])
        entropyDiff = sp.diff(entropy, omega_h)
        entropySecondDiff = sp.diff(entropyDiff, omega_h)
        entropyDiff = entropyDiff.subs(omega_h, rhsOmega)
        entropySecondDiff = entropySecondDiff.subs(omega_h, rhsOmega)

        newtonEq = sp.Eq(lhsOmega, rhsOmega - entropyDiff / entropySecondDiff)
        newtonIterationEquations.append(newtonEq)

    # final update equations
    return updateRule.newWithSubexpressions(newUpdateEquations, rrFactorDefinitions + fEqEqs + newtonIterationEquations)


