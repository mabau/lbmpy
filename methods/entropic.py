import sympy as sp
from pystencils.transformations import fastSubs
from lbmpy.methods.relaxationrates import getShearRelaxationRate


def addEntropyCondition(collisionRule, omegaOutputField=None):
    """
    Transforms an update rule with two relaxation rate into a single relaxation rate rule, where the second
    rate is locally chosen to maximize an entropy condition. This function works for update rules which are
    linear in the relaxation rate, as all moment-based methods are. Cumulant update rules don't work since they are
    quadratic. For these, use :func:`addIterativeEntropyCondition`

    The entropy is approximated such that the optimality condition can be written explicitly, no Newton iterations
    have to be done.

    :param collisionRule: collision rule with two relaxation times
    :param omegaOutputField: pystencils field where computed omegas are stored
    :return: new collision rule which only one relaxation rate
    """
    if collisionRule.method.conservedQuantityComputation.zeroCenteredPdfs:
        raise NotImplementedError("Entropic Methods only implemented for models where pdfs are centered around 1")

    omega_s, omega_h = _getRelaxationRates(collisionRule)

    decomp = RelaxationRatePolynomialDecomposition(collisionRule, [omega_h], [omega_s])
    dh = []
    for entry in decomp.relaxationRateFactors(omega_h):
        assert len(entry) == 1, "The non-iterative entropic procedure works only for moment based methods, which have" \
                                "an update rule linear in the relaxation rate."
        dh.append(entry[0])
    ds = []
    for entry in decomp.relaxationRateFactors(omega_s):
        assert len(entry) <= 1, "The non-iterative entropic procedure works only for moment based methods, which have" \
                                "an update rule linear in the relaxation rate."
        if len(entry) == 0:
            entry.append(0)
        ds.append(entry[0])

    stencil = collisionRule.method.stencil
    Q = len(stencil)
    fSymbols = collisionRule.method.preCollisionPdfSymbols

    dsSymbols = [sp.Symbol("entropicDs_%d" % (i,)) for i in range(Q)]
    dhSymbols = [sp.Symbol("entropicDh_%d" % (i,)) for i in range(Q)]
    feqSymbols = [sp.Symbol("entropicFeq_%d" % (i,)) for i in range(Q)]

    subexprs = [sp.Eq(a, b) for a, b in zip(dsSymbols, ds)] + \
               [sp.Eq(a, b) for a, b in zip(dhSymbols, dh)] + \
               [sp.Eq(a, f_i + ds_i + dh_i) for a, f_i, ds_i, dh_i in zip(feqSymbols, fSymbols, dsSymbols, dhSymbols)]

    optimalOmegaH = _getEntropyMaximizingOmega(omega_s, feqSymbols, dsSymbols, dhSymbols)

    subexprs += [sp.Eq(omega_h, optimalOmegaH)]

    newUpdateEquations = []
    for updateEq in collisionRule.mainEquations:
        index = collisionRule.method.postCollisionPdfSymbols.index(updateEq.lhs)
        newEq = sp.Eq(updateEq.lhs, fSymbols[index] + omega_s * dsSymbols[index] + omega_h * dhSymbols[index])
        newUpdateEquations.append(newEq)
    newCollisionRule = collisionRule.copy(newUpdateEquations, collisionRule.subexpressions + subexprs)
    newCollisionRule.simplificationHints['entropic'] = True
    newCollisionRule.simplificationHints['entropicNewtonIterations'] = None

    if omegaOutputField:
        from lbmpy.updatekernels import writeQuantitiesToField
        newCollisionRule = writeQuantitiesToField(newCollisionRule, omega_h, omegaOutputField)

    return newCollisionRule


def addIterativeEntropyCondition(collisionRule, newtonIterations=3, initialValue=1, omegaOutputField=None):
    """
    More generic, but slower version of :func:`addEntropyCondition`

    A fixed number of Newton iterations is used to determine the maximum entropy relaxation rate.

    :param collisionRule: collision rule with two relaxation times
    :param newtonIterations: (integer) number of newton iterations
    :param initialValue: initial value of the relaxation rate
    :param omegaOutputField: pystencils field where computed omegas are stored
    :return: new collision rule which only one relaxation rate
    """
    if collisionRule.method.conservedQuantityComputation.zeroCenteredPdfs:
        raise NotImplementedError("Entropic Methods only implemented for models where pdfs are centered around 1")

    omega_s, omega_h = _getRelaxationRates(collisionRule)

    decomp = RelaxationRatePolynomialDecomposition(collisionRule, [omega_h], [omega_s])

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
        newUpdateEquations.append(sp.Eq(collisionRule.method.postCollisionPdfSymbols[i], updateEqRhs))
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
    newSubexpressions = collisionRule.subexpressions + rrFactorDefinitions + fEqEqs + newtonIterationEquations
    newCollisionRule = collisionRule.copy(newUpdateEquations, newSubexpressions)
    newCollisionRule.simplificationHints['entropic'] = True
    newCollisionRule.simplificationHints['entropicNewtonIterations'] = newtonIterations

    if omegaOutputField:
        from lbmpy.updatekernels import writeQuantitiesToField
        newCollisionRule = writeQuantitiesToField(newCollisionRule, omega_h, omegaOutputField)

    return newCollisionRule


# --------------------------------- Helper Functions and Classes -------------------------------------------------------


def discreteEntropy(function, reference):
    r"""
    Computes relative entropy between a function :math:`f` and a reference function :math:`r`,
    which is chosen as the equilibrium for entropic methods

    .. math ::
        S = - \sum_i f_i \ln \frac{f_i}{r_i}
    """
    return -sum([f_i * sp.ln(f_i / r_i) for f_i, r_i in zip(function, reference)])


def discreteApproxEntropy(function, reference):
    r"""
    Computes an approximation of the relative entropy between a function :math:`f` and a reference function :math:`r`,
    which is chosen as the equilibrium for entropic methods. The non-approximated version is :func:`discreteEntropy`.

    This approximation assumes that the argument of the logarithm is close to 1, i.e. that the function and reference
    are close, then :math:`\ln \frac{f_i}{r_i} \approx  \frac{f_i}{r_i} - 1`

    .. math ::
        S = - \sum_i f_i \left( \frac{f_i}{r_i} - 1 \right)
    """
    return -sum([f_i * ((f_i / r_i)-1) for f_i, r_i in zip(function, reference)])


def _getEntropyMaximizingOmega(omega_s, f_eq, ds, dh):
    dsdh = sum([ds_i * dh_i / f_eq_i for ds_i, dh_i, f_eq_i in zip(ds, dh, f_eq)])
    dhdh = sum([dh_i * dh_i / f_eq_i for dh_i, f_eq_i in zip(dh, f_eq)])
    return 1 - ((omega_s - 1) * dsdh / dhdh)


class RelaxationRatePolynomialDecomposition(object):

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


def _getRelaxationRates(collisionRule):
    sh = collisionRule.simplificationHints
    assert 'relaxationRates' in sh, "Needs simplification hint 'relaxationRates': Sequence of relaxation rates"

    relaxationRates = set(sh['relaxationRates'])
    if len(relaxationRates) != 2:
        raise ValueError("Entropic methods can only be created for methods with two relaxation rates.\n"
                         "One free relaxation rate determining the viscosity and one to be determined by the "
                         "entropy condition")

    method = collisionRule.method
    omega_s = getShearRelaxationRate(method)
    assert omega_s in relaxationRates

    relaxationRatesWithoutOmegaS = relaxationRates - {omega_s}
    omega_h = list(relaxationRatesWithoutOmegaS)[0]
    return omega_s, omega_h
