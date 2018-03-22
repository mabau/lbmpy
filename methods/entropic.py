import sympy as sp
from pystencils.transformations import fastSubs
from lbmpy.relaxationrates import getShearRelaxationRate


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
        raise NotImplementedError("Entropic Methods only implemented for models where pdfs are centered around 1. "
                                  "Use compressible=1")

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

    constPart = decomp.constantExprs()
    for updateEq in collisionRule.mainEquations:
        index = collisionRule.method.postCollisionPdfSymbols.index(updateEq.lhs)
        newEq = sp.Eq(updateEq.lhs, constPart[index] + omega_s * dsSymbols[index] + omega_h * dhSymbols[index])
        newUpdateEquations.append(newEq)
    newCollisionRule = collisionRule.copy(newUpdateEquations, collisionRule.subexpressions + subexprs)
    newCollisionRule.simplificationHints['entropic'] = True
    newCollisionRule.simplificationHints['entropicNewtonIterations'] = None

    if omegaOutputField:
        from lbmpy.updatekernels import writeQuantitiesToField
        newCollisionRule = writeQuantitiesToField(newCollisionRule, omega_h, omegaOutputField)

    return newCollisionRule


def addIterativeEntropyCondition(collisionRule, freeOmega=None, newtonIterations=3, initialValue=1,
                                 omegaOutputField=None):
    """
    More generic, but slower version of :func:`addEntropyCondition`

    A fixed number of Newton iterations is used to determine the maximum entropy relaxation rate.

    :param collisionRule: collision rule with two relaxation times
    :param freeOmega: relaxation rate which should be determined by entropy condition. If left to None, the
                      relaxation rate is automatically detected, which works only if there are 2 relaxation times
    :param newtonIterations: (integer) number of newton iterations
    :param initialValue: initial value of the relaxation rate
    :param omegaOutputField: pystencils field where computed omegas are stored
    :return: new collision rule which only one relaxation rate
    """

    if collisionRule.method.conservedQuantityComputation.zeroCenteredPdfs:
        raise NotImplementedError("Entropic Methods only implemented for models where pdfs are centered around 1")

    if freeOmega is None:
        _, freeOmega = _getRelaxationRates(collisionRule)

    decomp = RelaxationRatePolynomialDecomposition(collisionRule, [freeOmega], [])

    newUpdateEquations = []

    # 1) decompose into constant + freeOmega * ent1 + freeOmega**2 * ent2
    polynomialSubexpressions = []
    rrPolynomials = []
    for i, constantExpr in enumerate(decomp.constantExprs()):
        constantExprEq = sp.Eq(decomp.symbolicConstantExpr(i), constantExpr)
        polynomialSubexpressions.append(constantExprEq)
        rrPolynomial = constantExprEq.lhs

        factors = decomp.relaxationRateFactors(freeOmega)
        for idx, f in enumerate(factors[i]):
            power = idx + 1
            symbolicFactor = decomp.symbolicRelaxationRateFactors(freeOmega, power)[i]
            polynomialSubexpressions.append(sp.Eq(symbolicFactor, f))
            rrPolynomial += freeOmega ** power * symbolicFactor
        rrPolynomials.append(rrPolynomial)
        newUpdateEquations.append(sp.Eq(collisionRule.method.postCollisionPdfSymbols[i], rrPolynomial))

    # 2) get equilibrium from method and define subexpressions for it
    eqTerms = [eq.rhs for eq in collisionRule.method.getEquilibrium().mainEquations]
    eqSymbols = sp.symbols("entropicFeq_:%d" % (len(eqTerms,)))
    eqSubexpressions = [sp.Eq(a, b) for a, b in zip(eqSymbols, eqTerms)]

    # 3) find coefficients of entropy derivatives
    entropyDiff = sp.diff(discreteApproxEntropy(rrPolynomials, eqSymbols), freeOmega)
    coefficientsFirstDiff = [c.expand() for c in reversed(sp.poly(entropyDiff, freeOmega).all_coeffs())]
    symCoeffDiff1 = sp.symbols("entropicDiffCoeff_:%d" % (len(coefficientsFirstDiff,)))
    coefficientEqs = [sp.Eq(a, b) for a, b in zip(symCoeffDiff1, coefficientsFirstDiff)]
    symCoeffDiff2 = [(i+1) * coeff for i, coeff in enumerate(symCoeffDiff1[1:])]

    # 4) define Newtons method update iterations
    newtonIterationEquations = []
    intermediateOmegas = [sp.Symbol("omega_iter_%i" % (i,)) for i in range(newtonIterations+1)]
    intermediateOmegas[0] = initialValue
    intermediateOmegas[-1] = freeOmega
    for omega_idx in range(len(intermediateOmegas)-1):
        rhsOmega = intermediateOmegas[omega_idx]
        lhsOmega = intermediateOmegas[omega_idx+1]
        diff1Poly = sum([coeff * rhsOmega**i for i, coeff in enumerate(symCoeffDiff1)])
        diff2Poly = sum([coeff * rhsOmega**i for i, coeff in enumerate(symCoeffDiff2)])
        newtonEq = sp.Eq(lhsOmega, rhsOmega - diff1Poly / diff2Poly)
        newtonIterationEquations.append(newtonEq)

    # 5) final update equations
    newSubExprs = polynomialSubexpressions + eqSubexpressions + coefficientEqs + newtonIterationEquations
    newCollisionRule = collisionRule.copy(newUpdateEquations, collisionRule.subexpressions + newSubExprs)
    newCollisionRule.simplificationHints['entropic'] = True
    newCollisionRule.simplificationHints['entropicNewtonIterations'] = newtonIterations

    if omegaOutputField:
        from lbmpy.updatekernels import writeQuantitiesToField
        newCollisionRule = writeQuantitiesToField(newCollisionRule, freeOmega, omegaOutputField)

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

    def symbolicConstantExpr(self, i):
        return sp.Symbol("entOffset_%d" % (i,))

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


if __name__ == '__main__':
    from lbmpy.creationfunctions import createLatticeBoltzmannUpdateRule

    createLatticeBoltzmannUpdateRule(stencil='D2Q9', compressible=True, method='trt-kbc-n4', entropic=True,
                                     force=[0.01, 0])

