import sympy as sp
from lbmpy.equilibria import standardDiscreteEquilibrium, getWeights


def discreteEntropyFromWeights(function, weights):
    return -sum([f_i * sp.ln(f_i / w_i) for f_i, w_i in zip(function, weights)])


def discreteEntropy(function, stencil):
    return discreteEntropyFromWeights(function, getWeights(stencil))


def discreteApproxEntropy(function, stencil):
    weights = getWeights(stencil)
    return -sum([f_i * ((f_i / w_i)-1) for f_i, w_i in zip(function, weights)])


def findEntropyMaximizingOmega(stencil, relaxationRate, affinePart=None, linearPart=None):
    Q = len(stencil)
    symOffsets = [sp.Symbol("O_%d" % (i,)) for i in range(Q)]
    symFactors = [sp.Symbol("F_%d" % (i,)) for i in range(Q)]

    eqs = [symOffsets[i] + relaxationRate * symFactors[i] for i in range(Q)]

    h = discreteApproxEntropy(eqs, stencil)
    h_diff = sp.cancel(sp.diff(h, relaxationRate))

    solveResult = sp.solve(h_diff, relaxationRate)
    assert len(solveResult) == 1, "Could not solve for optimal omega" + str(len(solveResult))
    result = sp.simplify(solveResult[0])
    if affinePart:
        result = result.subs({a: b for a, b in zip(symOffsets, affinePart)})
    if linearPart:
        result = result.subs({a: b for a, b in zip(symFactors, linearPart)})

    return result


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


def determineRelaxationRateByEntropyConditionWrong(updateRule, relaxationRate):
    affine, linear, quadratic = decompositionByRelaxationRate(updateRule, relaxationRate)
    for i in quadratic:
        if i != 0:
            raise NotImplementedError("Works only for methods where relaxation time occurs linearly")

    lm = updateRule.latticeModel
    stencil = lm.stencil
    Q = len(stencil)
    affineSymbols = [sp.Symbol("entropicAffine_%d" % (i,)) for i in range(Q)]
    linearSymbols = [sp.Symbol("entropicLinear_%d" % (i,)) for i in range(Q)]

    newSubexpressions = [sp.Eq(a, b) for a, b in zip(affineSymbols, affine)] + \
                        [sp.Eq(a, b) for a, b in zip(linearSymbols, linear)]

    exprForRelaxationRate = findEntropyMaximizingOmega(stencil, relaxationRate, affineSymbols, linearSymbols)
    newSubexpressions += [sp.Eq(relaxationRate, exprForRelaxationRate)]

    newUpdateEquations = []
    for updateEq in updateRule.updateEquations:
        index = lm.pdfDestinationSymbols.index(updateEq.lhs)
        newEq = sp.Eq(updateEq.lhs, affineSymbols[index] + relaxationRate * linearSymbols[index])
        newUpdateEquations.append(newEq)
    return updateRule.newWithSubexpressions(newUpdateEquations, newSubexpressions)
