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


def determineRelaxationRateByEntropyCondition(updateRule, relaxationRate):
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
