import sympy as sp
import operator
from collections import defaultdict


def replaceAdditive(expr, replacement, subExpression, minimalMatchingTerms):
    """
    Transformation for replacing a given subexpression inside a sum

    Example 1:
        expr = 3*x + 3 * y
        replacement = k
        subExpression = x+y
        return = 3*k

    Example 2:
        expr = 3*x + 3 * y + z
        replacement = k
        subExpression = x+y+z
        return:
            if minimalMatchingTerms >=3 the expression would not be altered
            if smaller than 3 the result is 3*k - 2*z

    :param expr: input expression
    :param replacement: expression that is inserted for subExpression (if found)
    :param subExpression: expression to replace
    :param minimalMatchingTerms: how many sum-terms have to match to do the replacement
    :return:
    """
    if minimalMatchingTerms <= 0:
        minimalMatchingTerms = 1

    def visit(currentExpr):
        if currentExpr.is_Add:
            exprCoeffs = currentExpr.as_coefficients_dict()
            subexprCoeffDict = subExpression.as_coefficients_dict()
            intersection = set(subexprCoeffDict.keys()).intersection(set(exprCoeffs))
            if len(intersection) >= minimalMatchingTerms:
                # find common factor
                factors = defaultdict(lambda: 0)
                for commonSymbol in subexprCoeffDict.keys():
                    if commonSymbol not in exprCoeffs:
                        continue
                    factor = exprCoeffs[commonSymbol] / subexprCoeffDict[commonSymbol]
                    factors[sp.simplify(factor)] += 1

                commonFactor = max(factors.items(), key=operator.itemgetter(1))[0]
                if factors[commonFactor] >= minimalMatchingTerms:
                    return currentExpr - commonFactor * subExpression + commonFactor * replacement

        # if no subexpression was found
        paramList = [visit(a) for a in currentExpr.args]
        if not paramList:
            return currentExpr
        else:
            return currentExpr.func(*paramList)

    return visit(expr)


def replaceSecondOrderProducts(expr, searchSymbols, positive=None, replaceMixed=None):
    """
    Replaces second order mixed terms like x*y by 2* ( (x+y)**2 - x**2 - y**2 )
    This makes the term longer - simplify usually is undoing these - however this
    transformation can be done to find more common sub-expressions
    :param expr: input expression
    :param searchSymbols: list of symbols that are searched for
                            Example: given [ x,y,z] terms like x*y, x*z, z*y are replaced
    :param positive: there are two ways to do this substitution, either with term
                    (x+y)**2 or (x-y)**2 . if positive=True the first version is done,
                    if positive=False the second version is done, if positive=None the
                    sign is determined by the sign of the mixed term that is replaced
    :param replaceMixed: if a list is passed here the expr x+y or x-y is replaced by a special new symbol
                         the replacement equation is added to the list
    :return:
    """
    if replaceMixed is not None:
        mixedSymbolsReplaced = set([e.lhs for e in replaceMixed])

    if expr.is_Mul:
        distinctVelTerms = set()
        nrOfVelTerms = 0
        otherFactors = 1
        for t in expr.args:
            if t in searchSymbols:
                nrOfVelTerms += 1
                distinctVelTerms.add(t)
            else:
                otherFactors *= t
        if len(distinctVelTerms) == 2 and nrOfVelTerms == 2:
            u, v = list(distinctVelTerms)
            if positive is None:
                otherFactorsWithoutSymbols = otherFactors
                for s in otherFactors.atoms(sp.Symbol):
                    otherFactorsWithoutSymbols = otherFactorsWithoutSymbols.subs(s, 1)
                positive = otherFactorsWithoutSymbols.is_positive
                assert positive is not None
            sign = 1 if positive else -1
            if replaceMixed is not None:
                newSymbolStr = 'P' if positive else 'M'
                mixedSymbolName = u.name + newSymbolStr + v.name
                mixedSymbol = sp.Symbol(mixedSymbolName.replace("_", ""))
                if mixedSymbol not in mixedSymbolsReplaced:
                    mixedSymbolsReplaced.add(mixedSymbol)
                    replaceMixed.append(sp.Eq(mixedSymbol, u + sign * v))
            else:
                mixedSymbol = u + sign * v
            return sp.Rational(1, 2) * sign * otherFactors * (mixedSymbol ** 2 - u ** 2 - v ** 2)

    paramList = [replaceSecondOrderProducts(a, searchSymbols, positive, replaceMixed) for a in expr.args]
    result = expr.func(*paramList) if paramList else expr
    return result


def removeHigherOrderTerms(term, order=3, symbols=None):
    """
    Remove all terms from a sum that contain 'order' or more factors of given 'symbols'
    Example: symbols = ['u_x', 'u_y'] and order =2
             removes terms u_x**2, u_x*u_y, u_y**2, u_x**3, ....
    """
    from sympy.core.power import Pow
    from sympy.core.add import Add, Mul

    result = 0
    term = term.expand()

    if not symbols:
        symbols = sp.symbols(" ".join(["u_%d" % (i,) for i in range(3)]))
        symbols += sp.symbols(" ".join(["u_%d" % (i,) for i in range(3)]), real=True)

    def velocityFactorsInProduct(product):
        uFactorCount = 0
        for factor in product.args:
            if type(factor) == Pow:
                if factor.args[0] in symbols:
                    uFactorCount += factor.args[1]
            if factor in symbols:
                uFactorCount += 1
        return uFactorCount

    if type(term) == Mul:
        if velocityFactorsInProduct(term) <= order:
            return term
        else:
            return sp.Rational(0, 1)

    if type(term) != Add:
        return term

    for sumTerm in term.args:
        if velocityFactorsInProduct(sumTerm) <= order:
            result += sumTerm
    return result


def completeTheSquare(expr, symbolToComplete, newVariable):
    """
    Transforms second order polynomial into only squared part i.e.
        a*symbolToComplete**2 + b*symbolToComplete + c
          is transformed into
        newVariable**2 + d

    returns replacedExpr, "a tuple to to replace newVariable such that old expr comes out again"

    if given expr is not a second order polynomial:
        return expr, None
    """
    p = sp.Poly(expr, symbolToComplete)
    coeffs = p.all_coeffs()
    if len(coeffs) != 3:
        return expr, None
    a, b, _ = coeffs
    expr = expr.subs(symbolToComplete, newVariable - b / (2 * a))
    return sp.simplify(expr), (newVariable, symbolToComplete + b / (2 * a))


def makeExponentialFuncArgumentSquares(expr, variablesToCompleteSquares):
    """Completes squares in arguments of exponential which makes them simpler to integrate
    Very useful for integrating Maxwell-Boltzmann and its moment generating function"""
    expr = sp.simplify(expr)
    dim = len(variablesToCompleteSquares)
    dummies = [sp.Dummy() for i in range(dim)]

    def visit(term):
        if term.func == sp.exp:
            expArg = term.args[0]
            for i in range(dim):
                expArg, substitution = completeTheSquare(expArg, variablesToCompleteSquares[i], dummies[i])
            return sp.exp(sp.simplify(expArg))
        else:
            paramList = [visit(a) for a in term.args]
            if not paramList:
                return term
            else:
                return term.func(*paramList)

    result = visit(expr)
    for i in range(dim):
        result = result.subs(dummies[i], variablesToCompleteSquares[i])
    return result


def pow2mul(expr):
    """
    Convert integer powers in an expression to Muls, like a**2 => a*a.
    """
    pows = list(expr.atoms(sp.Pow))
    if any(not e.is_Integer for b, e in (i.as_base_exp() for i in pows)):

        raise ValueError("A power contains a non-integer exponent")
    repl = zip(pows, (sp.Mul(*[b]*e, evaluate=False) for b, e in (i.as_base_exp() for i in pows)))
    return expr.subs(repl)

