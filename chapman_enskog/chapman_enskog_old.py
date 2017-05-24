import sympy as sp
from sympy.physics.quantum import Ket as Func
from sympy.physics.quantum import Operator
from functools import reduce
import operator

# Disable Ket notation |f> for functions
Func.lbracket_latex =''
Func.rbracket_latex = ''


def prod(seq):
    return reduce(operator.mul, seq, 1)


def allIn(a, b):
    """Tests if all elements of a are contained in b"""
    return all(element in b for element in a)


def isDerivative(term):
    return isinstance(term, Operator) and term.args[0].name.startswith("\\partial")


def getDiffOperator(subscript, superscript=None):
    symbolName = "\partial_" + subscript
    if superscript:
        symbolName += "^" + superscript
    return Operator(symbolName)


def normalizeProduct(prod):
    """Takes a sympy Mul node and returns a list of factors (with Pow nodes resolved)"""
    def handlePow(pow):
        if pow.exp.is_integer and pow.exp.is_number and pow.exp > 0:
            return [pow.base] * pow.exp
        else:
            return [pow]

    if prod.func == sp.Symbol or prod.func == Func:
        return [prod]

    if prod.func == sp.Pow:
        return handlePow(prod)

    assert prod.func == sp.Mul

    result = []
    for a in prod.args:
        if a.func == sp.Pow:
            result += handlePow(a)
        else:
            result.append(a)
    return result


def splitDerivativeProd(expr):
    """Splits a term a b \partial c d  into three parts: functions before operator (here a b), 
       the operator itself, and terms the operator acts on (here c d). If the passed expressions does
       not have above form, None is returned"""
    productArgs = normalizeProduct(expr)
    argumentsActedOn = []
    remainingArguments = []
    derivOp = None
    for t in reversed(productArgs):
        if isDerivative(t) and derivOp is None:
            derivOp = t
        elif derivOp is None:
            argumentsActedOn.append(t)
        else:
            remainingArguments.append(t)
    if derivOp is None:
        return None
    return list(reversed(remainingArguments)), derivOp, argumentsActedOn


def getName(obj):
    if type(obj) in [Operator, Func]:
        return obj.args[0].name
    else:
        return obj.name


def combineDerivatives(expr):
    from collections import defaultdict

    def searchAndCombine(termList):
        if len(termList) <= 1:
            return termList, False

        newTermList = []
        handledIdxSet = set()

        changes = False
        for i in range(len(termList)):
            if i in handledIdxSet:
                continue
            for j in range(i + 1, len(termList)):
                pre1, post1 = termList[i]
                pre2, post2 = termList[j]
                rest1 = [item for item in pre1 if item not in post2]
                rest2 = [item for item in pre2 if item not in post1]
                restEqual = (len(rest1) == len(rest2)) and (set(rest1) == set(rest2))
                if allIn(post1, pre2) and allIn(post2, pre1) and restEqual:
                    handledIdxSet.add(j)
                    newTermList.append((rest1, post1 + post2))
                    changes = True
                    break
                else:
                    newTermList.append(termList[i])
        lastIdx = len(termList) - 1
        if lastIdx not in handledIdxSet:
            newTermList.append(termList[lastIdx])

        return newTermList, changes

    expr = expr.expand()
    if expr.func != sp.Add:
        return expr

    operatorDict = defaultdict(list)  # maps the operator to a list of tuples with terms before and after the operator
    result = 0
    for term in expr.args:
        splitted = splitDerivativeProd(term)
        if splitted:
            pre, op, post = splitted
            operatorDict[op].append((pre, post))
        else:
            result += term

    for op, termList in operatorDict.items():
        newTermList, changes = searchAndCombine(termList)
        while changes:
            newTermList, changes = searchAndCombine(newTermList)

        for pre, post in newTermList:
            result += prod(pre) * op * prod(post)

    return result


def expandDerivatives(expr):
    """Fully expands all derivatives by applying product rule"""

    def handleProduct(term):
        splittedTerm = splitDerivativeProd(term)
        if splittedTerm is None:
            return term
        remainingArguments, derivOp, argumentsActedOn = splittedTerm

        result = 0
        for i in range(len(argumentsActedOn)):
            beforeOp = prod([argumentsActedOn[j] for j in range(len(argumentsActedOn)) if j != i])
            result += beforeOp * derivOp * argumentsActedOn[i]
        return prod(remainingArguments) * result

    expr = expr.expand()
    if expr.func != sp.Add:
        return handleProduct(expr)
    else:
        return sum(handleProduct(t) for t in expr.args)


def commuteTerms(expr):
    def commute(term):
        forward = {obj: sp.Symbol(getName(obj)) for obj in term.atoms(Func)}
        backward = {sp.Symbol(getName(obj)): obj for obj in term.atoms(Func)}
        return term.subs(forward).subs(backward)

    def handleProduct(term):
        splittedTerm = splitDerivativeProd(term)
        if splittedTerm is None:
            return term
        remainingArguments, derivOp, argumentsActedOn = splittedTerm

        return commute(prod(remainingArguments)) * derivOp * commute(prod(argumentsActedOn))

    expr = expr.expand()
    if expr.func != sp.Add:
        return handleProduct(expr)
    else:
        return sum(handleProduct(t) for t in expr.args)


#def splitDerivativeTerm(term):
#    if term.func == sp.Mul:
#        derivativeIdx = [i for i, arg in enumerate(term.args) if isDerivative(arg)]
#        if len(derivativeIdx) > 0:
#            idx = derivativeIdx[-1]
#            diffOperator = term.args[idx]
#            assert not diffOperator.is_commutative
#            derivedTerms = [t for t in term.args[idx + 1:] if not t.is_commutative]
#
#            newDerivedTerms = []
#            for t in derivedTerms:
#                if t.func == sp.Pow and t.exp.is_integer and t.exp.is_number and 0 < t.exp < 10:
#                    newDerivedTerms += [t.base] * t.exp
#                else:
#                    newDerivedTerms.append(t)
#            derivedTerms = newDerivedTerms
#
#            notDerivedTerms = list(term.args[:idx]) + [t for t in term.args[idx + 1:] if t.is_commutative]
#            return diffOperator, notDerivedTerms, derivedTerms
#    return None, [], []
#
#
#def productRule(expr):
#    """
#    Applies product rule to terms with differential operator(s)
#    :param expr: sympy expression to apply product rule to
#    :return:
#    """
#    def visit(term):
#        diffOperator, notDerivedTerms, derivedTerms = splitDerivativeTerm(term)
#        if len(derivedTerms) > 1:
#            result = 0
#            for i in range(len(derivedTerms)):
#                beforeOperator = [t for j, t in enumerate(derivedTerms) if j != i]
#                result += prod(notDerivedTerms + beforeOperator + [diffOperator, derivedTerms[i]])
#            return result
#
#        return term if not term.args else term.func(*[visit(a) for a in term.args])
#
#    return visit(expr)
#
#
#def substituteDerivative(expr, diffOperatorToSearch, termsToSearch, substitution):
#    def visit(term):
#        diffOperator, notDerivedTerms, derivedTerms = splitDerivativeTerm(term)
#        if diffOperator == diffOperatorToSearch:
#            termMatches = True
#            for innerTerm in termsToSearch:
#                if innerTerm not in notDerivedTerms and innerTerm not in derivedTerms:
#                    termMatches = False
#                    break
#            if termMatches:
#                pass
#
#            if len(derivedTerms) > 1:
#                result = 0
#                for i in range(len(derivedTerms)):
#                    beforeOperator = [t for j, t in enumerate(derivedTerms) if j != i]
#                    result += prod(notDerivedTerms + beforeOperator + [diffOperator, derivedTerms[i]])
#                return result
#
#        return term if not term.args else term.func(*[visit(a) for a in term.args])
#
#    return visit(expr)
#
import sympy
import sympy.core
import sympy as sp


class UnappliedCeDifferential(sympy.Expr):
    is_commutative = True
    is_number = False
    is_Rational = False

    def __new__(cls, label=-1, index=-1, **kwargs):
        return sympy.Expr.__new__(cls, sp.sympify(label), sp.sympify(index), **kwargs)

    @property
    def label(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    def _latex(self, printer, *args):
        result = "{\partial"
        if self.index >= 0:
            result += "^{(%s)}" % (self.index,)
        if self.label != -1:
            result += "_{%s}" % (self.label,)
        result += "}"
        return result


class CeDifferential(sympy.Expr):
    is_commutative = True
    is_number = False
    is_Rational = False

    def __new__(cls, argument, label=-1, index=-1, **kwargs):
        return sympy.Expr.__new__(cls, sp.sympify(label), sp.sympify(index), argument, **kwargs)

    #@classmethod
    #def class_key(cls):
    #    return 500, 100, "ZZZZZCeF"

    @property
    def label(self):
        return self.args[0]

    @property
    def index(self):
        return self.args[1]

    @property
    def arg(self):
        return self.args[2]

    #def sort_key(self, **kwargs):
    #    #TODO
    #    return sp.Symbol("a").sort_key()

    def _latex(self, printer, *args):
        result = "{\partial"
        if self.index >= 0:
            result += "^{(%s)}" % (self.index,)
        if self.label != -1:
            result += "_{%s}" % (self.label,)

        contents = printer.doprint(self.arg)
        if isinstance(self.arg, int) or self.arg.func == sp.Symbol or self.arg.is_number or self.arg.func == CeDifferential:
            result += " " + contents
        else:
            result += " (" + contents + ") "

        result += "}"
        return result

    def __str__(self):
        return "Diff(%s, %s, %s)" % (self.arg, self.label, self.index)

    def splitLinear(self, functions):
        constant, variable = 1, 1

        if self.arg.func != sp.Mul:
            constant, variable = 1, self.arg
        else:
            for factor in normalizeProduct(self.arg):
                if factor in functions or isinstance(factor, CeDifferential):
                    variable *= factor
                else:
                    constant *= factor

        assert variable != 1, "splitLinear failed - Differential without function that is acted upon"
        return constant * CeDifferential(variable, label=self.label, index=self.index)


def expandDiffs(expr):
    """Fully expands all derivatives by applying product rule"""
    if isinstance(expr, CeDifferential):
        arg = expandDiffs(expr.args[0])
        if arg.func not in (sp.Mul, sp.Pow):
            return CeDifferential(arg, label=expr.label, index=expr.index)
        else:
            prodList = normalizeProduct(arg)
            result = 0
            for i in range(len(prodList)):
                preFactor = prod(prodList[j] for j in range(len(prodList)) if i != j)
                result += preFactor * CeDifferential(prodList[i], label=expr.label, index=expr.index)
            return result
    else:
        newArgs = [expandDiffs(e) for e in expr.args]
        return expr.func(*newArgs) if newArgs else expr


def applyLinearity(expr, functions):
    if isinstance(expr, CeDifferential):
        arg = applyLinearity(expr.arg, functions)
        if arg.func == sp.Add:
            result = 0
            for a in arg.args:
                result += CeDifferential(a, label=expr.label, index=expr.index).splitLinear(functions)
            return result
        else:
            return CeDifferential(arg, label=expr.label, index=expr.index).splitLinear(functions)
    else:
        newArgs = [applyLinearity(e, functions) for e in expr.args]
        return expr.func(*newArgs) if newArgs else expr


def diffSortKey(d):
    return str(d.index), str(d.label)


def normalizeDiffOrder(expression, functions):
    def visit(expr):
        if isinstance(expr, CeDifferential):
            nodes = [expr]
            while isinstance(nodes[-1].arg, CeDifferential):
                nodes.append(nodes[-1].arg)

            processedArg = visit(nodes[-1].arg)
            nodes.sort(key=diffSortKey)

            result = processedArg
            for d in reversed(nodes):
                result = CeDifferential(result, label=d.label, index=d.index)
            return result
        else:
            newArgs = [visit(e) for e in expr.args]
            return expr.func(*newArgs) if newArgs else expr

    expression = applyLinearity(expression.expand(), functions).expand()
    return visit(expression)



# Required Transformations:
# - linearity: move constants before diff nodes, split diff nodes at sums, transformation should get non-constant objects [ok]
# -  handle chained application
#       - ordering transformation, passing order into transformation [ok]
#       - merge into single derivative object?
# - at beginning: take operator equation and translate into derivative nodes
# - inverse product rule


def applyDiffs(expr, argument):
    def handleMul(mul):
        args = normalizeProduct(mul)
        diffs = [a for a in args if isinstance(a, UnappliedCeDifferential)]
        if len(diffs) == 0:
            return mul
        rest = [a for a in args if not isinstance(a, UnappliedCeDifferential)]
        diffs.sort(key=diffSortKey)
        result = argument
        for d in reversed(diffs):
            result = CeDifferential(result, label=d.label, index=d.index)
        return prod(rest) * result

    expr = expr.expand()
    if expr.func == sp.Mul or expr.func == sp.Pow:
        return handleMul(expr)
    elif expr.func == sp.Add:
        return expr.func(*[handleMul(a) for a in expr.args])
    else:
        return expr


if __name__ == '__main__':
    from sympy.abc import a, b, c, x, y, z, t
    import sympy as sp

    Diff = CeDifferential
    DiffOp = UnappliedCeDifferential



    dim = 2
    dimLabels = [sp.Rational(0, 1), sp.Rational(1, 1), sp.Rational(2, 1)][:dim]
    # dimLabels = ['0', '1', '2'][:dim]

    stencil = "D2Q9"
    equilibriumAccuracyOrder = 2

    c = sp.Matrix([sp.Symbol("c_{label}".format(label=label)) for label in dimLabels])
    dt, t, f, C = sp.symbols("Delta_t t f C")

    t = Diff(f) * sp.Symbol("k_0")
    print(sp.latex(t))

    taylorOrder = 2
    Dt = DiffOp(label=t)
    Dx = sp.Matrix([DiffOp(label=l) for l in dimLabels])
    ((dt / 2) * (Dt + c.dot(Dx))).expand()

    taylorOperator = sum(dt ** k * (Dt + c.dot(Dx)) ** k / sp.functions.factorial(k)
                         for k in range(1, taylorOrder + 1))

    eq_4_5 = applyDiffs(taylorOperator, f) - C
    eq_4_5

    Diff(dt * Diff(f)).splitLinear([f])
    #test = applyDiffs((dt / 2) * (Dt + c.dot(Dx)), eq_4_5)
    #res = applyLinearity(test.args[0], [f, C])
    #print(res)
    #print(sp.latex(res))
