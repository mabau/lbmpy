import sympy as sp


def scalarProduct(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def getSymbolicVelocityVector(dim, name="u"):
    return sp.symbols(" ".join(["%s_%d" % (name, i) for i in range(dim)]))


def getSymbolicDensity(name="rho"):
    return sp.symbols(name, positive=True)


def getSymbolicSoundSpeed(name="c_s"):
    return sp.symbols(name, positive=True)


def uniqueList(seq):
    seen = {}
    result = []
    for item in seq:
        if item in seen:
            continue
        seen[item] = 1
        result.append(item)
    return result


def matrixFromColumnVectors(columnVectors):
    c = columnVectors
    return sp.Matrix([list(c[i]) for i in range(len(c))]).transpose()


def findFactorToRemoveFractions(v):
    denominators = set()
    for e in v:
        if e.is_Rational:
            denominators.add(e.q)
    return sp.lcm(denominators)


def commonDenominator(expr):
    denominators = [r.q for r in expr.atoms(sp.Rational)]
    return sp.lcm(denominators)

