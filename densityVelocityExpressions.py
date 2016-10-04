import sympy as sp
import lbmpy.util as util


def getDensityVelocityExpressions(stencil, symbolicPdfs):
    """
    Returns a list of sympy equations to compute density and velocity for a given stencil
    with the minimum amount of operations
    """
    def filterOutPlusTerms(expr):
        result = 0
        for term in expr.args:
            if not type(term) is sp.Mul:
                result += term
        return result

    dim = len(stencil[0])

    subexpressions = []
    rho = sum(symbolicPdfs)
    u = [0] * dim
    for f, offset in zip(symbolicPdfs, stencil):
        for i in range(dim):
            u[i] += f * int(offset[i])

    plusTerms = [set(filterOutPlusTerms(u_i).args) for u_i in u]
    for i in range(dim):
        rhs = plusTerms[i]
        for j in range(i):
            rhs -= plusTerms[j]
        eq = sp.Eq(sp.Symbol("vel%dTerm" % (i,)), sum(rhs))
        subexpressions.append(eq)

    for subexpression in subexpressions:
        rho = rho.subs(subexpression.rhs, subexpression.lhs)
    rho = sp.Eq(util.getSymbolicDensity(), rho)

    symbolicVel = util.getSymbolicVelocityVector(dim)
    for i in range(dim):
        u[i] = u[i].subs(subexpressions[i].rhs, subexpressions[i].lhs)
        u[i] = sp.Eq(symbolicVel[i], u[i])

    return subexpressions + [rho] + u



