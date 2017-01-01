import sympy as sp
from lbmpy.methods.abstractlbmmethod import AbstractLbmMethod
from pystencils.equationcollection.equationcollection import EquationCollection


class HydrodynamicLbmMethod(AbstractLbmMethod):

    def __init__(self, stencil, forceModel, compressible, viscosityRelaxationRate, c_s_sq):
        super(HydrodynamicLbmMethod, self).__init__(stencil)
        self._compressible = compressible
        self._forceModel = forceModel
        self._viscosityRelaxationRate = viscosityRelaxationRate
        self._c_s_sq = c_s_sq

    @property
    def densitySymbol(self):
        return sp.Symbol('rho')

    @property
    def pressureSymbol(self):
        return sp.Symbol("p")

    @property
    def velocitySymbols(self):
        return sp.symbols("u_:%d" % (self.dim,))

    @property
    def availableMacroscopicQuantities(self):
        return {'density': self.densitySymbol,
                'pressure': self.pressureSymbol,
                'velocity': self.velocitySymbols}

    @property
    def viscosityRelaxationRate(self):
        return self._viscosityRelaxationRate

    @property
    def conservedQuantitiesSymbols(self):
        return self.availableMacroscopicQuantities.values()

    def getMacroscopicQuantitiesEquations(self, sequenceOfSymbols):
        # TODO shift of velocity by force model
        eqColl = _densityVelocityExpressions(self.stencil, self.preCollisionPdfSymbols, self._compressible,
                                             self._c_s_sq, self.densitySymbol, self.pressureSymbol,
                                             self.velocitySymbols)

        return eqColl.extract(sequenceOfSymbols)


def _densityVelocityExpressions(stencil, symbolicPdfs, compressible, c_s_sq,
                                symbolicDensity, symbolicPressure, symbolicVelocities):
    """
    Returns an equation collection, containing equations to compute density, velocity and pressure from pdf values
    """
    def filterOutPlusTerms(expr):
        result = 0
        for term in expr.args:
            if not type(term) is sp.Mul:
                result += term
        return result

    dim = len(stencil[0])

    subexpressions = []
    pdfSum = sum(symbolicPdfs)
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
        pdfSum = pdfSum.subs(subexpression.rhs, subexpression.lhs)

    for i in range(dim):
        u[i] = u[i].subs(subexpressions[i].rhs, subexpressions[i].lhs)

    equations = []
    if compressible:
        rho_inv = sp.Symbol("rho_inv")
        subexpressions.append(sp.Eq(rho_inv, 1 / symbolicDensity))
        equations += [sp.Eq(symbolicDensity, pdfSum)]
        equations += [sp.Eq(symbolicPressure, pdfSum * c_s_sq)]
        equations += [sp.Eq(u_i_sym, u_i * rho_inv) for u_i_sym, u_i in zip(symbolicVelocities, u)]
    else:
        subexpressions.append(sp.Eq(symbolicDensity, sp.Rational(1, 1)))
        subexpressions.append(sp.Eq(symbolicPressure, pdfSum * c_s_sq))
        equations += [sp.Eq(u_i_sym, u_i) for u_i_sym, u_i in zip(symbolicVelocities, u)]

    return EquationCollection(equations, subexpressions)
