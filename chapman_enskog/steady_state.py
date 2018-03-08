import sympy as sp
from pystencils.sympyextensions import normalizeProduct
from lbmpy.chapman_enskog.derivative import Diff, DiffOperator, expandUsingLinearity, normalizeDiffOrder
from lbmpy.chapman_enskog.chapman_enskog import expandedSymbol, useChapmanEnskogAnsatz


class SteadyStateChapmanEnskogAnalysis(object):

    def __init__(self, method, forceModelClass=None, order=4):
        self.method = method
        self.dim = method.dim
        self.order = order
        self.physicalVariables = list(sp.Matrix(self.method.momentEquilibriumValues).atoms(sp.Symbol))  # rho, u..
        self.eps = sp.Symbol("epsilon")

        self.fSym = sp.Symbol("f", commutative=False)
        self.fSyms = [expandedSymbol("f", superscript=i, commutative=False) for i in range(order + 1)]
        self.collisionOpSym = sp.Symbol("A", commutative=False)
        self.forceSym = sp.Symbol("F_q", commutative=False)
        self.velocitySyms = sp.Matrix([expandedSymbol("c", subscript=i, commutative=False) for i in range(self.dim)])

        self.F_q = [0] * len(self.method.stencil)
        self.forceModel = None
        if forceModelClass:
            accelerationSymbols = sp.symbols("a_:%d" % (self.dim,), commutative=False)
            self.physicalVariables += accelerationSymbols
            self.forceModel = forceModelClass(accelerationSymbols)
            self.F_q = self.forceModel(self.method)

        # Perform the analysis
        self.tayloredEquation = self._createTaylorExpandedEquation()
        insertedHierarchy, rawHierarchy = self._createPdfHierarchy(self.tayloredEquation)
        self.pdfHierarchy = insertedHierarchy
        self.pdfHierarchyRaw = rawHierarchy
        self.recombinedEq = self._recombinePdfs(self.pdfHierarchy)

        symbolsToValues = self._getSymbolsToValuesDict()
        self.continuityEquation = self._computeContinuityEquation(self.recombinedEq, symbolsToValues)
        self.momentumEquations = [self._computeMomentumEquation(self.recombinedEq, symbolsToValues, h)
                                  for h in range(self.dim)]

    def getPdfHierarchy(self, order, collisionOperatorSymbol=sp.Symbol("omega")):
        def substituteNonCommutingSymbols(eq):
            return eq.subs({a: sp.Symbol(a.name) for a in eq.atoms(sp.Symbol)})
        result = self.pdfHierarchy[order].subs(self.collisionOpSym, collisionOperatorSymbol)
        result = normalizeDiffOrder(result, functions=(self.fSyms[0], self.forceSym))
        return substituteNonCommutingSymbols(result)

    def getContinuityEquation(self, onlyOrder=None):
        return self._extractOrder(self.continuityEquation, onlyOrder)

    def getMomentumEquation(self, onlyOrder=None):
        return [self._extractOrder(e, onlyOrder) for e in self.momentumEquations]

    def _extractOrder(self, eq, order):
        if order is None:
            return eq
        elif order == 0:
            return eq.subs(self.eps, 0)
        else:
            return eq.coeff(self.eps ** order)

    def _createTaylorExpandedEquation(self):
        """
        Creates a generic, Taylor expanded lattice Boltzmann update equation with collision and force term.
        Collision operator and force terms are represented symbolically.
        """
        c = self.velocitySyms
        Dx = sp.Matrix([DiffOperator(target=l) for l in range(self.dim)])

        differentialOperator = sum((self.eps * c.dot(Dx)) ** n / sp.factorial(n)
                                   for n in range(1, self.order + 1))
        taylorExpansion = DiffOperator.apply(differentialOperator.expand(), self.fSym)

        fNonEq = self.fSym - self.fSyms[0]
        return taylorExpansion + self.collisionOpSym * fNonEq - self.eps * self.forceSym

    def _createPdfHierarchy(self, tayloredEquation):
        """
        Expresses the expanded pdfs f^1, f^2, ..  as functions of the equilibrium f^0.
        Returns a list where element [1] is the equation for f^1 etc.
        """
        chapmanEnskogHierarchy = useChapmanEnskogAnsatz(tayloredEquation, spatialDerivativeOrders=None,
                                                        pdfs=(['f', 0, self.order + 1],), commutative=False)
        chapmanEnskogHierarchy = [chapmanEnskogHierarchy[i] for i in range(self.order + 1)]

        insertedHierarchy = []
        rawHierarchy = []
        substitutionDict = {}
        for ceEq, f_i in zip(chapmanEnskogHierarchy, self.fSyms):
            newEq = -1 / self.collisionOpSym * (ceEq - self.collisionOpSym * f_i)
            rawHierarchy.append(newEq)
            newEq = expandUsingLinearity(newEq.subs(substitutionDict), functions=self.fSyms + [self.forceSym])
            if newEq:
                substitutionDict[f_i] = newEq
            insertedHierarchy.append(newEq)

        return insertedHierarchy, rawHierarchy

    def _recombinePdfs(self, pdfHierarchy):
        return sum(pdfHierarchy[i] * self.eps**(i-1) for i in range(1, self.order+1))

    def _computeContinuityEquation(self, recombinedEq, symbolsToValues):
        return self._computeMoments(recombinedEq, symbolsToValues)

    def _computeMomentumEquation(self, recombinedEq, symbolsToValues, coordinate):
        eq = sp.expand(self.velocitySyms[coordinate] * recombinedEq)

        result = self._computeMoments(eq, symbolsToValues)
        if self.forceModel and hasattr(self.forceModel, 'equilibriumVelocityShift'):
            compressible = self.method.conservedQuantityComputation.compressible
            shift = self.forceModel.equilibriumVelocityShift(sp.Symbol("rho") if compressible else 1)
            result += shift[coordinate]
        return result

    def _getSymbolsToValuesDict(self):
        result = {1 / self.collisionOpSym: self.method.inverseCollisionMatrix,
                  self.forceSym: sp.Matrix(self.forceModel(self.method)) if self.forceModel else 0,
                  self.fSyms[0]: self.method.getEquilibriumTerms()}
        for i, c_i in enumerate(self.velocitySyms):
            result[c_i] = sp.Matrix([d[i] for d in self.method.stencil])

        return result

    def _computeMoments(self, recombinedEq, symbolsToValues):
        eq = recombinedEq.expand()
        assert eq.func is sp.Add

        newProducts = []
        for product in eq.args:
            assert product.func is sp.Mul

            derivative = None

            newProd = 1
            for arg in reversed(normalizeProduct(product)):
                if isinstance(arg, Diff):
                    assert derivative is None, "More than one derivative term in the product"
                    derivative = arg
                    arg = arg.getArgRecursive()  # new argument is inner part of derivative

                if arg in symbolsToValues:
                    arg = symbolsToValues[arg]

                haveShape = hasattr(arg, 'shape') and hasattr(newProd, 'shape')
                if haveShape and arg.shape == newProd.shape and arg.shape[1] == 1:
                    newProd = sp.matrix_multiply_elementwise(newProd, arg)
                else:
                    newProd = arg * newProd

            newProd = sp.expand(sum(newProd))

            if derivative is not None:
                newProd = derivative.changeArgRecursive(newProd)

            newProducts.append(newProd)

        return normalizeDiffOrder(expandUsingLinearity(sp.Add(*newProducts), functions=self.physicalVariables))
