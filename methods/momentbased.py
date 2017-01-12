import sympy as sp
import collections
from collections import namedtuple, OrderedDict, defaultdict

from lbmpy.maxwellian_equilibrium import getMomentsOfDiscreteMaxwellianEquilibrium, \
    getMomentsOfContinuousMaxwellianEquilibrium
from lbmpy.methods.abstractlbmmethod import AbstractLbmMethod
from lbmpy.methods.conservedquantitycomputation import AbstractConservedQuantityComputation, DensityVelocityComputation
from lbmpy.moments import MOMENT_SYMBOLS, momentMatrix, exponentsToPolynomialRepresentations, isShearMoment, \
    momentsUpToComponentOrder, isEven, gramSchmidt, getOrder
from pystencils.equationcollection import EquationCollection
from pystencils.sympyextensions import commonDenominator


RelaxationInfo = namedtuple('Relaxationinfo', ['equilibriumValue', 'relaxationRate'])


class MomentBasedLbmMethod(AbstractLbmMethod):

    def __init__(self, stencil, momentToRelaxationInfoDict, conservedQuantityComputation, forceModel=None):
        """
        Moment based LBM is a class to represent the single (SRT), two (TRT) and multi relaxation time (MRT) methods.
        These methods work by transforming the pdfs into moment space using a linear transformation. In the moment
        space each component (moment) is relaxed to an equilibrium moment by a certain relaxation rate. These
        equilibrium moments can e.g. be determined by taking the equilibrium moments of the continuous Maxwellian.

        :param stencil: see :func:`lbmpy.stencils.getStencil`
        :param momentToRelaxationInfoDict: a dictionary mapping moments in either tuple or polynomial formulation
                                           to a RelaxationInfo, which consists of the corresponding equilibrium moment
                                           and a relaxation rate
        :param conservedQuantityComputation: instance of :class:`lbmpy.methods.AbstractConservedQuantityComputation`.
                                             This determines how conserved quantities are computed, and defines
                                             the symbols used in the equilibrium moments like e.g. density and velocity
        :param forceModel: force model instance, or None if no forcing terms are required
        """
        super(MomentBasedLbmMethod, self).__init__(stencil)

        assert isinstance(conservedQuantityComputation, AbstractConservedQuantityComputation)

        moments = []
        relaxationRates = []
        equilibriumMoments = []
        for moment, relaxInfo in momentToRelaxationInfoDict.items():
            moments.append(moment)
            relaxationRates.append(relaxInfo.relaxationRate)
            equilibriumMoments.append(relaxInfo.equilibriumValue)

        self._forceModel = forceModel
        self._moments = moments
        self._momentToRelaxationInfoDict = momentToRelaxationInfoDict
        self._momentMatrix = momentMatrix(moments, self.stencil)
        self._relaxationRates = sp.Matrix(relaxationRates)
        self._equilibriumMoments = sp.Matrix(equilibriumMoments)
        self._conservedQuantityComputation = conservedQuantityComputation

        symbolsInEquilibriumMoments = self._equilibriumMoments.atoms(sp.Symbol)
        conservedQuantities = set()
        for v in self._conservedQuantityComputation.definedSymbols().values():
            if isinstance(v, collections.Sequence):
                conservedQuantities.update(v)
            else:
                conservedQuantities.add(v)
        undefinedEquilibriumSymbols = symbolsInEquilibriumMoments - conservedQuantities

        assert len(undefinedEquilibriumSymbols) == 0, "Undefined symbol(s) in equilibrium moment: %s" % \
                                                      (undefinedEquilibriumSymbols, )

        self._weights = None

    def _repr_html_(self):
        table = """
        <table style="border:none; width: 100%">
            <tr {nb}>
                <th {nb} >Moment</th>
                <th {nb} >Eq. Value </th>
                <th {nb} >Relaxation Time</th>
            </tr>
            {content}
        </table>
        """
        content = ""
        for rr, moment, eqValue in zip(self._relaxationRates, self._moments, self._equilibriumMoments):
            vals = {
                'rr': sp.latex(rr),
                'moment': sp.latex(moment),
                'eqValue': sp.latex(eqValue),
                'nb': 'style="border:none"',
            }
            content += """<tr {nb}>
                            <td {nb}>${moment}$</td>
                            <td {nb}>${eqValue}$</td>
                            <td {nb}>${rr}$</td>
                         </tr>\n""".format(**vals)
        return table.format(content=content, nb='style="border:none"')

    @property
    def zerothOrderEquilibriumMomentSymbol(self,):
        return self._conservedQuantityComputation.definedSymbols(order=0)[1]

    @property
    def firstOrderEquilibriumMomentSymbols(self,):
        return self._conservedQuantityComputation.definedSymbols(order=1)[1]

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self._computeWeights()
        return self._weights

    def _computeWeights(self):
        replacements = self._conservedQuantityComputation.defaultValues
        eqColl = self.getEquilibrium().newWithSubstitutionsApplied(replacements).insertSubexpressions()
        weights = []
        for eq in eqColl.mainEquations:
            value = eq.rhs.expand()
            assert len(value.atoms(sp.Symbol)) == 0, "Failed to compute weights"
            weights.append(value)
        return weights

    def getShearRelaxationRate(self):
        """
        Assumes that all shear moments are relaxed with same rate - returns this rate
        Shear moments in 3D are: x*y, x*z and y*z - in 2D its only x*y
        The shear relaxation rate determines the viscosity in hydrodynamic LBM schemes
        """
        relaxationRates = set()
        for moment, relaxInfo in self._momentToRelaxationInfoDict.items():
            if isShearMoment(moment):
                relaxationRates.add(relaxInfo.relaxationRate)
        if len(relaxationRates) == 1:
            return relaxationRates.pop()
        else:
            if len(relaxationRates) > 1:
                raise ValueError("Shear moments are relaxed with different relaxation times: %s" % (relaxationRates,))
            else:
                raise NotImplementedError("Shear moments seem to be not relaxed separately - "
                                          "Can not determine their relaxation rate automatically")

    def getEquilibrium(self):
        D = sp.eye(len(self._relaxationRates))
        return self._getCollisionRuleWithRelaxationMatrix(D)

    def getCollisionRule(self):
        D = sp.diag(*self._relaxationRates)
        eqColl = self._getCollisionRuleWithRelaxationMatrix(D)
        if self._forceModel is not None:
            forceModelTerms = self._forceModel(self)
            newEqs = [sp.Eq(eq.lhs, eq.rhs + fmt) for eq, fmt in zip(eqColl.mainEquations, forceModelTerms)]
            eqColl = eqColl.newWithAdditionalSubexpressions(newEqs, [])
        return eqColl

    @property
    def conservedQuantityComputation(self):
        return self._conservedQuantityComputation

    def _getCollisionRuleWithRelaxationMatrix(self, D):
        f = sp.Matrix(self.preCollisionPdfSymbols)
        M = self._momentMatrix
        m_eq = self._equilibriumMoments

        collisionRule = f + M.inv() * D * (m_eq - M * f)
        collisionEqs = [sp.Eq(lhs, rhs) for lhs, rhs in zip(self.postCollisionPdfSymbols, collisionRule)]

        eqValueEqs = self._conservedQuantityComputation.equilibriumInputEquationsFromPdfs(f)
        simplificationHints = eqValueEqs.simplificationHints
        # TODO add own simplification hints here
        return EquationCollection(collisionEqs, eqValueEqs.subexpressions + eqValueEqs.mainEquations,
                                  simplificationHints)


# ------------------------------------ Helper Functions ----------------------------------------------------------------


def sortMomentsIntoGroupsOfSameOrder(moments):
    """Returns a dictionary mapping the order (int) to a list of moments with that order."""
    result = defaultdict(list)
    for i, moment in enumerate(moments):
        order = getOrder(moment)
        result[order].append(moment)
    return result


def defaultRelaxationRateNames():
    nextIndex = 0

    def result(momentList):
        nonlocal nextIndex
        shearMomentInside = False
        allConservedMoments = True
        for m in momentList:
            if isShearMoment(m):
                shearMomentInside = True
            if not (getOrder(m) == 0 or getOrder(m) == 1):
                allConservedMoments = False

        if shearMomentInside:
            return sp.Symbol("omega")
        elif allConservedMoments:
            return 0
        else:
            nextIndex += 1
            return sp.Symbol("omega_%d" % (nextIndex,))

    return result


def relaxationRateFromMagicNumber(hydrodynamicRelaxationRate, magicNumber):
    omega = hydrodynamicRelaxationRate
    return (4 - 2 * omega) / (4 * magicNumber * omega + 2 - omega)


# -------------------- Generic Creators by matching equilibrium moments ------------------------------------------------


def createWithDiscreteMaxwellianEqMoments(stencil, momentToRelaxationRateDict, compressible=False, forceModel=None,
                                          equilibriumAccuracyOrder=2):
    """
    Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate. These moments are
    relaxed against the moments of the discrete Maxwellian distribution.

    :param stencil: nested tuple defining the discrete velocity space. See `func:lbmpy.stencils.getStencil`
    :param momentToRelaxationRateDict: dict that has as many entries as the stencil. Each moment, which can be
                                       represented by an exponent tuple or in polynomial form
                                       (see `lbmpy.moments`), is mapped to a relaxation rate.
    :param compressible: using the compressible or incompressible discrete Maxwellian
    :param forceModel: force model instance, or None if no external forces
    :param equilibriumAccuracyOrder: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
    :return: :class:`lbmpy.methods.MomentBasedLbmMethod` instance
    """
    momToRrDict = OrderedDict(momentToRelaxationRateDict)
    assert len(momToRrDict) == len(
        stencil), "The number of moments has to be the same as the number of stencil entries"

    densityVelocityComputation = DensityVelocityComputation(stencil, compressible, forceModel)
    eqMoments = getMomentsOfDiscreteMaxwellianEquilibrium(stencil, list(momToRrDict.keys()), c_s_sq=sp.Rational(1, 3),
                                                          compressible=compressible, order=equilibriumAccuracyOrder)
    rrDict = OrderedDict([(mom, RelaxationInfo(eqMom, rr))
                          for mom, rr, eqMom in zip(momToRrDict.keys(), momToRrDict.values(), eqMoments)])
    return MomentBasedLbmMethod(stencil, rrDict, densityVelocityComputation, forceModel)


def createWithContinuousMaxwellianEqMoments(stencil, momentToRelaxationRateDict, forceModel=None,
                                            equilibriumAccuracyOrder=None):
    """
    Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate. These moments are
    relaxed against the moments of the continuous Maxwellian distribution.
    For parameter description see :func:`lbmpy.methods.createWithDiscreteMaxwellianEqMoments`.
    By using the continuous Maxwellian we automatically get a compressible model.
    """
    momToRrDict = OrderedDict(momentToRelaxationRateDict)
    assert len(momToRrDict) == len(
        stencil), "The number of moments has to be the same as the number of stencil entries"
    dim = len(stencil[0])
    densityVelocityComputation = DensityVelocityComputation(stencil, True, forceModel)
    eqMoments = getMomentsOfContinuousMaxwellianEquilibrium(list(momToRrDict.keys()), stencil, dim,
                                                            c_s_sq=sp.Rational(1, 3),
                                                            order=equilibriumAccuracyOrder)
    rrDict = OrderedDict([(mom, RelaxationInfo(eqMom, rr))
                          for mom, rr, eqMom in zip(momToRrDict.keys(), momToRrDict.values(), eqMoments)])
    return MomentBasedLbmMethod(stencil, rrDict, densityVelocityComputation, forceModel)


# ------------------------------------ SRT / TRT/ MRT Creators ---------------------------------------------------------


def createSRT(stencil, relaxationRate, compressible=False, forceModel=None, equilibriumAccuracyOrder=2):
    r"""
    Creates a single relaxation time (SRT) lattice Boltzmann model also known as BGK model.

    :param stencil: nested tuple defining the discrete velocity space. See :func:`lbmpy.stencils.getStencil`
    :param relaxationRate: relaxation rate (inverse of the relaxation time)
                           usually called :math:`\omega` in LBM literature
    :param compressible: incompressible LBM methods split the density into :math:`\rho = \rho_0 + \Delta \rho`
             where :math:`\rho_0` is chosen as one, and the first moment of the pdfs is :math:`\Delta \rho` .
             This approximates the incompressible Navier-Stokes equations better than the standard
             compressible model.
    :param forceModel: force model instance, or None if no external forces
    :param equilibriumAccuracyOrder: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
    :return: :class:`lbmpy.methods.MomentBasedLbmMethod` instance
    """
    dim = len(stencil[0])
    moments = exponentsToPolynomialRepresentations(momentsUpToComponentOrder(2, dim=dim))
    rrDict = {m: relaxationRate for m in moments}
    return createWithDiscreteMaxwellianEqMoments(stencil, rrDict, compressible, forceModel, equilibriumAccuracyOrder)


def createTRT(stencil, relaxationRateEvenMoments, relaxationRateOddMoments, compressible=False,
              forceModel=None, equilibriumAccuracyOrder=2):
    """
    Creates a two relaxation time (TRT) lattice Boltzmann model, where even and odd moments are relaxed differently.
    In the SRT model the exact wall position of no-slip boundaries depends on the viscosity, the TRT method does not
    have this problem.

    Parameters are similar to :func:`lbmpy.methods.createSRT`, but instead of one relaxation rate there are
    two relaxation rates: one for even moments (determines viscosity) and one for odd moments.
    If unsure how to choose the odd relaxation rate, use the function :func:`lbmpy.methods.createTRTWithMagicNumber`.
    """
    dim = len(stencil[0])
    moments = exponentsToPolynomialRepresentations(momentsUpToComponentOrder(2, dim=dim))
    rrDict = {m: relaxationRateEvenMoments if isEven(m) else relaxationRateOddMoments for m in moments}
    return createWithDiscreteMaxwellianEqMoments(stencil, rrDict, compressible, forceModel, equilibriumAccuracyOrder)


def createTRTWithMagicNumber(stencil, relaxationRate, magicNumber=sp.Rational(3, 16), *args, **kwargs):
    """
    Creates a two relaxation time (TRT) lattice Boltzmann method, where the relaxation time for odd moments is
    determines from the even moment relaxation time and a "magic number".
    For possible parameters see :func:`lbmpy.methods.createTRT`
    """
    rrOdd = relaxationRateFromMagicNumber(relaxationRate, magicNumber)
    return createTRT(stencil, relaxationRateEvenMoments=relaxationRate, relaxationRateOddMoments=rrOdd, *args, **kwargs)


def createOrthogonalMRT(stencil, relaxationRateGetter=None, compressible=False,
                        forceModel=None, equilibriumAccuracyOrder=2):
    r"""
    Returns a orthogonal multi-relaxation time model for the stencils D2Q9, D3Q15, D3Q19 and D3Q27.
    These MRT methods are just one specific version - there are many MRT methods possible for all these stencils
    which differ by the linear combination of moments and the grouping into equal relaxation times.
    To create a generic MRT method use :func:`lbmpy.methods.createWithDiscreteMaxwellianEqMoments`

    :param stencil: nested tuple defining the discrete velocity space. See `func:lbmpy.stencils.getStencil`
    :param relaxationRateGetter: function getting a list of moments as argument, returning the associated relaxation
                                 time. The default returns:

                                    - 0 for moments of order 0 and 1 (conserved)
                                    - :math:`\omega`: from moments of order 2 (rate that determines viscosity)
                                    - numbered :math:`\omega_i` for the rest

    :param compressible: see :func:`lbmpy.methods.createWithDiscreteMaxwellianEqMoments`
    :param forceModel:  see :func:`lbmpy.methods.createWithDiscreteMaxwellianEqMoments`
    :param equilibriumAccuracyOrder:  see :func:`lbmpy.methods.createWithDiscreteMaxwellianEqMoments`
    """
    if relaxationRateGetter is None:
        relaxationRateGetter = defaultRelaxationRateNames()

    Q = len(stencil)
    D = len(stencil[0])
    x, y, z = MOMENT_SYMBOLS
    one = sp.Rational(1, 1)

    momentToRelaxationRateDict = OrderedDict()
    if (D, Q) == (2, 9):
        moments = exponentsToPolynomialRepresentations(momentsUpToComponentOrder(2, dim=D))
        orthogonalMoments = gramSchmidt(moments, stencil)
        orthogonalMomentsScaled = [e * commonDenominator(e) for e in orthogonalMoments]
        nestedMoments = list(sortMomentsIntoGroupsOfSameOrder(orthogonalMomentsScaled).values())
    elif (D, Q) == (3, 15):
        sq = x ** 2 + y ** 2 + z ** 2
        nestedMoments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 10, 11, 12, 13]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 6 * sq + 1],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [x * y * z]
        ]
    elif (D, Q) == (3, 19):
        sq = x ** 2 + y ** 2 + z ** 2
        nestedMoments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 11, 13, 14, 15]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 6 * sq + 1],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [(2 * sq - 3) * (3 * x ** 2 - sq), (2 * sq - 3) * (y ** 2 - z ** 2)],  # [10, 12]
            [(y ** 2 - z ** 2) * x, (z ** 2 - x ** 2) * y, (x ** 2 - y ** 2) * z]  # [16, 17, 18]
        ]
    elif (D, Q) == (3, 27):
        xsq, ysq, zsq = x ** 2, y ** 2, z ** 2
        allMoments = [
            sp.Rational(1, 1),  # 0
            x, y, z,  # 1, 2, 3
            x * y, x * z, y * z,  # 4, 5, 6
            xsq - ysq,  # 7
            (xsq + ysq + zsq) - 3 * zsq,  # 8
            (xsq + ysq + zsq) - 2,  # 9
            3 * (x * ysq + x * zsq) - 4 * x,  # 10
            3 * (xsq * y + y * zsq) - 4 * y,  # 11
            3 * (xsq * z + ysq * z) - 4 * z,  # 12
            x * ysq - x * zsq,  # 13
            xsq * y - y * zsq,  # 14
            xsq * z - ysq * z,  # 15
            x * y * z,  # 16
            3 * (xsq * ysq + xsq * zsq + ysq * zsq) - 4 * (xsq + ysq + zsq) + 4,  # 17
            3 * (xsq * ysq + xsq * zsq - 2 * ysq * zsq) - 2 * (2 * xsq - ysq - zsq),  # 18
            3 * (xsq * ysq - xsq * zsq) - 2 * (ysq - zsq),  # 19
            3 * (xsq * y * z) - 2 * (y * z),  # 20
            3 * (x * ysq * z) - 2 * (x * z),  # 21
            3 * (x * y * zsq) - 2 * (x * y),  # 22
            9 * (x * ysq * zsq) - 6 * (x * ysq + x * zsq) + 4 * x,  # 23
            9 * (xsq * y * zsq) - 6 * (xsq * y + y * zsq) + 4 * y,  # 24
            9 * (xsq * ysq * z) - 6 * (xsq * z + ysq * z) + 4 * z,  # 25
            27 * (xsq * ysq * zsq) - 18 * (xsq * ysq + xsq * zsq + ysq * zsq) + 12 * (xsq + ysq + zsq) - 8,  # 26
        ]
        nestedMoments = list(sortMomentsIntoGroupsOfSameOrder(allMoments).values())
    else:
        raise NotImplementedError("No MRT model is available (yet) for this stencil. "
                                  "Create a custom MRT using 'createWithDiscreteMaxwellianEqMoments'")

    for momentList in nestedMoments:
        rr = relaxationRateGetter(momentList)
        for m in momentList:
            momentToRelaxationRateDict[m] = rr

    return createWithDiscreteMaxwellianEqMoments(stencil, momentToRelaxationRateDict, compressible, forceModel,
                                                 equilibriumAccuracyOrder)

