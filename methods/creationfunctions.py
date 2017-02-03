import sympy as sp
from collections import OrderedDict, defaultdict
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod
from lbmpy.methods.momentbased import MomentBasedLbMethod
from lbmpy.stencils import stencilsHaveSameEntries, getStencil
from lbmpy.moments import isEven, gramSchmidt, getDefaultMomentSetForStencil, MOMENT_SYMBOLS, getOrder, isShearMoment
from pystencils.sympyextensions import commonDenominator
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.methods.abstractlbmethod import RelaxationInfo
from lbmpy.maxwellian_equilibrium import getMomentsOfDiscreteMaxwellianEquilibrium, \
    getMomentsOfContinuousMaxwellianEquilibrium, getCumulantsOfDiscreteMaxwellianEquilibrium, \
    getCumulantsOfContinuousMaxwellianEquilibrium


def createWithDiscreteMaxwellianEqMoments(stencil, momentToRelaxationRateDict, compressible=False, forceModel=None,
                                          equilibriumAccuracyOrder=2, cumulant=False):
    r"""
    Creates a moment-based LBM by taking a list of moments with corresponding relaxation rate. These moments are
    relaxed against the moments of the discrete Maxwellian distribution.

    :param stencil: nested tuple defining the discrete velocity space. See `func:lbmpy.stencils.getStencil`
    :param momentToRelaxationRateDict: dict that has as many entries as the stencil. Each moment, which can be
                                       represented by an exponent tuple or in polynomial form
                                       (see `lbmpy.moments`), is mapped to a relaxation rate.
    :param compressible: incompressible LBM methods split the density into :math:`\rho = \rho_0 + \Delta \rho`
         where :math:`\rho_0` is chosen as one, and the first moment of the pdfs is :math:`\Delta \rho` .
         This approximates the incompressible Navier-Stokes equations better than the standard
         compressible model.
    :param forceModel: force model instance, or None if no external forces
    :param equilibriumAccuracyOrder: approximation order of macroscopic velocity :math:`\mathbf{u}` in the equilibrium
    :param cumulant: if True relax cumulants instead of moments
    :return: :class:`lbmpy.methods.MomentBasedLbmMethod` instance
    """
    momToRrDict = OrderedDict(momentToRelaxationRateDict)
    assert len(momToRrDict) == len(stencil), \
        "The number of moments has to be the same as the number of stencil entries"

    densityVelocityComputation = DensityVelocityComputation(stencil, compressible, forceModel)

    if cumulant:
        eqValues = getCumulantsOfDiscreteMaxwellianEquilibrium(stencil, list(momToRrDict.keys()),
                                                               c_s_sq=sp.Rational(1, 3), compressible=compressible,
                                                               order=equilibriumAccuracyOrder)
    else:
        eqValues = getMomentsOfDiscreteMaxwellianEquilibrium(stencil, list(momToRrDict.keys()),
                                                             c_s_sq=sp.Rational(1, 3), compressible=compressible,
                                                             order=equilibriumAccuracyOrder)

    rrDict = OrderedDict([(mom, RelaxationInfo(eqMom, rr))
                          for mom, rr, eqMom in zip(momToRrDict.keys(), momToRrDict.values(), eqValues)])
    if cumulant:
        return CumulantBasedLbMethod(stencil, rrDict, densityVelocityComputation, forceModel)
    else:
        return MomentBasedLbMethod(stencil, rrDict, densityVelocityComputation, forceModel)


def createWithContinuousMaxwellianEqMoments(stencil, momentToRelaxationRateDict, compressible=False, forceModel=None,
                                            equilibriumAccuracyOrder=2, cumulant=False):
    r"""
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

    if cumulant:
        eqValues = getCumulantsOfContinuousMaxwellianEquilibrium(list(momToRrDict.keys()), dim,
                                                                 c_s_sq=sp.Rational(1, 3),
                                                                 order=equilibriumAccuracyOrder)
    else:
        eqValues = getMomentsOfContinuousMaxwellianEquilibrium(list(momToRrDict.keys()), dim, c_s_sq=sp.Rational(1, 3),
                                                               order=equilibriumAccuracyOrder)

    if not compressible:
        rho = densityVelocityComputation.definedSymbols(order=0)[1]
        u = densityVelocityComputation.definedSymbols(order=1)[1]
        eqValues = [compressibleToIncompressibleMomentValue(em, rho, u) for em in eqValues]

    rrDict = OrderedDict([(mom, RelaxationInfo(eqMom, rr))
                          for mom, rr, eqMom in zip(momToRrDict.keys(), momToRrDict.values(), eqValues)])
    if cumulant:
        return CumulantBasedLbMethod(stencil, rrDict, densityVelocityComputation, forceModel)
    else:
        return MomentBasedLbMethod(stencil, rrDict, densityVelocityComputation, forceModel)


# ------------------------------------ SRT / TRT/ MRT Creators ---------------------------------------------------------


def createSRT(stencil, relaxationRate, useContinuousMaxwellianEquilibrium=False, **kwargs):
    r"""
    Creates a single relaxation time (SRT) lattice Boltzmann model also known as BGK model.

    :param stencil: nested tuple defining the discrete velocity space. See :func:`lbmpy.stencils.getStencil`
    :param relaxationRate: relaxation rate (inverse of the relaxation time)
                           usually called :math:`\omega` in LBM literature
    :return: :class:`lbmpy.methods.MomentBasedLbmMethod` instance
    """
    moments = getDefaultMomentSetForStencil(stencil)
    rrDict = OrderedDict([(m, relaxationRate) for m in moments])
    if useContinuousMaxwellianEquilibrium:
        return createWithContinuousMaxwellianEqMoments(stencil, rrDict,  **kwargs)
    else:
        return createWithDiscreteMaxwellianEqMoments(stencil, rrDict, **kwargs)


def createTRT(stencil, relaxationRateEvenMoments, relaxationRateOddMoments,
              useContinuousMaxwellianEquilibrium=False, **kwargs):
    """
    Creates a two relaxation time (TRT) lattice Boltzmann model, where even and odd moments are relaxed differently.
    In the SRT model the exact wall position of no-slip boundaries depends on the viscosity, the TRT method does not
    have this problem.

    Parameters are similar to :func:`lbmpy.methods.createSRT`, but instead of one relaxation rate there are
    two relaxation rates: one for even moments (determines viscosity) and one for odd moments.
    If unsure how to choose the odd relaxation rate, use the function :func:`lbmpy.methods.createTRTWithMagicNumber`.
    """
    moments = getDefaultMomentSetForStencil(stencil)
    rrDict = OrderedDict([(m, relaxationRateEvenMoments if isEven(m) else relaxationRateOddMoments) for m in moments])
    if useContinuousMaxwellianEquilibrium:
        return createWithContinuousMaxwellianEqMoments(stencil, rrDict,  **kwargs)
    else:
        return createWithDiscreteMaxwellianEqMoments(stencil, rrDict, **kwargs)


def createTRTWithMagicNumber(stencil, relaxationRate, magicNumber=sp.Rational(3, 16), **kwargs):
    """
    Creates a two relaxation time (TRT) lattice Boltzmann method, where the relaxation time for odd moments is
    determines from the even moment relaxation time and a "magic number".
    For possible parameters see :func:`lbmpy.methods.createTRT`
    """
    rrOdd = relaxationRateFromMagicNumber(relaxationRate, magicNumber)
    return createTRT(stencil, relaxationRateEvenMoments=relaxationRate, relaxationRateOddMoments=rrOdd, **kwargs)


def createOrthogonalMRT(stencil, relaxationRateGetter=None, useContinuousMaxwellianEquilibrium=False, **kwargs):
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
    """
    if relaxationRateGetter is None:
        relaxationRateGetter = defaultRelaxationRateNames()

    x, y, z = MOMENT_SYMBOLS
    one = sp.Rational(1, 1)

    momentToRelaxationRateDict = OrderedDict()
    if stencilsHaveSameEntries(stencil, getStencil("D2Q9")):
        moments = getDefaultMomentSetForStencil(stencil)
        orthogonalMoments = gramSchmidt(moments, stencil)
        orthogonalMomentsScaled = [e * commonDenominator(e) for e in orthogonalMoments]
        nestedMoments = list(sortMomentsIntoGroupsOfSameOrder(orthogonalMomentsScaled).values())
    elif stencilsHaveSameEntries(stencil, getStencil("D3Q15")):
        sq = x ** 2 + y ** 2 + z ** 2
        nestedMoments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 6 * sq + 1],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 10, 11, 12, 13]
            [x * y * z]
        ]
    elif stencilsHaveSameEntries(stencil, getStencil("D3Q19")):
        sq = x ** 2 + y ** 2 + z ** 2
        nestedMoments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 6 * sq + 1],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 11, 13, 14, 15]
            [(2 * sq - 3) * (3 * x ** 2 - sq), (2 * sq - 3) * (y ** 2 - z ** 2)],  # [10, 12]
            [(y ** 2 - z ** 2) * x, (z ** 2 - x ** 2) * y, (x ** 2 - y ** 2) * z]  # [16, 17, 18]
        ]
    elif stencilsHaveSameEntries(stencil, getStencil("D3Q27")):
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

    if useContinuousMaxwellianEquilibrium:
        return createWithContinuousMaxwellianEqMoments(stencil, momentToRelaxationRateDict, **kwargs)
    else:
        return createWithDiscreteMaxwellianEqMoments(stencil, momentToRelaxationRateDict, **kwargs)


# ----------------------------------------- Comparison view for notebooks ----------------------------------------------


def compareMomentBasedLbMethods(reference, other):
    import ipy_table
    table = []
    captionRows = [len(table)]
    table.append(['Shared Moment', 'ref', 'other', 'difference'])

    referenceMoments = set(reference.moments)
    otherMoments = set(other.moments)
    for moment in referenceMoments.intersection(otherMoments):
        referenceValue = reference.momentToRelaxationInfoDict[moment].equilibriumValue
        otherValue = other.momentToRelaxationInfoDict[moment].equilibriumValue
        diff = sp.simplify(referenceValue - otherValue)
        table.append(["$%s$" % (sp.latex(moment), ),
                      "$%s$" % (sp.latex(referenceValue), ),
                      "$%s$" % (sp.latex(otherValue), ),
                      "$%s$" % (sp.latex(diff),)])

    onlyInRef = referenceMoments - otherMoments
    if onlyInRef:
        captionRows.append(len(table))
        table.append(['Only in Ref', 'value', '', ''])
        for moment in onlyInRef:
            val = reference.momentToRelaxationInfoDict[moment].equilibriumValue
            table.append(["$%s$" % (sp.latex(moment),),
                          "$%s$" % (sp.latex(val),),
                          " ", " "])

    onlyInOther = otherMoments - referenceMoments
    if onlyInOther:
        captionRows.append(len(table))
        table.append(['Only in Other', '', 'value', ''])
        for moment in onlyInOther:
            val = other.momentToRelaxationInfoDict[moment].equilibriumValue
            table.append(["$%s$" % (sp.latex(moment),),
                          " ",
                          "$%s$" % (sp.latex(val),),
                          " "])

    tableDisplay = ipy_table.make_table(table)
    for rowIdx in captionRows:
        for col in range(4):
            ipy_table.set_cell_style(rowIdx, col, color='#bbbbbb')
    return tableDisplay


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


def compressibleToIncompressibleMomentValue(term, rho, u):
    term = term.expand()
    if term.func != sp.Add:
        args = [term, ]
    else:
        args = term.args

    res = 0
    for t in args:
        containedSymbols = t.atoms(sp.Symbol)
        if rho in containedSymbols and len(containedSymbols.intersection(set(u))) > 0:
            res += t / rho
        else:
            res += t
    return res
