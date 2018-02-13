import sympy as sp
from collections import defaultdict

from lbmpy.chapman_enskog.derivative import expandUsingLinearity, Diff, fullDiffExpand
from pystencils.equationcollection.simplifications import sympyCseOnEquationList
from pystencils.sympyextensions import multidimensionalSummation as multiSum, normalizeProduct, prod

orderParameterSymbolName = "phi"
surfaceTensionSymbolName = "tau"
interfaceWidthSymbol = sp.Symbol("alpha")


def symmetricSymbolicSurfaceTension(i, j):
    """Returns symbolic surface tension. The function is symmetric, i.e. interchanging i and j yields the same result.
    If both phase indices i and j are chosen equal, zero is returned"""
    if i == j:
        return 0
    index = (i, j) if i < j else (j, i)
    return sp.Symbol("%s_%d_%d" % ((surfaceTensionSymbolName, ) + index))


def symbolicOrderParameters(numSymbols):
    return sp.symbols("%s_:%i" % (orderParameterSymbolName, numSymbols))


def freeEnergyFunction3Phases(orderParameters=None, interfaceWidth=interfaceWidthSymbol, transformed=True,
                              includeBulk=True, includeInterface=True, expandDerivatives=True,
                              kappa=sp.symbols("kappa_:3")):
    kappaPrime = tuple(interfaceWidth**2 * k for k in kappa)
    C = sp.symbols("C_:3")

    bulkFreeEnergy = sum(k * C_i ** 2 * (1 - C_i) ** 2  / 2 for k, C_i in zip(kappa, C))
    surfaceFreeEnergy = sum(k * Diff(C_i) ** 2 / 2 for k, C_i in zip(kappaPrime, C))

    F = 0
    if includeBulk:
        F += bulkFreeEnergy
    if includeInterface:
        F += surfaceFreeEnergy

    if not transformed:
        return F

    if orderParameters:
        rho, phi, psi = orderParameters
    else:
        rho, phi, psi = sp.symbols("rho phi psi")
    rhoDef = C[0] + C[1] + C[2]
    phiDef = C[0] - C[1]
    psiDef = C[2]

    concentrationToOrderParamRelation = {rho: rhoDef, phi: phiDef, psi: psiDef}
    orderParamToConcentrationRelation = sp.solve([rhoDef - rho, phiDef - phi, psiDef - psi], C)

    F = F.subs(orderParamToConcentrationRelation)
    if expandDerivatives:
        F = expandUsingLinearity(F, functions=orderParameters)

    return F


def freeEnergyFunctionalNPhasesPenaltyTerm(orderParameters, interfaceWidth=interfaceWidthSymbol, kappa=None,
                                           penaltyTermFactor=0.01):
    numPhases = len(orderParameters)
    if kappa is None:
        kappa = sp.symbols("kappa_:%d" % (numPhases,))
    if not hasattr(kappa, "__length__"):
        kappa = [kappa] * numPhases

    def f(x):
        return x ** 2 * (1 - x) ** 2

    bulk = sum(f(c) * k / 2 for c, k in zip(orderParameters, kappa))
    interface = sum(Diff(c) ** 2 / 2 * interfaceWidth ** 2 * k
                    for c, k in zip(orderParameters, kappa))

    bulkPenaltyTerm = (1 - sum(c for c in orderParameters)) ** 2
    return bulk + interface + penaltyTermFactor * bulkPenaltyTerm


def freeEnergyFunctionalNPhases(numPhases=None, surfaceTensions=symmetricSymbolicSurfaceTension,
                                interfaceWidth=interfaceWidthSymbol, orderParameters=None,
                                includeBulk=True, includeInterface=True, symbolicLambda=False,
                                symbolicDependentVariable=False):
    r"""
    Returns a symbolic expression for the free energy of a system with N phases and
    specified surface tensions. The total free energy is the sum of a bulk and an interface component.

    .. math ::

        F_{bulk} = \int \frac{3}{\sqrt{2} \eta}
            \sum_{\substack{\alpha,\beta=0 \\ \alpha \neq \beta}}^{N-1}
            \frac{\tau(\alpha,\beta)}{2} \left[ f(\phi_\alpha) + f(\phi_\beta)
            - f(\phi_\alpha + \phi_\beta)  \right] \; d\Omega

        F_{interface} = \int \sum_{\alpha,\beta=0}^{N-2} \frac{\Lambda_{\alpha\beta}}{2}
                        \left( \nabla \phi_\alpha \cdot \nabla \phi_\beta \right)\; d\Omega

        \Lambda_{\alpha \beta} = \frac{3 \eta}{\sqrt{2}}  \left[ \tau(\alpha,N-1) + \tau(\beta,N-1) -
                                 \tau(\alpha,\beta)  \right]

        f(c) = c^2( 1-c)^2

    :param numPhases: number of phases, called N above
    :param surfaceTensions: surface tension function, called with two phase indices (two integers)
    :param interfaceWidth: called :math:`\eta` above, controls the interface width
    :param orderParameters: explicitly

    Parameter useful for viewing / debugging the function
    :param includeBulk: if false no bulk term is added
    :param includeInterface:if false no interface contribution is added
    :param symbolicLambda: surface energy coefficient is represented by symbol, not in expanded form
    :param symbolicDependentVariable: last phase variable is defined as 1-otherPhaseVars, if this is set to True
                                      it is represented by phi_A for better readability
    """
    assert not (numPhases is None and orderParameters is None)
    if orderParameters is None:
        phi = symbolicOrderParameters(numPhases-1)
    else:
        phi = orderParameters
        numPhases = len(phi) + 1

    if not symbolicDependentVariable:
        phi = tuple(phi) + (1 - sum(phi),)
    else:
        phi = tuple(phi) + (sp.Symbol("phi_D"), )

    # Compared to handwritten notes we scale the interface width parameter here to obtain the correct
    # equations for the interface profile and the surface tensions i.e. to pass tests
    # test_analyticInterfaceSolution and test_surfaceTensionDerivation
    interfaceWidth *= sp.sqrt(2)

    def f(c):
        return c ** 2 * (1 - c) ** 2

    def lambdaCoeff(k, l):
        if symbolicLambda:
            return sp.Symbol("Lambda_%d%d" % ((k, l) if k < l else (l, k)))
        N = numPhases - 1
        if k == l:
            assert surfaceTensions(l, l) == 0
        return 3 / sp.sqrt(2) * interfaceWidth * (surfaceTensions(k, N) + surfaceTensions(l, N) - surfaceTensions(k, l))

    def bulkTerm(i, j):
        return surfaceTensions(i, j) / 2 * (f(phi[i]) + f(phi[j]) - f(phi[i] + phi[j]))

    F_bulk = 3 / sp.sqrt(2) / interfaceWidth * sum(bulkTerm(i, j) for i, j in multiSum(2, numPhases) if i != j)
    F_interface = sum(lambdaCoeff(i, j) / 2 * Diff(phi[i]) * Diff(phi[j]) for i, j in multiSum(2, numPhases-1))

    result = 0
    if includeBulk:
        result += F_bulk
    if includeInterface:
        result += F_interface
    return result


def separateIntoBulkAndInterface(freeEnergy):
    """Separates the bulk and interface parts of a free energy

    >>> F = freeEnergyFunctionalNPhases(3)
    >>> bulk, inter = separateIntoBulkAndInterface(F)
    >>> assert sp.expand(bulk - freeEnergyFunctionalNPhases(3, includeInterface=False)) == 0
    >>> assert sp.expand(inter - freeEnergyFunctionalNPhases(3, includeBulk=False)) == 0
    """
    freeEnergy = freeEnergy.expand()
    bulkPart = freeEnergy.subs({a: 0 for a in freeEnergy.atoms(Diff)})
    interfacePart = freeEnergy - bulkPart
    return bulkPart, interfacePart


def analyticInterfaceProfile(x, interfaceWidth=interfaceWidthSymbol):
    """Analytic expression for a 1D interface normal to x with given interface width

    The following doctest shows that the returned analytical solution is indeed a solution of the ODE that we
    get from the condition :math:`\mu_0 = 0` (thermodynamic equilibrium) for a situation with only a single order
    parameter, i.e. at a transition between two phases.
    >>> numPhases = 4
    >>> x, phi = sp.Symbol("x"), symbolicOrderParameters(numPhases-1)
    >>> F = freeEnergyFunctionalNPhases(orderParameters=phi)
    >>> mu = chemicalPotentialsFromFreeEnergy(F)
    >>> mu0 = mu[0].subs({p: 0 for p in phi[1:]})  # mu[0] as function of one order parameter only
    >>> solution = analyticInterfaceProfile(x)
    >>> solutionSubstitution = {phi[0]: solution, Diff(Diff(phi[0])): sp.diff(solution, x, x) }
    >>> sp.expand(mu0.subs(solutionSubstitution))  # inserting solution should solve the mu_0=0 equation
    0
    """
    return (1 + sp.tanh(x / (2 * interfaceWidth))) / 2


def chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters=None):
    """Computes chemical potentials as functional derivative of free energy"""
    syms = freeEnergy.atoms(sp.Symbol)
    if orderParameters is None:
        orderParameters = [s for s in syms if s.name.startswith(orderParameterSymbolName)]
        orderParameters.sort(key=lambda e: e.name)
        orderParameters = orderParameters[:-1]
    constants = [s for s in syms if s not in orderParameters]
    return sp.Matrix([functionalDerivative(freeEnergy, op, constants) for op in orderParameters])


def forceFromPhiAndMu(orderParameters, dim, mu=None):
    if mu is None:
        mu = sp.symbols("mu_:%d" % (len(orderParameters),))

    return sp.Matrix([sum(- c_i * Diff(mu_i, a) for c_i, mu_i in zip(orderParameters, mu))
                      for a in range(dim)])


def substituteLaplacianBySum(eq, dim):
    """Substitutes abstract laplacian represented by ∂∂ by a sum over all dimensions
    i.e. in case of 3D: ∂∂ is replaced by ∂0∂0 + ∂1∂1 + ∂2∂2
    :param eq: the term where the substitutions should be made
    :param dim: spatial dimension, in example above, 3
    """
    functions = [d.args[0] for d in eq.atoms(Diff)]
    substitutions = {Diff(Diff(op)): sum(Diff(Diff(op, i), i) for i in range(dim))
                     for op in functions}
    return fullDiffExpand(eq.subs(substitutions))


def functionalDerivative(functional, v, constants=None):
    """
    Computes functional derivative of functional with respect to v using Euler-Lagrange equation

    .. math ::

        \frac{\delta F}{\delta v} =
                \frac{\partial F}{\partial v} - \nabla \cdot \frac{\partial F}{\partial \nabla v}

    - assumes that gradients are represented by Diff() node (from Chapman Enskog module)
    - Diff(Diff(r)) represents the divergence of r
    - the constants parameter is a list with symbols not affected by the derivative. This is used for simplification
      of the derivative terms.
    """
    functional = expandUsingLinearity(functional, constants=constants)
    diffs = functional.atoms(Diff)

    diffV = Diff(v)
    #assert diffV in diffs  # not necessary in general, but for this use case this should be true

    nonDiffPart = functional.subs({d: 0 for d in diffs})

    partialF_partialV = sp.diff(nonDiffPart, v)

    dummy = sp.Dummy()
    partialF_partialGradV = functional.subs(diffV, dummy).diff(dummy).subs(dummy, diffV)

    result = partialF_partialV - Diff(partialF_partialGradV)
    return expandUsingLinearity(result, constants=constants)


def coshIntegral(f, var):
    """Integrates a function f that has exactly one cosh term, from -oo to oo, by
    substituting a new helper variable for the cosh argument"""
    coshTerm = list(f.atoms(sp.cosh))
    assert len(coshTerm) == 1
    integral = sp.Integral(f, var)
    transformedInt = integral.transform(coshTerm[0].args[0], sp.Symbol("u", real=True))
    return sp.integrate(transformedInt.args[0], (transformedInt.args[1][0], -sp.oo, sp.oo))


def finiteDifferences2ndOrder(term, dx=1):
    """Substitutes symbolic integral of field access by second order accurate finite differences.
    The only valid argument of Diff objects are field accesses (usually center field accesses)"""
    def diffOrder(e):
        if not isinstance(e, Diff):
            return 0
        else:
            return 1 + diffOrder(e.args[0])

    def visit(e):
        order = diffOrder(e)
        if order == 0:
            paramList = [visit(a) for a in e.args]
            return e if not paramList else e.func(*paramList)
        elif order == 1:
            fa = e.args[0]
            index = e.label
            return (fa.neighbor(index, 1) - fa.neighbor(index, -1)) / (2 * dx)
        elif order == 2:
            indices = sorted([e.label, e.args[0].label])
            fa = e.args[0].args[0]
            if indices[0] == indices[1]:
                result = (-2 * fa + fa.neighbor(indices[0], -1) + fa.neighbor(indices[0], +1))
            else:
                offsets = [(1,1), [-1, 1], [1, -1], [-1, -1]]
                result = sum(o1*o2 * fa.neighbor(indices[0], o1).neighbor(indices[1], o2) for o1, o2 in offsets) / 4
            return result / (dx**2)
        else:
            raise NotImplementedError("Term contains derivatives of order > 2")

    return visit(term)


def symmetricTensorLinearization(dim):
    nextIdx = 0
    resultMap = {}
    for idx in multiSum(2, dim):
        idx = tuple(sorted(idx))
        if idx in resultMap:
            continue
        else:
            resultMap[idx] = nextIdx
            nextIdx += 1
    return resultMap

# ----------------------------------------- Pressure Tensor ------------------------------------------------------------


def extractGamma(freeEnergy, orderParameters):
    """Extracts parameters before the gradient terms"""
    result = defaultdict(lambda: 0)
    freeEnergy = freeEnergy.expand()
    assert freeEnergy.func == sp.Add
    for product in freeEnergy.args:
        product = normalizeProduct(product)
        diffFactors = [e for e in product if e.func == Diff]
        if len(diffFactors) == 0:
            continue

        if len(diffFactors) != 2:
            raise ValueError("Could not extract Λ because of term " + str(product))

        indices = sorted([orderParameters.index(d.args[0]) for d in diffFactors])
        result[tuple(indices)] += prod(e for e in product if e.func != Diff)
        if diffFactors[0] == diffFactors[1]:
            result[tuple(indices)] *= 2
    return result


def pressureTensorBulkComponent(freeEnergy, orderParameters):
    """Diagonal component of pressure tensor in bulk"""
    bulkFreeEnergy, _ = separateIntoBulkAndInterface(freeEnergy)
    muBulk = chemicalPotentialsFromFreeEnergy(bulkFreeEnergy, orderParameters)
    return sum(c_i * mu_i for c_i, mu_i in zip(orderParameters, muBulk)) - bulkFreeEnergy


def pressureTensorInterfaceComponent(freeEnergy, orderParameters, dim, a, b):
    gamma = extractGamma(freeEnergy, orderParameters)
    d = Diff
    result = 0
    for i, c_i in enumerate(orderParameters):
        for j, c_j in enumerate(orderParameters):
            t = d(c_i, a) * d(c_j, b) + d(c_i, b) * d(c_j, a)
            if a == b:
                t -= sum(d(c_i, g) * d(c_j, g) for g in range(dim))
                t -= sum(c_i * d(d(c_j, g), g) for g in range(dim))
                t -= sum(c_j * d(d(c_i, g), g) for g in range(dim))
            gamma_ij = gamma[(i, j)] if i < j else gamma[(j, i)]
            result += t * gamma_ij / 2
    return result


def pressureTensorFromFreeEnergy(freeEnergy, orderParameters, dim):
    def getEntry(i, j):
        pIf = pressureTensorInterfaceComponent(freeEnergy, orderParameters, dim, i, j)
        pB = pressureTensorBulkComponent(freeEnergy, orderParameters) if i == j else 0
        return sp.expand(pIf + pB)

    return sp.Matrix(dim, dim, getEntry)


def forceFromPressureTensor(pressureTensor, functions=None):
    assert len(pressureTensor.shape) == 2 and pressureTensor.shape[0] == pressureTensor.shape[1]
    dim = pressureTensor.shape[0]

    def forceComponent(b):
        r = -sum(Diff(pressureTensor[a, b], a) for a in range(dim))
        r = fullDiffExpand(r, functions=functions)
        return r

    return sp.Matrix([forceComponent(b) for b in range(dim)])