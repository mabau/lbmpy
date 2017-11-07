import sympy as sp
import numpy as np

from lbmpy.boundaries.periodicityhandling import PeriodicityHandling
from lbmpy.creationfunctions import updateWithDefaultParameters, createLatticeBoltzmannFunction, \
    createLatticeBoltzmannUpdateRule, createLatticeBoltzmannAst
from lbmpy.macroscopic_value_kernels import compileMacroscopicValuesSetter
from lbmpy.stencils import getStencil
from pystencils.equationcollection.simplifications import sympyCseOnEquationList
from pystencils.field import Field, getLayoutOfArray
from pystencils.slicing import addGhostLayers, removeGhostLayers
from pystencils.sympyextensions import kroneckerDelta, multidimensionalSummation
from lbmpy.maxwellian_equilibrium import getWeights
from lbmpy.chapman_enskog.derivative import expandUsingLinearity, Diff
from lbmpy.methods.creationfunctions import createFromEquilibrium
from lbmpy.moments import getDefaultMomentSetForStencil
from lbmpy.updatekernels import createPdfArray
from pystencils.sympyextensions import multidimensionalSummation as multiSum

orderParameterSymbolName = "phi"
surfaceTensionSymbolName = "tau"
interfaceWidthSymbol = sp.Symbol("eta")


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
    assert diffV in diffs  # not necessary in general, but for this use case this should be true

    nonDiffPart = functional.subs({d: 0 for d in diffs})

    partialF_partialV = sp.diff(nonDiffPart, v)

    dummy = sp.Dummy()
    partialF_partialGradV = functional.subs(diffV, dummy).diff(dummy).subs(dummy, diffV)

    result = partialF_partialV - Diff(partialF_partialGradV)
    return expandUsingLinearity(result, constants=constants)


def discreteLaplace(field, index, dx):
    """Returns second order Laplace stencil"""
    dim = field.spatialDimensions
    count = 0
    result = 0
    for d in range(dim):
        for offset in (-1, 1):
            count += 1
            result += field.neighbor(d, offset)(index)

    result -= count * field.center()(index)
    result /= dx ** 2
    return result


def symmetricSymbolicSurfaceTension(i, j):
    """Returns symbolic surface tension. The function is symmetric, i.e. interchanging i and j yields the same result.
    If both phase indices i and j are chosen equal, zero is returned"""
    if i == j:
        return 0
    index = (i, j) if i < j else (j, i)
    return sp.Symbol("%s_%d_%d" % ((surfaceTensionSymbolName, ) + index))


def symbolicOrderParameters(numPhases):
    """
    Returns a tuple with numPhases entries, where the all but the last are numbered symbols and the last entry
    is 1 - others
    """
    phi = sp.symbols("%s_:%i" % (orderParameterSymbolName, numPhases-1))
    phi = phi + (1 - sum(phi),)  # choose last order parameter as 1 - sum(others)
    return phi


def freeEnergyFunctional(numPhases, surfaceTensions=symmetricSymbolicSurfaceTension,
                         interfaceWidth=interfaceWidthSymbol, orderParameters=None):
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
    """
    if orderParameters is None:
        phi = symbolicOrderParameters(numPhases)
    else:
        phi = orderParameters

    tauFactor = sp.Rational(1, 2)  # originally this was 1 / sp.sqrt(2)

    def f(c):
        return c ** 2 * (1 - c) ** 2

    def lambdaCoeff(k, l):
        N = numPhases - 1
        if k == l:
            assert surfaceTensions(l, l) == 0
        return 6 * tauFactor * interfaceWidth * (surfaceTensions(k, N) + surfaceTensions(l, N) - surfaceTensions(k, l))

    def bulkTerm(i, j):
        return surfaceTensions(i, j) / 2 * (f(phi[i]) + f(phi[j]) - f(phi[i] + phi[j]))

    F_bulk = 3 * tauFactor / interfaceWidth * sum(bulkTerm(i, j) for i, j in multiSum(2, numPhases) if i != j)
    F_interface = sum(lambdaCoeff(i, j) / 2 * Diff(phi[i]) * Diff(phi[j]) for i, j in multiSum(2, numPhases))

    return F_bulk + F_interface


def analyticInterfaceProfile(x, interfaceWidth=interfaceWidthSymbol):
    """Analytic expression for a 1D interface normal to x with given interface width

    The following doctest shows that the returned analytical solution is indeed a solution of the ODE that we
    get from the condition :math:`\mu_0 = 0` (thermodynamic equilibrium) for a situation with only a single order
    parameter, i.e. at a transition between two phases.
    >>> numPhases = 4
    >>> x, phi = sp.Symbol("x"), symbolicOrderParameters(numPhases)
    >>> F = freeEnergyFunctional(numPhases)
    >>> mu = chemicalPotentialsFromFreeEnergy(F)
    >>> mu0 = mu[0].subs({p: 0 for p in phi[1:-1]})  # mu[0] as function of one order parameter only
    >>> solution = analyticInterfaceProfile(x)
    >>> solutionSubstitution = {phi[0]: solution, Diff(Diff(phi[0])): sp.diff(solution, x, x) }
    >>> sp.expand(mu0.subs(solutionSubstitution))  # inserting solution should solve the mu_0=0 equation
    0
    """
    return (1 + sp.tanh(x / (2 * interfaceWidth))) / 2


def chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters=None):
    """
    Computes chemical potentials as functional derivative of free energy
    """
    syms = freeEnergy.atoms(sp.Symbol)
    if orderParameters is None:
        orderParameters = [s for s in syms if s.name.startswith(orderParameterSymbolName)]
        orderParameters.sort(key=lambda e: e.name)
    constants = [s for s in syms if not s.name.startswith(orderParameterSymbolName)]
    return sp.Matrix([functionalDerivative(freeEnergy, op, constants) for op in orderParameters[:-1]])


def createCahnHilliardEquilibrium(stencil, mu, gamma=1):
    """Returns LB equilibrium that solves the Cahn Hilliard equation

    ..math ::

        \partial_t \phi + \partial_i ( \phi v_i ) = M \nabla^2 \mu

    :param gamma: tunable parameter affecting the second order equilibrium moment
    """
    weights = getWeights(stencil, c_s_sq=sp.Rational(1, 3))

    kd = kroneckerDelta

    def s(*args):
        for r in multidimensionalSummation(*args, dim=len(stencil[0])):
            yield r

    op = sp.Symbol("rho")
    v = sp.symbols("u_:%d" % (len(stencil[0]),))

    equilibrium = []
    for d, w in zip(stencil, weights):
        c_s = sp.sqrt(sp.Rational(1, 3))
        result = gamma * mu / (c_s ** 2)
        result += op * sum(d[i] * v[i] for i, in s(1)) / (c_s ** 2)
        result += op * sum(v[i] * v[j] * (d[i] * d[j] - c_s ** 2 * kd(i, j)) for i, j in s(2)) / (2 * c_s ** 4)
        equilibrium.append(w * result)

    rho = sp.Symbol("rho")
    equilibrium[0] = rho - sp.expand(sum(equilibrium[1:]))
    return tuple(equilibrium)


def createCahnHilliardLbFunction(stencil, relaxationRate, velocityField, mu, orderParameterOut,
                                 optimizationParams, gamma=1):
    """
    Update rule for a LB scheme that solves Cahn-Hilliard.

    :param stencil:
    :param relaxationRate: relaxation rate controls the mobility
    :param velocityField: velocity field (output from N-S LBM)
    :param mu: chemical potential field
    :param orderParameterOut: field where order parameter :math:`\phi` is written to
    :param gamma: tunable equilibrium parameter
    """
    equilibrium = createCahnHilliardEquilibrium(stencil, mu, gamma)
    rrRates = {m: relaxationRate for m in getDefaultMomentSetForStencil(stencil)}
    method = createFromEquilibrium(stencil, tuple(equilibrium), rrRates, compressible=True)

    updateRule = createLatticeBoltzmannUpdateRule(method, optimizationParams,
                                                  output={'density': orderParameterOut},
                                                  velocityInput=velocityField)

    ast = createLatticeBoltzmannAst(updateRule=updateRule, optimizationParams=optimizationParams)
    return createLatticeBoltzmannFunction(ast=ast, optimizationParams=optimizationParams)


def createChemicalPotentialEvolutionEquations(freeEnergy, orderParameters, phiField, muField, dx=1):
    """Reads from order parameter (phi) field and updates chemical potentials"""
    chemicalPotential = chemicalPotentialsFromFreeEnergy(freeEnergy, orderParameters)
    laplaceDiscretization = {Diff(Diff(op)): discreteLaplace(phiField, i, dx)
                             for i, op in enumerate(orderParameters[:-1])}
    chemicalPotential = chemicalPotential.subs(laplaceDiscretization)
    chemicalPotential = chemicalPotential.subs({op: phiField(i) for i, op in enumerate(orderParameters[:-1])})

    muSweepEqs = [sp.Eq(muField(i), cp) for i, cp in enumerate(chemicalPotential)]
    return sympyCseOnEquationList(muSweepEqs)


def createForceUpdateEquations(numPhases, forceField, phiField, muField, dx=1):
    forceSweepEqs = []
    dim = phiField.spatialDimensions
    for d in range(dim):
        rhs = 0
        for i in range(numPhases - 1):
            rhs -= phiField(i) * (muField.neighbor(d, 1)(i) - muField.neighbor(d, -1)(i)) / (2 * dx)
        forceSweepEqs.append(sp.Eq(forceField(d), rhs))
    return forceSweepEqs


class PhasefieldScenario(object):
    def __init__(self, domainSize, numPhases, mobilityRelaxationRates=1.1,
                 surfaceTensionCallback=lambda i, j: 1e-3 if i !=j else 0, interfaceWidth=3, dx=1, gamma=1,
                 optimizationParams={}, initialVelocity=None, kernelParams={}, **kwargs):

        self.numPhases = numPhases
        self.timeStepsRun = 0
        self.domainSize = domainSize

        # ---- Parameter normalization
        if not hasattr(mobilityRelaxationRates, '__len__'):
            mobilityRelaxationRates = [mobilityRelaxationRates] * numPhases

        D = len(domainSize)

        ghostLayers = 1
        domainSizeWithGhostLayer = tuple([s + 2 * ghostLayers for s in domainSize])

        if 'stencil' not in kwargs:
            kwargs['stencil'] = 'D2Q9' if D == 2 else 'D3Q27'

        methodParameters, optimizationParams = updateWithDefaultParameters(kwargs, optimizationParams)

        stencil = getStencil(methodParameters['stencil'])
        fieldLayout = optimizationParams['fieldLayout']
        Q = len(stencil)

        if isinstance(initialVelocity, np.ndarray):
            assert initialVelocity.shape[-1] == D
            initialVelocity = addGhostLayers(initialVelocity, indexDimensions=1, ghostLayers=1,
                                             layout=getLayoutOfArray(self._pdfArrays[0]))
        elif initialVelocity is None:
            initialVelocity = [0] * D

        self.kernelParams = kernelParams

        # ---- Arrays
        self.velArr = np.zeros(domainSizeWithGhostLayer + (D,), order=fieldLayout)
        self.muArr = np.zeros(domainSizeWithGhostLayer + (numPhases - 1,), order=fieldLayout)
        self.phiArr = np.zeros(domainSizeWithGhostLayer + (numPhases - 1,), order=fieldLayout)
        self.forceArr = np.zeros(domainSizeWithGhostLayer + (D,), order=fieldLayout)

        self._pdfArrays = [[createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout'])
                            for i in range(numPhases)],
                           [createPdfArray(domainSize, Q, layout=optimizationParams['fieldLayout'])
                            for i in range(numPhases)]]

        # ---- Fields
        velField = Field.createFromNumpyArray('vel', self.velArr, indexDimensions=1)
        muField = Field.createFromNumpyArray('mu', self.muArr, indexDimensions=1)
        phiField = Field.createFromNumpyArray('phi', self.phiArr, indexDimensions=1)
        forceField = Field.createFromNumpyArray('F', self.forceArr, indexDimensions=1)

        orderParameters = symbolicOrderParameters(numPhases)
        freeEnergy = freeEnergyFunctional(numPhases, surfaceTensionCallback, interfaceWidth, orderParameters)

        # ---- Sweeps
        muSweepEquations = createChemicalPotentialEvolutionEquations(freeEnergy, orderParameters, phiField, muField, dx)
        forceSweepEquations = createForceUpdateEquations(numPhases, forceField, phiField, muField, dx)
        if optimizationParams['target'] == 'cpu':
            from pystencils.cpu import createKernel, makePythonFunction
            self.muSweep = makePythonFunction(createKernel(muSweepEquations))
            self.forceSweep = makePythonFunction(createKernel(forceSweepEquations))
        else:
            from pystencils.gpucuda import createCUDAKernel, makePythonFunction
            self.muSweep = makePythonFunction(createCUDAKernel(muSweepEquations))
            self.forceSweep = makePythonFunction(createCUDAKernel(forceSweepEquations))

        optimizationParams['pdfArr'] = self._pdfArrays[0][0]

        self.lbSweepHydro = createLatticeBoltzmannFunction(force=[forceField(i) for i in range(D)],
                                                           output={'velocity': velField},
                                                           optimizationParams=optimizationParams, **kwargs)
        self.lbSweepsCH = [createCahnHilliardLbFunction(stencil, mobilityRelaxationRates[i],
                                                        velField, muField(i), phiField(i), optimizationParams, gamma)
                           for i in range(numPhases-1)]

        self.lbSweeps = [self.lbSweepHydro] + self.lbSweepsCH

        self._pdfPeriodicityHandler = PeriodicityHandling(self._pdfArrays[0][0].shape, (True, True, True),
                                                          optimizationParams['target'])

        assert self.muArr.shape == self.phiArr.shape
        self._muPhiPeriodicityHandler = PeriodicityHandling(self.muArr.shape, (True, True, True),
                                                            optimizationParams['target'])

        # Pdf array initialization
        hydroLbmInit = compileMacroscopicValuesSetter(self.lbSweepHydro.method,
                                                      {'density': 1.0, 'velocity': initialVelocity},
                                                      pdfArr=self._pdfArrays[0][0], target='cpu')
        hydroLbmInit(pdfs=self._pdfArrays[0][0], F=self.forceArr, **self.kernelParams)
        self.initializeCahnHilliardPdfsAccordingToPhi()

        self._nonPdfArrays = {
            'phiArr': self.phiArr,
            'muArr': self.muArr,
            'velArr': self.velArr,
            'forceArr': self.forceArr,
        }
        self._nonPdfGpuArrays = None
        self._pdfGpuArrays = None
        self.target = optimizationParams['target']

        self.hydroVelocitySetter = None

    def updateHydroPdfsAccordingToVelocity(self):
        if self.hydroVelocitySetter is None:
            self.hydroVelocitySetter = compileMacroscopicValuesSetter(self.lbSweepHydro.method,
                                                                      {'density': 1.0, 'velocity': self.velArr},
                                                                      pdfArr=self._pdfArrays[0][0], target='cpu')
        self.hydroVelocitySetter(pdfs=self._pdfArrays[0][0], F=self.forceArr, **self.kernelParams)

    def _arraysFromCpuToGpu(self):
        import pycuda.gpuarray as gpuarray
        if self._nonPdfGpuArrays is None:
            self._nonPdfGpuArrays = {name: gpuarray.to_gpu(arr) for name, arr in self._nonPdfArrays.items()}
            self._pdfGpuArrays = [[gpuarray.to_gpu(arr) for arr in self._pdfArrays[0]],
                                  [gpuarray.to_gpu(arr) for arr in self._pdfArrays[1]]]
        else:
            for name, arr in self._nonPdfArrays.items():
                self._nonPdfGpuArrays[name].set(arr)
            for i in range(2):
                for cpuArr, gpuArr in zip(self._pdfArrays[i], self._pdfGpuArrays[i]):
                    gpuArr.set(cpuArr)

    def _arraysFromGpuToCpu(self):
        for name, arr in self._nonPdfArrays.items():
            self._nonPdfGpuArrays[name].get(arr)
        for cpuArr, gpuArr in zip(self._pdfArrays[0], self._pdfGpuArrays[0]):
            gpuArr.get(cpuArr)

    def initializeCahnHilliardPdfsAccordingToPhi(self):
        for i in range(1, self.numPhases):
            self._pdfArrays[0][i].fill(0)
            self._pdfArrays[0][i][..., 0] = self.phiArr[..., i-1]

    def gaussianSmoothPhiFields(self, sigma):
        from scipy.ndimage.filters import gaussian_filter
        for i in range(self.phiArr.shape[-1]):
            gaussian_filter(self.phi[..., i], sigma, output=self.phi[..., i], mode='wrap')

    @property
    def phi(self):
        return removeGhostLayers(self.phiArr, indexDimensions=1)

    @property
    def mu(self):
        return removeGhostLayers(self.muArr, indexDimensions=1)

    @property
    def velocity(self):
        return removeGhostLayers(self.velArr, indexDimensions=1)

    def run(self, timeSteps=1):
        """Run the scenario for the given amount of time steps"""
        if self.target == 'gpu':
            self._arraysFromCpuToGpu()
            self._timeLoop(self._pdfGpuArrays, timeSteps=timeSteps, **self._nonPdfGpuArrays)
            self._arraysFromGpuToCpu()
        else:
            self._timeLoop(self._pdfArrays, timeSteps=timeSteps, **self._nonPdfArrays)
        self.timeStepsRun += timeSteps

    def _timeLoop(self, pdfArrays, phiArr, muArr, velArr, forceArr, timeSteps):
        for t in range(timeSteps):
            self._muPhiPeriodicityHandler(pdfs=phiArr)
            self.muSweep(phi=phiArr, mu=muArr)

            self._muPhiPeriodicityHandler(pdfs=muArr)
            self.forceSweep(mu=muArr, phi=phiArr, F=forceArr)

            for src in pdfArrays[0]:
                self._pdfPeriodicityHandler(pdfs=src)

            for sweep, src, dst in zip(self.lbSweeps, *pdfArrays):
                sweep(src=src, dst=dst, F=forceArr, phi=phiArr, vel=velArr, mu=muArr)

            pdfArrays[0], pdfArrays[1] = pdfArrays[1], pdfArrays[0]


# ------------------------------2D Angle measurement -----------------------------------------

from matplotlib.path import Path
from matplotlib._contour import QuadContourGenerator
import itertools
import scipy
import warnings


def getIsolines(dataset, level=0.5, refinementFactor=1):
    indexArrays = np.meshgrid(*[np.arange(s) for s in dataset.shape])
    gen = QuadContourGenerator(*indexArrays, dataset, None, True, 0)
    result = gen.create_contour(level)
    if refinementFactor > 1:
        result = [Path(p).interpolated(refinementFactor).vertices for p in result]
    return result


def findJumpIndices(array, threshold=0, minLength=3):
    jumps = []
    offset = 0
    while True:
        if array[0] < threshold:
            jump = np.argmax(array > threshold)
        else:
            jump = np.argmax(array < threshold)
        if jump == 0:
            return jumps
        if len(array) <= minLength + jump:
            return jumps
        jumps.append(offset + jump)
        offset += jump + minLength

        array = array[jump + minLength:]


def findBranchingPoint(pathVertices1, pathVertices2, maxDistance=0.1):
    tree = scipy.spatial.KDTree(pathVertices1)
    distances, indices = tree.query(pathVertices2, k=1, distance_upper_bound=maxDistance)
    distances[distances == np.inf] = -1
    jumpIndices = findJumpIndices(distances, 0, 3)
    return pathVertices2[jumpIndices]


def findAllBranchingPoints(phaseField1, phaseField2, maxDistance=0.1):
    result = []
    isoLines = [getIsolines(p, level=0.5, refinementFactor=4) for p in (phaseField1, phaseField2)]
    for path1, path2 in itertools.product(*isoLines):
        bbs = findBranchingPoint(path1, path2, maxDistance)
        result += list(bbs)
    return np.array(result)


def findIntersections(pathVertices1, pathVertices2):
    from numpy import where, dstack, diff, meshgrid

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # min, max and all for arrays
        amin = lambda x1, x2: where(x1 < x2, x1, x2)
        amax = lambda x1, x2: where(x1 > x2, x1, x2)
        aall = lambda abools: dstack(abools).all(axis=2)
        slope = lambda line: (lambda d: d[:, 1] / d[:, 0])(diff(line, axis=0))

        x11, x21 = meshgrid(pathVertices1[:-1, 0], pathVertices2[:-1, 0])
        x12, x22 = meshgrid(pathVertices1[1:, 0], pathVertices2[1:, 0])
        y11, y21 = meshgrid(pathVertices1[:-1, 1], pathVertices2[:-1, 1])
        y12, y22 = meshgrid(pathVertices1[1:, 1], pathVertices2[1:, 1])

        m1, m2 = meshgrid(slope(pathVertices1), slope(pathVertices2))
        m1inv, m2inv = 1 / m1, 1 / m2

        yi = (m1 * (x21 - x11 - m2inv * y21) + y11) / (1 - m1 * m2inv)
        xi = (yi - y21) * m2inv + x21

        xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12),
                  amin(x21, x22) < xi, xi <= amax(x21, x22))
        yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
                  amin(y21, y22) < yi, yi <= amax(y21, y22))

        return xi[aall(xconds)], yi[aall(yconds)]


def findAllIntersectionPoints(phaseField1, phaseField2):
    isoLines = [getIsolines(p, level=1.0/3, refinementFactor=4)
                for p in (phaseField1, phaseField2)]
    result = []
    for path1, path2 in itertools.product(*isoLines):
        xArr, yArr = findIntersections(path1, path2)
        if xArr is not None and yArr is not None:
            for x, y in zip(xArr, yArr):
                result.append(np.array([x,y]))
    return np.array(result)


if __name__ == '__main__':
    test = np.array([-1, -1, -2, 3, 2, 2, 5, 2, -2, 5, -3, -5, -2, 5, 5, 5, 5])
    findJumpIndices(test)
    exit(0)



    sc = PhasefieldScenario((3, 3), 4, relaxationRate=1.9, mobilityRelaxationRates=1.4242,
                            optimizationParams={'target': 'gpu'})
    sc.run(1)
    import matplotlib.pyplot as plt
