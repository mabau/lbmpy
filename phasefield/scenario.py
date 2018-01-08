import numpy as np

from lbmpy.boundaries.periodicityhandling import PeriodicityHandling
from lbmpy.creationfunctions import updateWithDefaultParameters, createLatticeBoltzmannFunction
from lbmpy.macroscopic_value_kernels import compileMacroscopicValuesSetter
from lbmpy.phasefield.cahn_hilliard_lbm import createCahnHilliardLbFunction
from lbmpy.stencils import getStencil
from lbmpy.updatekernels import createPdfArray
from pystencils.field import Field, getLayoutOfArray
from pystencils.slicing import addGhostLayers, removeGhostLayers
from lbmpy.phasefield.analytical import symbolicOrderParameters, freeEnergyFunctionalNPhases, \
    createChemicalPotentialEvolutionEquations, createForceUpdateEquations


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
        freeEnergy = freeEnergyFunctionalNPhases(numPhases, surfaceTensionCallback, interfaceWidth, orderParameters)

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

        useFdForCahnHilliard = False
        if useFdForCahnHilliard:
            dt = 0.01
            mobility = 1
            from lbmpy.phasefield.analytical import cahnHilliardFdKernel
            self.sweepsCH = [cahnHilliardFdKernel(i, phiField, muField, velField, mobility,
                                                  dx, dt, optimizationParams['target'])
                             for i in range(numPhases-1)]
        else:
            self.sweepsCH = [createCahnHilliardLbFunction(stencil, mobilityRelaxationRates[i],
                                                          velField, muField(i), phiField(i), optimizationParams, gamma)
                             for i in range(numPhases-1)]

        self.lbSweeps = [self.lbSweepHydro] + self.sweepsCH

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
