import numpy as np
from lbmpy.boundaries.boundary_handling import BoundaryHandling
from lbmpy.creationfunctions import switchToSymbolicRelaxationRatesForEntropicMethods, createLatticeBoltzmannFunction, \
    updateWithDefaultParameters
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.stencils import getStencil
from pystencils.datahandling.serial_datahandling import SerialDataHandling
from pystencils import createKernel, makeSlice
from pystencils.slicing import SlicedGetter
from pystencils.timeloop import TimeLoop


class LatticeBoltzmannStep:

    vtkScenarioNrCounter = 0

    def __init__(self, domainSize=None, lbmKernel=None, periodicity=False,
                 kernelParams={}, dataHandling=None, name="lbm", optimizationParams={},
                 velocityDataName=None, densityDataName=None,
                 computeVelocityInEveryStep=False, computeDensityInEveryStep=False,
                 velocityInputArrayName=None,
                 **methodParameters):

        # --- Parameter normalization  ---
        if dataHandling is not None:
            if domainSize is not None:
                raise ValueError("When passing a dataHandling, the domainSize parameter can not be specified")

        if dataHandling is None:
            if domainSize is None:
                raise ValueError("Specify either domainSize or dataHandling")
            dataHandling = SerialDataHandling(domainSize, defaultGhostLayers=1, periodicity=periodicity)

        if 'stencil' not in methodParameters:
            methodParameters['stencil'] = 'D2Q9' if dataHandling.dim == 2 else 'D3Q27'

        methodParameters, optimizationParams = updateWithDefaultParameters(methodParameters, optimizationParams)

        if lbmKernel:
            Q = len(lbmKernel.method.stencil)
        else:
            Q = len(getStencil(methodParameters['stencil']))
        target = optimizationParams['target']

        self.name = name
        self._dataHandling = dataHandling
        self._pdfArrName = name + "_pdfSrc"
        self._tmpArrName = name + "_pdfTmp"
        self.velocityDataName = name + "_velocity" if velocityDataName is None else velocityDataName
        self.densityDataName = name + "_density" if densityDataName is None else densityDataName

        self._gpu = target == 'gpu'
        layout = optimizationParams['fieldLayout']
        self._dataHandling.addArray(self._pdfArrName, fSize=Q, gpu=self._gpu, layout=layout)
        self._dataHandling.addArray(self._tmpArrName, fSize=Q, gpu=self._gpu, cpu=not self._gpu, layout=layout)

        if velocityDataName is None:
            self._dataHandling.addArray(self.velocityDataName, fSize=self._dataHandling.dim,
                                        gpu=self._gpu and computeVelocityInEveryStep, layout=layout)
        if densityDataName is None:
            self._dataHandling.addArray(self.densityDataName, fSize=1,
                                        gpu=self._gpu and computeDensityInEveryStep, layout=layout)

        if computeVelocityInEveryStep:
            methodParameters['output']['velocity'] = self._dataHandling.fields[self.velocityDataName]
        if computeDensityInEveryStep:
            methodParameters['output']['density'] = self._dataHandling.fields[self.densityDataName]
        if velocityInputArrayName is not None:
            methodParameters['velocityInput'] = self._dataHandling.fields[velocityInputArrayName]

        self.kernelParams = kernelParams

        # --- Kernel creation ---
        if lbmKernel is None:
            switchToSymbolicRelaxationRatesForEntropicMethods(methodParameters, self.kernelParams)
            optimizationParams['symbolicField'] = dataHandling.fields[self._pdfArrName]
            methodParameters['fieldName'] = self._pdfArrName
            methodParameters['secondFieldName'] = self._tmpArrName
            self._lbmKernel = createLatticeBoltzmannFunction(optimizationParams=optimizationParams, **methodParameters)
        else:
            assert self._dataHandling.dim == self._lbmKernel.method.dim, \
                "Error: %dD Kernel for %D domain" % (self._lbmKernel.method.dim, self._dataHandling.dim)
            self._lbmKernel = lbmKernel

        # -- Boundary Handling  & Synchronization ---
        self._sync = dataHandling.synchronizationFunction([self._pdfArrName], methodParameters['stencil'], target)
        self._boundaryHandling = BoundaryHandling(self._lbmKernel.method, self._dataHandling, self._pdfArrName,
                                                  name=name + "_boundaryHandling",
                                                  target=target, openMP=optimizationParams['openMP'])

        # -- Macroscopic Value Kernels
        self._getterKernel, self._setterKernel = self._compilerMacroscopicSetterAndGetter()

        self.timeStepsRun = 0

        for b in self._dataHandling.iterate():
            b[self.densityDataName].fill(1.0)
            b[self.velocityDataName].fill(0.0)
        self.setPdfFieldsFromMacroscopicValues()

        # -- VTK output
        self.vtkWriter = self.dataHandling.vtkWriter(name + str(LatticeBoltzmannStep.vtkScenarioNrCounter),
                                                     [self.velocityDataName, self.densityDataName])
        LatticeBoltzmannStep.vtkScenarioNrCounter += 1

    @property
    def boundaryHandling(self):
        """Boundary handling instance of the scenario. Use this to change the boundary setup"""
        return self._boundaryHandling

    @property
    def dataHandling(self):
        return self._dataHandling

    @property
    def dim(self):
        return self._dataHandling.dim

    @property
    def method(self):
        return self._lbmKernel.method

    @property
    def domainSize(self):
        return self._dataHandling.shape

    @property
    def numberOfCells(self):
        result = 1
        for d in self.domainSize:
            result *= d
        return result

    @property
    def ast(self):
        return self._lbmKernel.ast

    @property
    def pdfArrayName(self):
        return self._pdfArrName

    def _getSlice(self, dataName, sliceObj, masked):
        if sliceObj is None:
            sliceObj = makeSlice[:, :] if self.dim == 2 else makeSlice[:, :, 0.5]

        indexSlice = None
        if len(sliceObj) > self.dim:
            indexSlice = sliceObj[self.dim:]
            sliceObj = sliceObj[:self.dim]
            assert len(indexSlice) == 1

        result = self._dataHandling.gatherArray(dataName, sliceObj)
        if result is None:
            return

        if masked:
            mask = self.boundaryHandling.getMask(sliceObj, 'fluid', True)
            if len(mask.shape) < len(result.shape):
                assert len(mask.shape) + 1 == len(result.shape)
                mask = np.repeat(mask[..., np.newaxis], result.shape[-1], axis=2)

            result = np.ma.masked_array(result, mask)
        if indexSlice:
            result = result[..., indexSlice[-1]]
        return result.squeeze()

    def velocitySlice(self, sliceObj=None, masked=True):
        return self._getSlice(self.velocityDataName, sliceObj, masked)

    def densitySlice(self, sliceObj=None, masked=True):
        return self._getSlice(self.densityDataName, sliceObj, masked)

    @property
    def velocity(self):
        return SlicedGetter(self.velocitySlice)

    @property
    def density(self):
        return SlicedGetter(self.densitySlice)

    def preRun(self):
        if self._gpu:
            self._dataHandling.toGpu(self._pdfArrName)
            if self._dataHandling.isOnGpu(self.velocityDataName):
                self._dataHandling.toGpu(self.velocityDataName)
            if self._dataHandling.isOnGpu(self.densityDataName):
                self._dataHandling.toGpu(self.densityDataName)

    def setPdfFieldsFromMacroscopicValues(self):
        self._dataHandling.runKernel(self._setterKernel, **self.kernelParams)

    def timeStep(self):
        self._sync()
        self._boundaryHandling(**self.kernelParams)
        self._dataHandling.runKernel(self._lbmKernel, **self.kernelParams)
        self._dataHandling.swap(self._pdfArrName, self._tmpArrName, self._gpu)
        self.timeStepsRun += 1

    def postRun(self):
        if self._gpu:
            self._dataHandling.toCpu(self._pdfArrName)
        self._dataHandling.runKernel(self._getterKernel, **self.kernelParams)

    def run(self, timeSteps):
        self.preRun()
        for i in range(timeSteps):
            self.timeStep()
        self.postRun()

    def benchmarkRun(self, timeSteps):
        timeLoop = TimeLoop()
        timeLoop.addStep(self)
        durationOfTimeStep = timeLoop.benchmarkRun(timeSteps)
        mlups = self.numberOfCells / durationOfTimeStep * 1e-6
        return mlups

    def benchmark(self, timeForBenchmark=5, initTimeSteps=10, numberOfTimeStepsForEstimation=20):
        timeLoop = TimeLoop()
        timeLoop.addStep(self)
        durationOfTimeStep = timeLoop.benchmark(timeForBenchmark, initTimeSteps, numberOfTimeStepsForEstimation)
        mlups = self.numberOfCells / durationOfTimeStep * 1e-6
        return mlups

    def writeVTK(self):
        self.vtkWriter(self.timeStepsRun)

    def _compilerMacroscopicSetterAndGetter(self):
        lbMethod = self._lbmKernel.method
        D = lbMethod.dim
        Q = len(lbMethod.stencil)
        cqc = lbMethod.conservedQuantityComputation
        pdfField = self._dataHandling.fields[self._pdfArrName]
        rhoField = self._dataHandling.fields[self.densityDataName]
        velField = self._dataHandling.fields[self.velocityDataName]
        pdfSymbols = [pdfField(i) for i in range(Q)]

        getterEqs = cqc.outputEquationsFromPdfs(pdfSymbols, {'density': rhoField, 'velocity': velField})
        getterKernel = createKernel(getterEqs, target='cpu').compile()

        inpEqs = cqc.equilibriumInputEquationsFromInitValues(rhoField.center, [velField(i) for i in range(D)])
        setterEqs = lbMethod.getEquilibrium(conservedQuantityEquations=inpEqs)
        setterEqs = setterEqs.copyWithSubstitutionsApplied({sym: pdfField(i)
                                                            for i, sym in enumerate(lbMethod.postCollisionPdfSymbols)})

        setterEqs = createSimplificationStrategy(lbMethod)(setterEqs)
        setterKernel = createKernel(setterEqs, target='cpu').compile()
        return getterKernel, setterKernel
