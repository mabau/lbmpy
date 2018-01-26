import numpy as np
from lbmpy.boundaries.boundary_handling import BoundaryHandling
from lbmpy.creationfunctions import switchToSymbolicRelaxationRatesForEntropicMethods, createLatticeBoltzmannFunction, \
    updateWithDefaultParameters
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.stencils import getStencil
from pystencils.datahandling.serial_datahandling import SerialDataHandling
from pystencils import createKernel, makeSlice


class LatticeBoltzmannStep:

    def __init__(self, domainSize=None, lbmKernel=None, periodicity=False,
                 kernelParams={}, dataHandling=None, name="lbm", optimizationParams={}, **methodParameters):

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
        target = optimizationParams['target']

        # --- Kernel creation ---
        if lbmKernel:
            Q = len(lbmKernel.method.stencil)
        else:
            Q = len(getStencil(methodParameters['stencil']))

        self._dataHandling = dataHandling
        self._pdfArrName = name + "_pdfSrc"
        self._tmpArrName = name + "_pdfTmp"
        self.velocityDataName = name + "_velocity"
        self.densityDataName = name + "_density"

        self._gpu = target == 'gpu'
        layout = optimizationParams['fieldLayout']
        self._dataHandling.addArray(self._pdfArrName, fSize=Q, gpu=self._gpu, layout=layout)
        self._dataHandling.addArray(self._tmpArrName, fSize=Q, gpu=self._gpu, cpu=not self._gpu, layout=layout)
        self._dataHandling.addArray(self.velocityDataName, fSize=self._dataHandling.dim, gpu=False, layout=layout)
        self._dataHandling.addArray(self.densityDataName, fSize=1, gpu=False, layout=layout)

        self._kernelParams = kernelParams

        if lbmKernel is None:
            switchToSymbolicRelaxationRatesForEntropicMethods(methodParameters, self._kernelParams)
            optimizationParams['symbolicField'] = dataHandling.fields[self._pdfArrName]
            methodParameters['fieldName'] = self._pdfArrName
            methodParameters['secondFieldName'] = self._tmpArrName
            self._lbmKernel = createLatticeBoltzmannFunction(optimizationParams=optimizationParams, **methodParameters)
        else:
            assert self._dataHandling.dim == self._lbmKernel.method.dim, \
                "Error: %dD Kernel for %D domain" % (self._lbmKernel.method.dim, self._dataHandling.dim)
            self._lbmKernel = lbmKernel

        # -- Boundary Handling  & Synchronization ---
        if self._gpu:
            self._sync = dataHandling.synchronizationFunctionGPU([self._pdfArrName], methodParameters['stencil'])
        else:
            self._sync = dataHandling.synchronizationFunctionCPU([self._pdfArrName], methodParameters['stencil'])
        self._boundaryHandling = BoundaryHandling(self._lbmKernel.method, self._dataHandling, self._pdfArrName,
                                                  name=name + "_boundaryHandling",
                                                  target=target, openMP=optimizationParams['openMP'])

        # -- Macroscopic Value Kernels
        self._getterKernel, self._setterKernel = self._compilerMacroscopicSetterAndGetter()

        self.timeStepsRun = 0

        for b in self._dataHandling.iterate():
            b[self.densityDataName].fill(1.0)
            b[self.velocityDataName].fill(0.0)

        # -- VTK output
        self.vtkWriter = self.dataHandling.vtkWriter(name, [self.velocityDataName, self.densityDataName])

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
    def pdfArrayName(self):
        return self._pdfArrName

    def _getSlice(self, dataName, sliceObj):
        if sliceObj is None:
            sliceObj = makeSlice[:, :] if self.dim == 2 else makeSlice[:, :, 0.5]
        for arr in self._dataHandling.gatherArray(dataName, sliceObj):
            return np.squeeze(arr)
        return None

    def velocitySlice(self, sliceObj=None):
        return self._getSlice(self.velocityDataName, sliceObj)

    def densitySlice(self, sliceObj=None):
        return self._getSlice(self.densityDataName, sliceObj)

    def preRun(self):
        self._dataHandling.runKernel(self._setterKernel)
        if self._gpu:
            self._dataHandling.toGpu(self._pdfArrName)

    def timeStep(self):
        self._sync()
        self._boundaryHandling()
        self._dataHandling.runKernel(self._lbmKernel, **self._kernelParams)
        self._dataHandling.swap(self._pdfArrName, self._tmpArrName, self._gpu)
        self.timeStepsRun += 1

    def postRun(self):
        if self._gpu:
            self._dataHandling.toCpu(self._pdfArrName)
        self._dataHandling.runKernel(self._getterKernel)

    def run(self, timeSteps):
        self.preRun()
        for i in range(timeSteps):
            self.timeStep()
        self.postRun()

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
