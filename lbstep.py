from lbmpy.boundary_handling import BoundaryHandling
from lbmpy.creationfunctions import switchToSymbolicRelaxationRatesForEntropicMethods, createLatticeBoltzmannFunction, \
    updateWithDefaultParameters
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.stencils import getStencil
from pystencils.datahandling import SerialDataHandling
from pystencils import createKernel


class LatticeBoltzmannStep:

    def __init__(self, domainSize=None, lbmKernel=None, periodicity=False,
                 kernelParams={}, dataHandling=None, dataPrefix="", optimizationParams={}, **methodParameters):

        # --- Parameter normalization  ---
        if dataHandling is not None:
            if domainSize is not None:
                raise ValueError("When passing a dataHandling, the domainSize parameter can not be specified")

        if dataHandling is None:
            if domainSize is None:
                raise ValueError("Specify either domainSize or dataHandling")
            dataHandling = SerialDataHandling(domainSize, defaultGhostLayers=1, periodicity=periodicity)

        methodParameters, optimizationParams = updateWithDefaultParameters(methodParameters, optimizationParams)
        target = optimizationParams['target']

        # --- Kernel creation ---
        if lbmKernel:
            Q = len(lbmKernel.method.stencil)
        else:
            Q = len(getStencil(methodParameters['stencil']))

        self._dataHandling = dataHandling
        self._pdfArrName = dataPrefix + "pdfSrc"
        self._tmpArrName = dataPrefix + "pdfTmp"
        self._velocityArrName = dataPrefix + "velocity"
        self._densityArrName = dataPrefix + "density"

        self._gpu = target == 'gpu'
        layout = optimizationParams['fieldLayout']
        self._dataHandling.addArray(self._pdfArrName, fSize=Q, gpu=self._gpu, layout=layout)
        self._dataHandling.addArray(self._tmpArrName, fSize=Q, gpu=self._gpu, cpu=not self._gpu, layout=layout)
        self._dataHandling.addArray(self._velocityArrName, fSize=self._dataHandling.dim, gpu=False, layout=layout)
        self._dataHandling.addArray(self._densityArrName, fSize=1, gpu=False, layout=layout)

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
                                                  name=dataPrefix + "_boundaryHandling",
                                                  target=target, openMP=optimizationParams['openMP'])

        # -- Macroscopic Value Kernels
        self._getterKernel, self._setterKernel = self._compilerMacroscopicSetterAndGetter()

        self.timeStepsRun = 0

        for b in self._dataHandling.iterate():
            b[self._densityArrName].fill(1.0)
            b[self._velocityArrName].fill(0.0)

    @property
    def boundaryHandling(self):
        """Boundary handling instance of the scenario. Use this to change the boundary setup"""
        return self._boundaryHandling

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

    def _compilerMacroscopicSetterAndGetter(self):
        lbMethod = self._lbmKernel.method
        D = lbMethod.dim
        Q = len(lbMethod.stencil)
        cqc = lbMethod.conservedQuantityComputation
        pdfField = self._dataHandling.fields[self._pdfArrName]
        rhoField = self._dataHandling.fields[self._densityArrName]
        velField = self._dataHandling.fields[self._velocityArrName]
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


if __name__ == '__main__':
    from pycuda import autoinit
    from lbmpy.boundaries import NoSlip, UBB
    from pystencils import makeSlice
    step = LatticeBoltzmannStep((30, 30), relaxationRate=1.8, periodicity=True,
                                optimizationParams={'target': 'cpu', 'openMP': False})

    wall = NoSlip()
    movingWall = UBB((0.001, 0))

    bh = step.boundaryHandling
    bh.setBoundary(wall, makeSlice[0, :])
    bh.setBoundary(wall, makeSlice[-1, :])
    bh.setBoundary(wall, makeSlice[:, 0])
    bh.setBoundary(movingWall, makeSlice[:, -1])
    bh.prepare()

    step.run(5000)
