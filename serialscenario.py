from functools import partial
import numpy as np
from pystencils import Field
from pystencils.slicing import sliceFromDirection
from lbmpy.creationfunctions import createLatticeBoltzmannFunction
from lbmpy.macroscopicValueKernels import compileMacroscopicValuesGetter, compileMacroscopicValuesSetter
from lbmpy.boundaries import BoundaryHandling, noSlip, ubb


def runScenario(domainSize, boundarySetupFunction, methodParameters, optimizationParameters, preUpdateFunctions=[]):
    ghostLayers = 1
    domainSizeWithGhostLayer = tuple([s + 2 * ghostLayers for s in domainSize])
    D = len(domainSize)

    if 'stencil' not in methodParameters:
        methodParameters['stencil'] = 'D2Q9' if D == 2 else 'D3Q27'

    # Create kernel
    lbmKernel = createLatticeBoltzmannFunction(**methodParameters, optimizationParams=optimizationParameters)
    method = lbmKernel.method
    assert D == method.dim, "Domain size and stencil do not match"
    Q = len(method.stencil)

    # Create pdf fields
    pdfSrc = np.zeros(domainSizeWithGhostLayer + (Q,))
    pdfDst = np.zeros_like(pdfSrc)

    # Boundary setup
    if boundarySetupFunction is not None:
        symPdfField = Field.createFromNumpyArray('pdfs', pdfSrc, indexDimensions=1)
        boundaryHandling = BoundaryHandling(symPdfField, domainSize, lbmKernel.method)
        boundarySetupFunction(boundaryHandling=boundaryHandling, method=method)
        boundaryHandling.prepare()
    else:
        boundaryHandling = None

    # Macroscopic value input/output
    densityArr = np.zeros(domainSizeWithGhostLayer)
    velocityArr = np.zeros(domainSizeWithGhostLayer + (D,))
    getMacroscopic = compileMacroscopicValuesGetter(method, ['density', 'velocity'], pdfArr=pdfSrc)
    setMacroscopic = compileMacroscopicValuesSetter(method, {'density': 1.0, 'velocity': [0]*D}, pdfArr=pdfSrc)
    setMacroscopic(pdfs=pdfSrc)

    # Run simulation
    def timeLoop(timeSteps):
        nonlocal pdfSrc, pdfDst, densityArr, velocityArr
        for t in range(timeSteps):
            for f in preUpdateFunctions:
                f(pdfSrc)
            if boundaryHandling is not None:
                boundaryHandling(pdfs=pdfSrc)
            lbmKernel(src=pdfSrc, dst=pdfDst)
            pdfSrc, pdfDst = pdfDst, pdfSrc
        getMacroscopic(pdfs=pdfSrc, density=densityArr, velocity=velocityArr)
        return pdfSrc, densityArr, velocityArr

    return timeLoop


def runLidDrivenCavity(domainSize, lidVelocity=0.005, optimizationParameters={}, **kwargs):
    def boundarySetupFunction(boundaryHandling, method):
        myUbb = partial(ubb, velocity=[lidVelocity, 0, 0][:method.dim])
        myUbb.name = 'ubb'
        boundaryHandling.setBoundary(myUbb, sliceFromDirection('N', method.dim))
        for direction in ('W', 'E', 'S') if method.dim == 2 else ('W', 'E', 'S', 'T', 'B'):
            boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))
    return runScenario(domainSize, boundarySetupFunction, kwargs, optimizationParameters)


def runForceDrivenChannel(dim, force, domainSize=None, radius=None, length=None, optimizationParameters={}, **kwargs):
    assert dim in (2, 3)
    kwargs['force'] = tuple([force, 0, 0][:dim])

    if radius is not None:
        assert length is not None
        if dim == 3:
            domainSize = (length, 2*radius+1, 2*radius+1)
            roundChannel = True
        else:
            if domainSize is None:
                domainSize = (length, 2*radius)
    else:
        roundChannel = False

    def boundarySetupFunction(boundaryHandling, method):
        if method.dim == 2:
            for direction in ('N', 'S'):
                boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))
        elif method.dim == 3:
            if roundChannel:
                noSlipIdx = boundaryHandling.addBoundary(noSlip)
                ff = boundaryHandling.flagField
                yMid = ff.shape[1] // 2
                zMid = ff.shape[2] // 2
                y, z = np.meshgrid(range(ff.shape[1]), range(ff.shape[2]))
                ff[(y - yMid) ** 2 + (z - zMid) ** 2 > radius ** 2] = noSlipIdx
            else:
                for direction in ('N', 'S', 'T', 'B'):
                    boundaryHandling.setBoundary(noSlip, sliceFromDirection(direction, method.dim))

    def periodicity(pdfArr):
        pdfArr[0, :, :] = pdfArr[-2, :, :]
        pdfArr[-1, :, :] = pdfArr[1, :, :]

    assert domainSize is not None
    if 'forceModel' not in kwargs:
        kwargs['forceModel'] = 'guo'

    return runScenario(domainSize, boundarySetupFunction, kwargs, optimizationParameters,
                       preUpdateFunctions=[periodicity])


