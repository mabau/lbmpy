import sympy as sp
import waLBerla as wlb
from lbmpy.creationfunctions import createLatticeBoltzmannFunction, updateWithDefaultParameters, \
    switchToSymbolicRelaxationRatesForEntropicMethods
from lbmpy.parallel.boundaryhandling import BoundaryHandling


class Scenario(object):

    def __init__(self, blocks, methodParameters, optimizationParams, lbmKernel=None,
                 initialVelocityCallback=None, preUpdateFunctions=[], kernelParams={},
                 blockDataPrefix='', directCommunication=False):

        methodParameters, optimizationParams = updateWithDefaultParameters(methodParameters, optimizationParams)
        target = optimizationParams['target']

        # ----- Create LBM kernel
        if lbmKernel is None:
            switchToSymbolicRelaxationRatesForEntropicMethods(methodParameters, kernelParams)
            methodParameters['optimizationParams'] = optimizationParams
            self._lbmKernel = createLatticeBoltzmannFunction(**methodParameters)
        else:
            self._lbmKernel = lbmKernel

        self._blockDataPrefix = blockDataPrefix

        # ----- Add fields
        numPdfs = len(self._lbmKernel.method.stencils)
        wlb.field.addToStorage(blocks, blockDataPrefix + 'pdfs', float, fSize=numPdfs, ghostLayers=1)
        wlb.field.addFlagFieldToStorage(blocks, blockDataPrefix + "flags", nrOfBits=16, ghostLayers=1)

        if target == 'gpu':
            wlb.cuda.addGpuFieldToStorage(blocks, blockDataPrefix + "gpuPdfs", float, fSize=numPdfs,
                                          usePitchedMem=False)

        # ----- Create communication scheme
        communicationStencil = methodParameters['stencil']

        communicationFunctions = {
            ('cpu', True): (wlb.createUniformDirectScheme, wlb.field.createMPIDatatypeInfo),
            ('cpu', False): (wlb.createUniformBufferedScheme, wlb.field.createPackInfo),
            ('gpu', True): (wlb.createUniformDirectScheme, wlb.cuda.createMPIDatatypeInfo),
            ('gpu', False): (wlb.createUniformBufferedScheme, wlb.cuda.createPackInfo),
        }
        createScheme, createCommunicationData = communicationFunctions[(target, directCommunication)]
        self._communicationScheme = createScheme(blocks, communicationStencil)
        self._communicationScheme.addDataToCommunicate(createCommunicationData(blocks, blockDataPrefix + 'pdfs'))

        # ----- Create boundary handling

        self._boundaryHandling = BoundaryHandling(blocks, self._lbmKernel.method, blockDataPrefix + 'pdfs',
                                                  blockDataPrefix + "flags", target=target)

if __name__ == '__main__':
    class A(object):
        def __call__(self, call_from):
            print("foo from A, call from %s" % call_from)


    class B(object):
        def __call__(self, call_from):
            print( "foo from B, call from %s" % call_from)


    class C(object):
        def __call__(self, call_from):
            print( "foo from C, call from %s" % call_from )


    class D(A, B, C):
        def foo(self):
            for cls in D.__bases__:
                cls.__call__(self, "D")


    d = D()
    d.foo()