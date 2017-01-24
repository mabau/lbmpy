"""
Factory functions for standard LBM methods
"""
import sympy as sp
from copy import copy
from lbmpy.stencils import getStencil
from lbmpy.methods.momentbased import createSRT, createTRT, createOrthogonalMRT
import lbmpy.forcemodels as forceModels
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.updatekernels import createStreamPullKernel, createPdfArray
from pystencils.cpu.kernelcreation import createKernel as createCpuKernel, addOpenMP
from pystencils.gpucuda.kernelcreation import createCUDAKernel as createGpuKernel
from pystencils.cpu import makePythonFunction as makePythonCpuFunction
from pystencils.gpucuda import makePythonFunction as makePythonGpuFunction


def _getParams(params, optParams):
    defaultMethodDescription = {
        'target': 'cpu',
        'stencil': 'D2Q9',
        'method': 'srt',  # can be srt, trt or mrt
        'relaxationRates': sp.symbols("omega_:10"),
        'compressible': False,
        'order': 2,

        'forceModel': 'none',  # can be 'simple', 'luo' or 'guo'
        'force': (0, 0, 0),
    }

    defaultOptimizationDescription = {
        'doCseInOpposingDirections': False,
        'doOverallCse': False,
        'split': True,

        'fieldSize': None,
        'fieldLayout': 'reverseNumpy',  # can be 'numpy' (='c'), 'reverseNumpy' (='f'), 'fzyx', 'zyxf'

        'openMP': True,
    }
    unknownParams = [k for k in params.keys() if k not in defaultMethodDescription]
    unknownOptParams = [k for k in optParams.keys() if k not in defaultOptimizationDescription]
    if unknownParams:
        raise ValueError("Unknown parameter(s): " + ",".join(unknownParams))
    if unknownOptParams:
        raise ValueError("Unknown optimization parameter(s): " + ",".join(unknownOptParams))

    paramsResult = copy(defaultMethodDescription)
    paramsResult.update(params)
    optParamsResult = copy(defaultOptimizationDescription)
    optParamsResult.update(optParams)
    return paramsResult, optParamsResult


def createLatticeBoltzmannFunction(optimizationParams={}, **kwargs):
    params, optParams = _getParams(kwargs, optimizationParams)

    ast = createLatticeBoltzmannKernel(**params, optimizationParams=optParams)
    if params['target'] == 'cpu':
        if 'openMP' in optParams:
            if isinstance(optParams['openMP'], bool) and optParams['openMP']:
                addOpenMP(ast)
            elif isinstance(optParams['openMP'], int):
                addOpenMP(ast, numThreads=optParams['openMP'])
        return makePythonCpuFunction(ast)
    elif params['target'] == 'gpu':
        return makePythonGpuFunction(ast)
    else:
        return ValueError("'target' has to be either 'cpu' or 'gpu'")


def createLatticeBoltzmannKernel(optimizationParams={}, **kwargs):
    params, optParams = _getParams(kwargs, optimizationParams)

    updateRule = createLatticeBoltzmannUpdateRule(**params, optimizationParams=optimizationParams)

    if params['target'] == 'cpu':
        if 'splitGroups' in updateRule.simplificationHints:
            splitGroups = updateRule.simplificationHints['splitGroups']
        else:
            splitGroups = ()
        return createCpuKernel(updateRule.allEquations, splitGroups=splitGroups)
    elif params['target'] == 'gpu':
        return createGpuKernel(updateRule.allEquations)
    else:
        return ValueError("'target' has to be either 'cpu' or 'gpu'")


def createLatticeBoltzmannUpdateRule(optimizationParams={}, **kwargs):
    params, optParams = _getParams(kwargs, optimizationParams)
    stencil = getStencil(params['stencil'])
    method = createLatticeBoltzmannCollisionRule(**params)

    splitInnerLoop = 'split' in optParams and optParams['split']

    dirCSE = 'doCseInOpposingDirections'
    doCseInOpposingDirections = False if dirCSE not in optParams else optParams[dirCSE]
    doOverallCse = False if 'doOverallCse' not in optParams else optParams['doOverallCse']
    simplification = createSimplificationStrategy(method, doCseInOpposingDirections, doOverallCse, splitInnerLoop)
    collisionRule = simplification(method.getCollisionRule())

    if 'fieldSize' in optParams and optParams['fieldSize']:
        npField = createPdfArray(optParams['fieldSize'], len(stencil), layout=optParams['fieldLayout'])
        updateRule = createStreamPullKernel(collisionRule, numpyField=npField)
    else:
        layoutName = optParams['fieldLayout']
        if layoutName == 'fzyx' or 'zyxf':
            dim = len(stencil[0])
            layoutName = tuple(reversed(range(dim)))
        updateRule = createStreamPullKernel(collisionRule, genericLayout=layoutName)

    return updateRule


def createLatticeBoltzmannCollisionRule(**params):
    params, _ = _getParams(params, {})

    stencil = getStencil(params['stencil'])
    dim = len(stencil[0])

    if 'forceModel' in params:
        forceModelName = params['forceModel']
        if forceModelName.lower() == 'none':
            forceModel = None
        elif forceModelName.lower() == 'simple':
            forceModel = forceModels.Simple(params['force'][:dim])
        elif forceModelName.lower() == 'luo':
            forceModel = forceModels.Luo(params['force'][:dim])
        elif forceModelName.lower() == 'guo':
            forceModel = forceModels.Guo(params['force'][:dim])
        else:
            raise ValueError("Unknown force model %s" % (forceModelName,))
    else:
        forceModel = None

    commonParams = {
        'compressible': params['compressible'],
        'equilibriumAccuracyOrder': params['order'],
        'forceModel': forceModel,
    }
    methodName = params['method']
    relaxationRates = params['relaxationRates']

    if methodName.lower() == 'srt':
        assert len(relaxationRates) >= 1, "Not enough relaxation rates"
        method = createSRT(stencil, relaxationRates[0], **commonParams)
    elif methodName.lower() == 'trt':
        assert len(relaxationRates) >= 2, "Not enough relaxation rates"
        method = createTRT(stencil, relaxationRates[0], relaxationRates[1], **commonParams)
    elif methodName.lower() == 'mrt':
        nextRelaxationRate = 0

        def relaxationRateGetter(momentGroup):
            nonlocal nextRelaxationRate
            res = relaxationRates[nextRelaxationRate]
            nextRelaxationRate += 1
            return res
        method = createOrthogonalMRT(stencil, relaxationRateGetter, **commonParams)
    else:
        raise ValueError("Unknown method %s" % (methodName,))

    return method

