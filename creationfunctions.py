"""
Factory functions for standard LBM methods
"""
import sympy as sp
from copy import copy

from lbmpy.methods.creationfunctions import createKBCTypeTRT, createRawMRT, createThreeRelaxationRateMRT
from lbmpy.methods.entropic import addIterativeEntropyCondition, addEntropyCondition
from lbmpy.stencils import getStencil
from lbmpy.methods import createSRT, createTRT, createOrthogonalMRT
import lbmpy.forcemodels as forceModels
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.updatekernels import createStreamPullKernel, createPdfArray


def _getParams(params, optParams):
    defaultMethodDescription = {
        'stencil': 'D2Q9',
        'method': 'srt',  # can be srt, trt or mrt
        'relaxationRates': sp.symbols("omega_:10"),
        'compressible': False,
        'equilibriumAccuracyOrder': 2,

        'entropic': False,
        'entropicNewtonIterations': None,
        'omegaOutputField': None,

        'useContinuousMaxwellianEquilibrium': False,
        'cumulant': False,
        'forceModel': 'none',  # can be 'simple', 'luo' or 'guo'
        'force': (0, 0, 0),
        'initialVelocity': None,
    }

    defaultOptimizationDescription = {
        'doCseInOpposingDirections': False,
        'doOverallCse': False,
        'split': False,

        'fieldSize': None,
        'fieldLayout': 'reverseNumpy',  # can be 'numpy' (='c'), 'reverseNumpy' (='f'), 'fzyx', 'zyxf'

        'target': 'cpu',
        'openMP': True,
        'pdfArr': None,
    }
    unknownParams = [k for k in params.keys() if k not in defaultMethodDescription]
    unknownOptParams = [k for k in optParams.keys() if k not in defaultOptimizationDescription]
    if unknownParams:
        raise ValueError("Unknown parameter(s): " + ", ".join(unknownParams))
    if unknownOptParams:
        raise ValueError("Unknown optimization parameter(s): " + ",".join(unknownOptParams))

    paramsResult = copy(defaultMethodDescription)
    paramsResult.update(params)
    optParamsResult = copy(defaultOptimizationDescription)
    optParamsResult.update(optParams)
    return paramsResult, optParamsResult


def createLatticeBoltzmannFunction(ast=None, optimizationParams={}, **kwargs):
    params, optParams = _getParams(kwargs, optimizationParams)

    if ast is None:
        params['optimizationParams'] = optParams
        ast = createLatticeBoltzmannAst(**params)

    if optParams['target'] == 'cpu':
        from pystencils.cpu import makePythonFunction as makePythonCpuFunction, addOpenMP
        if 'openMP' in optParams:
            if isinstance(optParams['openMP'], bool) and optParams['openMP']:
                addOpenMP(ast)
            elif not isinstance(optParams['openMP'], bool) and isinstance(optParams['openMP'], int):
                addOpenMP(ast, numThreads=optParams['openMP'])
        res = makePythonCpuFunction(ast)
    elif optParams['target'] == 'gpu':
        from pystencils.gpucuda import makePythonFunction as makePythonGpuFunction
        res = makePythonGpuFunction(ast)
    else:
        return ValueError("'target' has to be either 'cpu' or 'gpu'")

    res.method = ast.method
    res.updateRule = ast.updateRule
    res.ast = ast
    return res


def createLatticeBoltzmannAst(updateRule=None, optimizationParams={}, **kwargs):
    params, optParams = _getParams(kwargs, optimizationParams)

    if updateRule is None:
        params['optimizationParams'] = optimizationParams
        updateRule = createLatticeBoltzmannUpdateRule(**params)

    if optParams['target'] == 'cpu':
        from pystencils.cpu import createKernel
        if 'splitGroups' in updateRule.simplificationHints:
            print("splitting!")
            splitGroups = updateRule.simplificationHints['splitGroups']
        else:
            splitGroups = ()
        res = createKernel(updateRule.allEquations, splitGroups=splitGroups)
    elif optParams['target'] == 'gpu':
        from pystencils.gpucuda import createCUDAKernel
        res = createCUDAKernel(updateRule.allEquations)
    else:
        return ValueError("'target' has to be either 'cpu' or 'gpu'")

    res.method = updateRule.method
    res.updateRule = updateRule
    return res


def createLatticeBoltzmannUpdateRule(lbMethod=None, optimizationParams={}, **kwargs):
    params, optParams = _getParams(kwargs, optimizationParams)
    stencil = getStencil(params['stencil'])

    if lbMethod is None:
        lbMethod = createLatticeBoltzmannMethod(**params)

    splitInnerLoop = 'split' in optParams and optParams['split']

    dirCSE = 'doCseInOpposingDirections'
    doCseInOpposingDirections = False if dirCSE not in optParams else optParams[dirCSE]
    doOverallCse = False if 'doOverallCse' not in optParams else optParams['doOverallCse']
    simplification = createSimplificationStrategy(lbMethod, doCseInOpposingDirections, doOverallCse, splitInnerLoop)
    collisionRule = simplification(lbMethod.getCollisionRule())

    if params['entropic']:
        if params['entropicNewtonIterations']:
            if isinstance(params['entropicNewtonIterations'], bool):
                iterations = 3
            else:
                iterations = params['entropicNewtonIterations']
            collisionRule = addIterativeEntropyCondition(collisionRule, newtonIterations=iterations,
                                                         omegaOutputField=params['omegaOutputField'])
        else:
            collisionRule = addEntropyCondition(collisionRule, omegaOutputField=params['omegaOutputField'])

    if 'fieldSize' in optParams and optParams['fieldSize']:
        npField = createPdfArray(optParams['fieldSize'], len(stencil), layout=optParams['fieldLayout'])
        updateRule = createStreamPullKernel(collisionRule, numpyField=npField)
    else:
        if 'pdfArr' in optParams:
            updateRule = createStreamPullKernel(collisionRule, numpyField=optParams['pdfArr'])
        else:
            layoutName = optParams['fieldLayout']
            if layoutName == 'fzyx' or 'zyxf':
                dim = len(stencil[0])
                layoutName = tuple(reversed(range(dim)))
            updateRule = createStreamPullKernel(collisionRule, genericLayout=layoutName)

    return updateRule


def createLatticeBoltzmannMethod(**params):
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
        'equilibriumAccuracyOrder': params['equilibriumAccuracyOrder'],
        'forceModel': forceModel,
        'useContinuousMaxwellianEquilibrium': params['useContinuousMaxwellianEquilibrium'],
        'cumulant': params['cumulant'],
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
        nextRelaxationRate = [0]

        def relaxationRateGetter(momentGroup):
            res = relaxationRates[nextRelaxationRate[0]]
            nextRelaxationRate[0] += 1
            return res
        method = createOrthogonalMRT(stencil, relaxationRateGetter, **commonParams)
    elif methodName.lower() == 'mrt_raw':
        method = createRawMRT(stencil, relaxationRates, **commonParams)
    elif methodName.lower() == 'mrt3':
        method = createThreeRelaxationRateMRT(stencil, relaxationRates, **commonParams)
    elif methodName.lower().startswith('trt-kbc-n'):
        if params['stencil'] == 'D2Q9':
            dim = 2
        elif params['stencil'] == 'D3Q27':
            dim = 3
        else:
            raise NotImplementedError("KBC type TRT methods can only be constructed for D2Q9 and D3Q27 stencils")
        methodNr = methodName[-1]
        method = createKBCTypeTRT(dim, relaxationRates[0], relaxationRates[1], 'KBC-N' + methodNr, **commonParams)
    else:
        raise ValueError("Unknown method %s" % (methodName,))

    return method

