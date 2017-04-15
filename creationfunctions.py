r"""
Creating LBM kernels
====================

Parameters
----------

The following list describes common parameters for the creation functions. They have to be passed as keyword parameters.


Method parameters
^^^^^^^^^^^^^^^^^

General:

- ``stencil='D2Q9'``: stencil name e.g. 'D2Q9', 'D3Q19'. See :func:`pystencils.stencils.getStencil` for details
- ``method='srt'``: name of lattice Boltzmann method. This determines the selection and relaxation pattern of
  moments/cumulants, i.e. which moment/cumulant basis is chosen, and which of the basis vectors are relaxed together
    - ``srt``: single relaxation time (:func:`lbmpy.methods.createSRT`)
    - ``trt``: two relaxation time, first relaxation rate is for even moments and determines the viscosity (as in SRT),
      the second relaxation rate is used for relaxing odd moments, and controls the bulk viscosity.
      (:func:`lbmpy.methods.createTRT`)
    - ``mrt``: orthogonal multi relaxation time model, number of relaxation rates depends on the stencil
      (:func:`lbmpy.methods.createOrthogonalMRT`)
    - ``mrt3``: three relaxation time method, where shear moments are relaxed with first relaxation rate (and therefore
      determine viscosity, second rate relaxes the shear tensor trace (determines bulk viscosity) and last rate relaxes
      all other, higher order moments. If two relaxation rates are chosen the same this is equivalent to a KBC type
      relaxation (:func:`lbmpy.methods.createThreeRelaxationRateMRT`)
    - ``mrt_raw``: non-orthogonal MRT where all relaxation rates can be specified independently i.e. there are as many
      relaxation rates as stencil entries. Look at the generated method in Jupyter to see which moment<->relaxation rate
      mapping (:func:`lbmpy.methods.createRawMRT`)
    - ``trt-kbc-n<N>`` where <N> is 1,2,3 or 4. Special two-relaxation method. This is not the entropic method
      yet, only the relaxation pattern. To get the entropic method, see parameters below!
      (:func:`lbmpy.methods.createKBCTypeTRT`)
- ``relaxationRates``: sequence of relaxation rates, number depends on selected method. If you specify more rates than
  method needs, the additional rates are ignored. For SRT and TRT models it is possible ot define a single
  ``relaxationRate`` instead of a list, the second rate for TRT is then determined via magic number.
- ``compressible=False``: affects the selection of equilibrium moments. Both options approximate the *incompressible*
  Navier Stokes Equations. However when chosen as False, the approximation is better, the standard LBM derivation is
  compressible.
- ``equilibriumAccuracyOrder=2``: order in velocity, at which the equilibrium moment/cumulant approximation is
  truncated. Order 2 is sufficient to approximate Navier-Stokes
- ``forceModel=None``: possible values: ``None``, ``'simple'``, ``'luo'``, ``'guo'``. For details see
  :mod:`lbmpy.forcemodels`
- ``force=(0,0,0)``: either constant force or a symbolic expression depending on field value
- ``useContinuousMaxwellianEquilibrium=True``: way to compute equilibrium moments/cumulants, if False the standard
  discretized LBM equilibrium is used, otherwise the equilibrium moments are computed from the continuous Maxwellian.
  This makes only a difference if sparse stencils are used e.g. D2Q9 and D3Q27 are not affected, D319 and DQ15 are
- ``cumulant=False``: use cumulants instead of moments
- ``initialVelocity=None``: initial velocity in domain, can either be a tuple (x,y,z) velocity to set a constant
  velocity everywhere, or a numpy array with the same size of the domain, with a last coordinate of shape dim to set
  velocities on cell level

Entropic methods:

- ``entropic=False``: In case there are two distinct relaxation rate in a method, one of them (usually the one, not
  determining the viscosity) can be automatically chosen w.r.t an entropy condition. For details see
  :mod:`lbmpy.methods.entropic`
- ``entropicNewtonIterations=None``: For moment methods the entropy optimum can be calculated in closed form.
  For cumulant methods this is not possible, in that case it is computed using Newton iterations. This parameter can be
  used to force Newton iterations and specify how many should be done
- ``omegaOutputField=None``: you can pass a pystencils Field here, where the calculated free relaxation
  rate is written to


Optimization Parameters
^^^^^^^^^^^^^^^^^^^^^^^

Simplifications / Transformations:

- ``doCseInOpposingDirections=False``: run common subexpression elimination for opposing stencil directions
- ``doOverallCse=False``: run common subexpression elimination after all other simplifications have been executed
- ``split=False``: split innermost loop, to handle only 2 directions per loop. This reduces the number of parallel
  load/store streams and thus speeds up the kernel on most architectures


Field size information:

- ``pdfArr=None``: pass a numpy array here to create kernels with fixed size and create the loop nest according to layout
  of this array
- ``fieldSize=None``: create kernel for fixed field size
- ``fieldLayout='c'``:   ``'c'`` or ``'numpy'`` for standard numpy layout, ``'reverseNumpy'`` or ``'f'`` for fortran
  layout, this does not apply when pdfArr was given, then the same layout as pdfArr is used

GPU:

- ``target='cpu'``: ``'cpu'`` or ``'gpu'``, last option requires a CUDA enabled graphics card
  and installed *pycuda* package
- ``gpuIndexing='block'``: determines mapping of CUDA threads to cells. Can be either ``'block'`` or ``'line'``
- ``gpuIndexingParams='block'``: parameters passed to init function of gpu indexing.
  For ``'block'`` indexing one can e.g. specify the block size ``{'blockSize' : (128, 4, 1)}``

Other:

- ``openMP=True``: only applicable for cpu simulations. Can be a boolean to turn multi threading on/off, or an integer
  specifying the number of threads. If True is specified OpenMP chooses the number of threads
- ``doublePrecision=True``:  by default simulations run with double precision floating point numbers, by setting this
  parameter to False, single precision is used, which is much faster, especially on GPUs




Terminology and creation pipeline
---------------------------------

Kernel functions are created in three steps:

1. *Method*:
         the method defines the collision process. Currently there are two big categories:
         moment and cumulant based methods. A method defines how each moment or cumulant is relaxed by
         storing the equilibrium value and the relaxation rate for each moment/cumulant.
2. *Collision/Update Rule*:
         Methods can generate a "collision rule" which is an equation collection that define the
         post collision values as a function of the pre-collision values. On these equation collection
         simplifications are applied to reduce the number of floating point operations.
         At this stage an entropic optimization step can also be added to determine one relaxation rate by an
         entropy condition.
         Then a streaming rule is added which transforms the collision rule into an update rule.
         The streaming step depends on the pdf storage (source/destination, AABB pattern, EsoTwist).
         Currently only the simple source/destination  pattern is supported.
3. *AST*:
        The abstract syntax tree describes the structure of the kernel, including loops and conditionals.
        The ast can be modified e.g. to add OpenMP pragmas, reorder loops or apply other optimizations.
4. *Function*:
        This step compiles the AST into an executable function, either for CPU or GPUs. This function
        behaves like a normal Python function and runs one LBM time step.

The function :func:`createLatticeBoltzmannFunction` runs the whole pipeline, the other functions in this module
execute this pipeline only up to a certain step. Each function optionally also takes the result of the previous step.

For example, to modify the AST one can run::

    ast = createLatticeBoltzmannAst(...)
    # modify ast here
    func = createLatticeBoltzmannFunction(ast=ast, ...)

"""
import sympy as sp
from copy import copy
from functools import partial

from lbmpy.methods import createSRT, createTRT, createOrthogonalMRT, createKBCTypeTRT, \
    createRawMRT, createThreeRelaxationRateMRT
from lbmpy.methods.entropic import addIterativeEntropyCondition, addEntropyCondition
from lbmpy.methods.relaxationrates import relaxationRateFromMagicNumber
from lbmpy.stencils import getStencil
import lbmpy.forcemodels as forceModels
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.updatekernels import createStreamPullKernel, createPdfArray


def updateWithDefaultParameters(params, optParams):
    defaultMethodDescription = {
        'stencil': 'D2Q9',
        'method': 'srt',  # can be srt, trt or mrt
        'relaxationRates': sp.symbols("omega_:10"),
        'compressible': False,
        'equilibriumAccuracyOrder': 2,

        'forceModel': 'none',  # can be 'simple', 'luo' or 'guo'
        'force': (0, 0, 0),
        'useContinuousMaxwellianEquilibrium': True,
        'cumulant': False,
        'initialVelocity': None,

        'entropic': False,
        'entropicNewtonIterations': None,
        'omegaOutputField': None,
    }

    defaultOptimizationDescription = {
        'doCseInOpposingDirections': False,
        'doOverallCse': False,
        'split': False,

        'fieldSize': None,
        'fieldLayout': 'c',  # can be 'numpy' (='c'), 'reverseNumpy' (='f'), 'fzyx', 'zyxf'

        'target': 'cpu',
        'openMP': True,
        'pdfArr': None,
        'doublePrecision': True,

        'gpuIndexing': 'block',
        'gpuIndexingParams': {},
    }
    if 'relaxationRate' in params:
        if 'relaxationRates' not in params:
            params['relaxationRates'] = [params['relaxationRate'],
                                         relaxationRateFromMagicNumber(params['relaxationRate'])]
            del params['relaxationRate']

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
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

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
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    if updateRule is None:
        params['optimizationParams'] = optimizationParams
        updateRule = createLatticeBoltzmannUpdateRule(**params)

    if optParams['target'] == 'cpu':
        from pystencils.cpu import createKernel
        if 'splitGroups' in updateRule.simplificationHints:
            splitGroups = updateRule.simplificationHints['splitGroups']
        else:
            splitGroups = ()
        res = createKernel(updateRule.allEquations, splitGroups=splitGroups,
                           typeForSymbol='double' if optParams['doublePrecision'] else 'float')
    elif optParams['target'] == 'gpu':
        from pystencils.gpucuda import createCUDAKernel
        from pystencils.gpucuda.indexing import LineIndexing, BlockIndexing
        assert optParams['gpuIndexing'] in ('line', 'block')
        indexingCreator = LineIndexing if optParams['gpuIndexing'] == 'line' else BlockIndexing
        if optParams['gpuIndexingParams']:
            indexingCreator = partial(indexingCreator, **optParams['gpuIndexingParams'])
        res = createCUDAKernel(updateRule.allEquations,
                               typeForSymbol='double' if optParams['doublePrecision'] else 'float',
                               indexingCreator=indexingCreator)
    else:
        return ValueError("'target' has to be either 'cpu' or 'gpu'")

    res.method = updateRule.method
    res.updateRule = updateRule
    return res


def createLatticeBoltzmannUpdateRule(lbMethod=None, optimizationParams={}, **kwargs):
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)
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
            if isinstance(params['entropicNewtonIterations'], bool) or params['cumulant']:
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
    params, _ = updateWithDefaultParameters(params, {})

    stencil = getStencil(params['stencil'])
    dim = len(stencil[0])

    forceIsZero = True
    for f_i in params['force']:
        if f_i != 0:
            forceIsZero = False

    noForceModel = 'forceModel' not in params or params['forceModel'] == 'none' or params['forceModel'] is None
    if not forceIsZero and noForceModel:
        params['forceModel'] = 'guo'

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

