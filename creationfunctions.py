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
    - ``trt-kbc-n<N>`` where <N> is 1,2,3 or 4. Special two-relaxation rate method. This is not the entropic method
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
- ``forceModel=None``: possible values: ``None``, ``'simple'``, ``'luo'``, ``'guo'`` ``'buick'``, or an instance of a
  class implementing the same methods as the classes in :mod:`lbmpy.forcemodels`. For details, see
  :mod:`lbmpy.forcemodels`
- ``force=(0,0,0)``: either constant force or a symbolic expression depending on field value
- ``useContinuousMaxwellianEquilibrium=True``: way to compute equilibrium moments/cumulants, if False the standard
  discretized LBM equilibrium is used, otherwise the equilibrium moments are computed from the continuous Maxwellian.
  This makes only a difference if sparse stencils are used e.g. D2Q9 and D3Q27 are not affected, D319 and DQ15 are
- ``cumulant=False``: use cumulants instead of moments
- ``initialVelocity=None``: initial velocity in domain, can either be a tuple (x,y,z) velocity to set a constant
  velocity everywhere, or a numpy array with the same size of the domain, with a last coordinate of shape dim to set
  velocities on cell level
- ``output={}``: a dictionary mapping macroscopic quantites e.g. the strings 'density' and 'velocity' to pystencils
                fields. In each timestep the corresponding quantities are written to the given fields.
- ``velocityInput``: symbolic field where the velocities are read from (for advection diffusion LBM)
- ``kernelType``: supported values: 'streamPullCollide' (default), 'collideOnly' 


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
- ``builtinPeriodicity=(False,False,False)``: instead of handling periodicity by copying ghost layers, the periodicity
  is built into the kernel. This parameters specifies if the domain is periodic in (x,y,z) direction. Even if the
  periodicity is built into the kernel, the fields have one ghost layer to be consistent with other functions. 
    

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
from pystencils.cache import diskcacheNoFallback
from lbmpy.methods import createSRT, createTRT, createOrthogonalMRT, createKBCTypeTRT, \
    createRawMRT, createThreeRelaxationRateMRT
from lbmpy.methods.entropic import addIterativeEntropyCondition, addEntropyCondition
from lbmpy.methods.entropic_eq_srt import createEntropicSRT
from lbmpy.methods.relaxationrates import relaxationRateFromMagicNumber
from lbmpy.stencils import getStencil, stencilsHaveSameEntries
import lbmpy.forcemodels as forcemodels
from lbmpy.simplificationfactory import createSimplificationStrategy
from lbmpy.updatekernels import StreamPullTwoFieldsAccessor, PeriodicTwoFieldsAccessor, CollideOnlyInplaceAccessor, \
    createLBMKernel
from pystencils.data_types import collateTypes
from pystencils.equationcollection.equationcollection import EquationCollection
from pystencils.field import getLayoutOfArray, Field
from pystencils import createKernel


def createLatticeBoltzmannFunction(ast=None, optimizationParams={}, **kwargs):
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    if ast is None:
        params['optimizationParams'] = optParams
        ast = createLatticeBoltzmannAst(**params)

    res = ast.compile()

    res.method = ast.method
    res.updateRule = ast.updateRule
    res.ast = ast
    return res


def createLatticeBoltzmannAst(updateRule=None, optimizationParams={}, **kwargs):
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    if updateRule is None:
        params['optimizationParams'] = optimizationParams
        updateRule = createLatticeBoltzmannUpdateRule(**params)

    fieldTypes = set(fa.field.dtype for fa in updateRule.freeSymbols if isinstance(fa, Field.Access))
    res = createKernel(updateRule, target=optParams['target'], dataType=collateTypes(fieldTypes),
                       cpuOpenMP=optParams['openMP'], cpuVectorizeInfo=optParams['vectorization'],
                       gpuIndexing=optParams['gpuIndexing'], gpuIndexingParams=optParams['gpuIndexingParams'],
                       ghostLayers=1)

    res.method = updateRule.method
    res.updateRule = updateRule
    return res


@diskcacheNoFallback
def createLatticeBoltzmannCollisionRule(lbMethod=None, optimizationParams={}, **kwargs):
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    if lbMethod is None:
        lbMethod = createLatticeBoltzmannMethod(**params)

    splitInnerLoop = 'split' in optParams and optParams['split']

    dirCSE = 'doCseInOpposingDirections'
    doCseInOpposingDirections = False if dirCSE not in optParams else optParams[dirCSE]
    doOverallCse = False if 'doOverallCse' not in optParams else optParams['doOverallCse']
    simplification = createSimplificationStrategy(lbMethod, doCseInOpposingDirections, doOverallCse, splitInnerLoop)
    cqc = lbMethod.conservedQuantityComputation

    if params['velocityInput'] is not None:
        eqs = [sp.Eq(cqc.zerothOrderMomentSymbol, sum(lbMethod.preCollisionPdfSymbols))]
        velocityField = params['velocityInput']
        eqs += [sp.Eq(uSym, velocityField(i)) for i, uSym in enumerate(cqc.firstOrderMomentSymbols)]
        eqs = EquationCollection(eqs, [])
        collisionRule = lbMethod.getCollisionRule(conservedQuantityEquations=eqs)
    else:
        collisionRule = lbMethod.getCollisionRule()

    if params['output']:
        outputEqs = cqc.outputEquationsFromPdfs(lbMethod.preCollisionPdfSymbols, params['output'])
        collisionRule = collisionRule.merge(outputEqs)

    return simplification(collisionRule)


@diskcacheNoFallback
def createLatticeBoltzmannUpdateRule(collisionRule=None, optimizationParams={}, **kwargs):
    params, optParams = updateWithDefaultParameters(kwargs, optimizationParams)

    if collisionRule is None:
        collisionRule = createLatticeBoltzmannCollisionRule(**params, optimizationParams=optParams)

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

    fieldDtype = 'float64' if optParams['doublePrecision'] else 'float32'

    if optParams['symbolicField'] is not None:
        srcField = optParams['symbolicField']
    elif optParams['fieldSize']:
        fieldSize = [s + 2 for s in optParams['fieldSize']] + [len(collisionRule.stencil)]
        srcField = Field.createFixedSize(params['fieldName'], fieldSize, indexDimensions=1,
                                         layout=optParams['fieldLayout'], dtype=fieldDtype)
    else:
        srcField = Field.createGeneric(params['fieldName'], spatialDimensions=collisionRule.method.dim,
                                       indexDimensions=1, layout=optParams['fieldLayout'], dtype=fieldDtype)

    dstField = srcField.newFieldWithDifferentName(params['secondFieldName'])

    if params['kernelType'] == 'streamPullCollide':
        accessor = StreamPullTwoFieldsAccessor
        if any(optParams['builtinPeriodicity']):
            accessor = PeriodicTwoFieldsAccessor(optParams['builtinPeriodicity'], ghostLayers=1)
        return createLBMKernel(collisionRule, srcField, dstField, accessor)
    elif params['kernelType'] == 'collideOnly':
        return createLBMKernel(collisionRule, srcField, srcField, CollideOnlyInplaceAccessor)
    else:
        raise ValueError("Invalid value of parameter 'kernelType'", params['kernelType'])


def createLatticeBoltzmannMethod(**params):
    params, _ = updateWithDefaultParameters(params, {}, failOnUnknownParameter=False)

    if isinstance(params['stencil'], tuple) or isinstance(params['stencil'], list):
        stencilEntries = params['stencil']
    else:
        stencilEntries = getStencil(params['stencil'])

    dim = len(stencilEntries[0])

    if isinstance(params['force'], Field):
        params['force'] = tuple(params['force'](i) for i in range(dim))

    forceIsZero = True
    for f_i in params['force']:
        if f_i != 0:
            forceIsZero = False

    noForceModel = 'forceModel' not in params or params['forceModel'] == 'none' or params['forceModel'] is None
    if not forceIsZero and noForceModel:
        params['forceModel'] = 'guo'

    if 'forceModel' in params:
        forceModel = forceModelFromString(params['forceModel'], params['force'][:dim])
    else:
        forceModel = None

    commonParams = {
        'compressible': params['compressible'],
        'equilibriumAccuracyOrder': params['equilibriumAccuracyOrder'],
        'forceModel': forceModel,
        'useContinuousMaxwellianEquilibrium': params['useContinuousMaxwellianEquilibrium'],
        'cumulant': params['cumulant'],
        'c_s_sq': params['c_s_sq'],
    }
    methodName = params['method']
    relaxationRates = params['relaxationRates']

    if methodName.lower() == 'srt':
        assert len(relaxationRates) >= 1, "Not enough relaxation rates"
        method = createSRT(stencilEntries, relaxationRates[0], **commonParams)
    elif methodName.lower() == 'trt':
        assert len(relaxationRates) >= 2, "Not enough relaxation rates"
        method = createTRT(stencilEntries, relaxationRates[0], relaxationRates[1], **commonParams)
    elif methodName.lower() == 'mrt':
        nextRelaxationRate = [0]

        def relaxationRateGetter(momentGroup):
            res = relaxationRates[nextRelaxationRate[0]]
            nextRelaxationRate[0] += 1
            return res
        method = createOrthogonalMRT(stencilEntries, relaxationRateGetter, **commonParams)
    elif methodName.lower() == 'mrt_raw':
        method = createRawMRT(stencilEntries, relaxationRates, **commonParams)
    elif methodName.lower() == 'mrt3':
        method = createThreeRelaxationRateMRT(stencilEntries, relaxationRates, **commonParams)
    elif methodName.lower().startswith('trt-kbc-n'):
        if stencilsHaveSameEntries(stencilEntries, getStencil("D2Q9")):
            dim = 2
        elif stencilsHaveSameEntries(stencilEntries, getStencil("D3Q27")):
            dim = 3
        else:
            raise NotImplementedError("KBC type TRT methods can only be constructed for D2Q9 and D3Q27 stencils")
        methodNr = methodName[-1]
        method = createKBCTypeTRT(dim, relaxationRates[0], relaxationRates[1], 'KBC-N' + methodNr, **commonParams)
    elif methodName.lower() == 'entropic-srt':
        method = createEntropicSRT(stencilEntries, relaxationRates[0], forceModel, params['compressible'])
    else:
        raise ValueError("Unknown method %s" % (methodName,))

    return method


# ----------------------------------------------------------------------------------------------------------------------


def forceModelFromString(forceModelName, forceValues):
    if type(forceModelName) is not str:
        forceModel = forceModelName
    elif forceModelName.lower() == 'none':
        forceModel = None
    elif forceModelName.lower() == 'simple':
        forceModel = forcemodels.Simple(forceValues)
    elif forceModelName.lower() == 'luo':
        forceModel = forcemodels.Luo(forceValues)
    elif forceModelName.lower() == 'guo':
        forceModel = forcemodels.Guo(forceValues)
    elif forceModelName.lower() == 'silva' or forceModelName.lower() == 'buick':
        forceModel = forcemodels.Buick(forceValues)
    else:
        raise ValueError("Unknown force model %s" % (forceModelName,))
    return forceModel


def switchToSymbolicRelaxationRatesForEntropicMethods(methodParameters, kernelParams):
    """
    For entropic kernels the relaxation rate has to be a variable. If a constant was passed a
    new dummy variable is inserted and the value of this variable is later on passed to the kernel
    """
    if methodParameters['entropic']:
        valueToSymbolMap = {}
        newRelaxationRates = []
        for rr in methodParameters['relaxationRates']:
            if not isinstance(rr, sp.Symbol):
                if rr not in valueToSymbolMap:
                    valueToSymbolMap[rr] = sp.Dummy()
                dummyVar = valueToSymbolMap[rr]
                newRelaxationRates.append(dummyVar)
                kernelParams[dummyVar.name] = rr
            else:
                newRelaxationRates.append(rr)
        if len(newRelaxationRates) < 2:
            newRelaxationRates.append(sp.Dummy())
        methodParameters['relaxationRates'] = newRelaxationRates


def updateWithDefaultParameters(params, optParams, failOnUnknownParameter=True):
    defaultMethodDescription = {
        'stencil': 'D2Q9',
        'method': 'srt',  # can be srt, trt or mrt
        'relaxationRates': None,
        'compressible': False,
        'equilibriumAccuracyOrder': 2,
        'c_s_sq': sp.Rational(1, 3),

        'forceModel': 'none',
        'force': (0, 0, 0),
        'useContinuousMaxwellianEquilibrium': True,
        'cumulant': False,
        'initialVelocity': None,

        'entropic': False,
        'entropicNewtonIterations': None,
        'omegaOutputField': None,

        'output': {},
        'velocityInput': None,

        'kernelType': 'streamPullCollide',

        'fieldName': 'src',
        'secondFieldName': 'dst',

        'lbMethod': None,
        'collisionRule': None,
        'updateRule': None,
        'ast': None,
    }

    defaultOptimizationDescription = {
        'doCseInOpposingDirections': False,
        'doOverallCse': False,
        'split': False,

        'fieldSize': None,
        'fieldLayout': 'fzyx',  # can be 'numpy' (='c'), 'reverseNumpy' (='f'), 'fzyx', 'zyxf'
        'symbolicField': None,

        'target': 'cpu',
        'openMP': False,
        'doublePrecision': True,

        'gpuIndexing': 'block',
        'gpuIndexingParams': {},

        'vectorization': None,

        'builtinPeriodicity': (False, False, False),
    }
    if 'relaxationRate' in params:
        if 'relaxationRates' not in params:
            if 'entropic' in params and params['entropic']:
                params['relaxationRates'] = [params['relaxationRate']]
            else:
                params['relaxationRates'] = [params['relaxationRate'],
                                             relaxationRateFromMagicNumber(params['relaxationRate'])]

            del params['relaxationRate']

    if 'pdfArr' in optParams:
        arr = optParams['pdfArr']
        optParams['fieldSize'] = tuple(e - 2 for e in arr.shape[:-1])
        optParams['fieldLayout'] = getLayoutOfArray(arr)
        del optParams['pdfArr']

    if failOnUnknownParameter:
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

    if paramsResult['relaxationRates'] is None:
        stencilParam = paramsResult['stencil']
        if isinstance(stencilParam, tuple) or isinstance(stencilParam, list):
            stencilEntries = stencilParam
        else:
            stencilEntries = getStencil(paramsResult['stencil'])
        paramsResult['relaxationRates'] = sp.symbols("omega_:%d" % len(stencilEntries))

    return paramsResult, optParamsResult