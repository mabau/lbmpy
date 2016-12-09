import sympy as sp
from pystencils.field import Field
from pystencils.cpu import createKernel, makePythonFunction
from lbmpy.equilibria import standardDiscreteEquilibrium
from lbmpy.densityVelocityExpressions import getDensityVelocityExpressions


def compileMacroscopicValuesGetter(latticeModel, pdfArr=None, macroscopicFieldLayout='numpy'):
    """
    Creates a function that computes density and/or velocity and stores it into given arrays
    :param latticeModel: lattice model (to get information about stencil, force velocity shift and compressibility)
    :param pdfArr: array to set can already be specified here, or later when the returned function is called
    :param macroscopicFieldLayout: layout specification for Field.createGeneric
    :return: a function, which has three parameters:
        - pdfArray, can be omitted if pdf array was already passed while compiling
        - densityOut, if not None, density is written to that array
        - velocityOUt, if not None, velocity is written to that array
    """
    if pdfArr is None:
        pdfField = Field.createGeneric('pdfs', latticeModel.dim, indexDimensions=1)
    else:
        pdfField = Field.createFromNumpyArray('pdfs', pdfArr, indexDimensions=1)

    rhoField = Field.createGeneric('rho', latticeModel.dim, indexDimensions=0, layout=macroscopicFieldLayout)
    velField = Field.createGeneric('vel', latticeModel.dim, indexDimensions=1, layout=macroscopicFieldLayout)

    lm = latticeModel
    Q = len(lm.stencil)
    symPdfs = [pdfField(i) for i in range(Q)]
    subexpressions, rho, velSubexpressions, u = getDensityVelocityExpressions(lm.stencil, symPdfs, lm.compressible)

    uRhs = [u_i.rhs for u_i in u]
    uLhs = [u_i.lhs for u_i in u]
    if hasattr(lm.forceModel, "macroscopicVelocity"):
        correctedVel = lm.forceModel.macroscopicVelocity(lm, uRhs, rho.lhs)
        u = [sp.Eq(a, b) for a, b in zip(uLhs, correctedVel)]

    eqsRhoKernel = subexpressions + [sp.Eq(rhoField(0), rho.rhs)]
    eqsVelKernel = subexpressions + [rho] + velSubexpressions + [sp.Eq(velField(i), u[i].rhs) for i in range(lm.dim)]
    eqsRhoAndVelKernel = eqsVelKernel + [sp.Eq(rhoField(0), rho.lhs)]

    kernelRho = makePythonFunction(createKernel(eqsRhoKernel))
    kernelVel = makePythonFunction(createKernel(eqsVelKernel))
    kernelRhoAndVel = makePythonFunction(createKernel(eqsRhoAndVelKernel))

    def getter(pdfs=None, densityOut=None, velocityOut=None):
        if pdfs is not None and pdfArr is not None:
            assert pdfs.shape == pdfArr.shape and pdfs.strides == pdfArr.strides, \
                "Pdf array not matching blueprint which was used to compile"
        if pdfs is None:
            assert pdfArr is not None, "Pdf array has to be passed either when compiling or when calling."
            pdfs = pdfArr

        assert not (densityOut is None and velocityOut is None), \
            "Specify either densityOut or velocityOut parameter or both"

        if densityOut is not None and velocityOut is None:
            kernelRho(pdfs=pdfs, rho=densityOut)
        elif densityOut is None and velocityOut is not None:
            kernelVel(pdfs=pdfs, vel=velocityOut)
        else:
            kernelRhoAndVel(pdfs=pdfs, rho=densityOut, vel=velocityOut)

    return getter


def compileMacroscopicValuesSetter(latticeModel, density=1, velocity=0, equilibriumOrder=2, pdfArr=None):
    """
    Creates a function that sets a pdf field to specified macroscopic quantities
    The returned function can be called with the pdf field to set as single argument

    :param latticeModel: lattice model (to get information about stencil, force velocity shift and compressibility)
    :param density: density used for equilibrium. Can either be scalar (all cells are set to same density) or array
    :param velocity: velocity for equilibrium. Can either be scalar (e.g. 0, sets all cells to (0,0,0) velocity)
                    or a tuple with D (2 or 3) entries to set same velocity in  the complete domain, or an array
                    specifying the velocity for each cell
    :param equilibriumOrder: approximation order of equilibrium
    :param pdfArr: array to set can already be specified here, or later when the returned function is called
    """
    if pdfArr is not None:
        pdfField = Field.createFromNumpyArray('pdfs', pdfArr, indexDimensions=1)
    else:
        pdfField = Field.createGeneric('pdfs', latticeModel.dim, indexDimensions=1)

    noOffset = tuple([0] * latticeModel.dim)
    kernelArguments = {}

    if hasattr(density, 'shape'):
        densityValue = Field.createFromNumpyArray('rho', density, indexDimensions=0)[noOffset]
        kernelArguments['rho'] = density
    else:
        densityValue = density

    if hasattr(velocity, 'shape'):
        assert velocity.shape[-1] == latticeModel.dim, "Wrong shape of velocity array"
        velocityValue = [Field.createFromNumpyArray('vel', velocity, indexDimensions=1)[noOffset](i)
                         for i in range(latticeModel.dim)]
        kernelArguments['vel'] = velocity
    else:
        if not hasattr(velocity, "__len__"):
            velocity = [velocity] * latticeModel.dim
        velocityValue = tuple(velocity)

    # force shift
    if latticeModel.forceModel and hasattr(latticeModel.forceModel, "macroscopicVelocity"):
        # force model knows only about one direction - use sympy to solve the shift equations to get backward
        fm = latticeModel.forceModel
        unshiftedVel = [sp.Symbol("v_unshifted_%d" % (i,)) for i in range(latticeModel.dim)]
        shiftedVel = fm.macroscopicVelocity(latticeModel, unshiftedVel, densityValue)
        velShiftEqs = [sp.Eq(a, b) for a, b in zip(velocityValue, shiftedVel)]
        solveRes = sp.solve(velShiftEqs, unshiftedVel)
        assert len(solveRes) == latticeModel.dim
        velocityValue = [solveRes[unshiftedVel_i] for unshiftedVel_i in unshiftedVel]

    eq = standardDiscreteEquilibrium(latticeModel.stencil, densityValue, velocityValue, equilibriumOrder,
                                     c_s_sq=sp.Rational(1, 3), compressible=latticeModel.compressible)
    updateEquations = [sp.Eq(pdfField(i), eq[i]) for i in range(len(latticeModel.stencil))]
    f = makePythonFunction(createKernel(updateEquations), kernelArguments)

    def setter(pdfs=None, **kwargs):
        if pdfs is not None and pdfArr is not None:
            assert pdfs.shape == pdfArr.shape and pdfs.strides == pdfArr.strides, \
                "Pdf array not matching blueprint which was used to compile"

        if pdfs is None:
            assert pdfArr is not None, "Pdf array has to be passed either when compiling or when calling."
            pdfs = pdfArr
        assert pdfs.shape[-1] == len(latticeModel.stencil), "Wrong shape of pdf array"
        assert len(pdfs.shape) == latticeModel.dim + 1, "Wrong shape of pdf array"
        if hasattr(density, 'shape'):
            assert pdfs.shape[:-1] == density.shape, "Density array shape does not match pdf array shape"
        if hasattr(velocity, 'shape'):
            assert pdfs.shape[:-1] == velocity.shape[:-1], "Velocity array shape does not match pdf array shape"
        f(pdfs=pdfs, **kwargs)  # kwargs passed here, since force model might need additional fields

    return setter
