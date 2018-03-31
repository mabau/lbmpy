import sympy as sp

from pystencils import Assignment
from lbmpy.relaxationrates import getShearRelaxationRate


def secondOrderMomentTensor(functionValues, stencil):
    """Returns (D x D) Matrix of second order moments of the given function where D is the dimension"""
    assert len(functionValues) == len(stencil)
    dim = len(stencil[0])
    return sp.Matrix(dim, dim, lambda i, j: sum(c[i] * c[j] * f for f, c in zip(functionValues, stencil)))


def frobeniusNorm(matrix, factor=1):
    """Computes the Frobenius norm of a matrix defined as the square root of the sum of squared matrix elements
    The optional factor is added inside the square root"""
    return sp.sqrt(sum(i * i for i in matrix) * factor)


def addSmagorinskyModel(collisionRule, smagorinskyConstant, omegaOutputField=None):
    method = collisionRule.method
    omega_s = getShearRelaxationRate(method)
    fNeq = sp.Matrix(method.preCollisionPdfSymbols) - method.getEquilibriumTerms()

    tau_0 = sp.Symbol("tau_0_")
    secondOrderNeqMoments = sp.Symbol("Pi")
    rho = method.zerothOrderEquilibriumMomentSymbol if method.conservedQuantityComputation.compressible else 1
    adaptedOmega = sp.Symbol("smagorinskyOmega")

    # for derivation see notebook demo_custom_LES_model.pynb
    eqs = [Assignment(tau_0, 1 / omega_s),
           Assignment(secondOrderNeqMoments, frobeniusNorm(secondOrderMomentTensor(fNeq, method.stencil), factor=2) / rho),
           Assignment(adaptedOmega, 2 / (tau_0 + sp.sqrt(18 * smagorinskyConstant**2 * secondOrderNeqMoments + tau_0**2)))]
    collisionRule.subexpressions += eqs
    collisionRule.topological_sort(sort_subexpressions=True, sort_main_assignments=False)

    if omegaOutputField:
        collisionRule.main_assignments.append(Assignment(omegaOutputField.center, adaptedOmega))

    return collisionRule
