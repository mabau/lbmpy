import sympy as sp
import lbmpy.util as util
from lbmpy.equilibria import getWeights


def scalarProduct(a, b):
    return sum(a_i * b_i for a_i, b_i in zip(a, b))


def getForceModelEquations(stencil, forceSymbols, viscosityRelaxationRate=None, bulkRelaxationRate=None):
    pass


def getForceModelSimple(stencil, force, **kwargs):
    dim = len(stencil[0])
    assert len(force) == dim
    weights = getWeights(stencil)
    return [3 * w_i * scalarProduct(force, direction) for direction, w_i in zip(stencil, weights)]


def getForceModelLuo(stencil, force, **kwargs):
    dim = len(stencil[0])
    assert len(force) == dim
    u = sp.Matrix(util.getSymbolicVelocityVector(dim))

    force = sp.Matrix(force)

    weights = getWeights(stencil)
    result = []
    for dir, w_i in zip(stencil, weights):
        dir = sp.Matrix(dir)
        result.append(3 * w_i * force.dot(dir - u + 3 * dir * dir.dot(u)))
    return result


