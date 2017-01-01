import sympy as sp
from collections import namedtuple
from lbmpy.methods.abstractlbmmethod import AbstractLbmMethod
from lbmpy.moments import MOMENT_SYMBOLS, momentMatrix

"""
Ways to create method:
    - moment (polynomial or tuple) mapped to relaxation rate
    - moment matrix & relaxation vector
    - createSRT, createTRT, createMRT
"""


RelaxationInfo = namedtuple('Relaxationinfo', ['equilibriumValue', 'relaxationRate'])


class MomentBasedLbmMethod(AbstractLbmMethod):

    def __init__(self, stencil, momentToRelaxationInfoDict, forceModel=None,
                 equilibriumValueComputation='defaultIncompressible',
                 macroscopicValueComputations='defaultIncompressible'):
        """

        :param stencil:
        :param momentToRelaxationInfoDict:
        :param forceModel:
        :param equilibriumValueComputation: equation collection where first equation corresponds to zeroth order moment
                                            and the following to the first order moments
        :param macroscopicValueComputations:
        """
        assert MOMENT_SYMBOLS[0] in momentToRelaxationInfoDict.keys(), ""
        pass

    def getEquilibrium(self):
        return

    def getCollisionRule(self):
        return


def createByMatchingMoments(stencil, moments, ):
    pass


def createSRT(stencil, relaxationRate):
    pass


def createTRT(stencil, relaxationRateEvenMoments, relaxationRateOddMoments):
    pass


def createMRT():
    pass




