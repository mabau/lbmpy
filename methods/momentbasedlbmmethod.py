import sympy as sp
from collections import namedtuple
from lbmpy.methods.abstractlbmmethod import AbstractLbmMethod
from lbmpy.methods.defaultMacroscopicValueEquations import densityVelocityExpressionsForEquilibrium, \
    densityVelocityExpressionsForOutput
from lbmpy.moments import MOMENT_SYMBOLS, momentMatrix
from pystencils.equationcollection import EquationCollection

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
        # Create moment matrix
        #
        super(MomentBasedLbmMethod, self).__init__(stencil)

        if equilibriumValueComputation in ('defaultIncompressible', 'defaultCompressible'):
            compressible = (equilibriumValueComputation == 'defaultCompressible')
            symbolicDensity = sp.Symbol("rho")
            symbolicVelocities = sp.Symbol("u_:%d" % ())
            eqs = densityVelocityExpressionsForEquilibrium(self.stencil, self.preCollisionPdfSymbols, compressible,
                                                           symbolicDensity, symbolicVelocities, forceModel)
            self._equilibriumValueEquations = eqs
        else:
            assert isinstance(equilibriumValueComputation, EquationCollection)
            self._equilibriumValueEquations = equilibriumValueComputation

    def getEquilibrium(self):
        # set relaxation rates to one
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




