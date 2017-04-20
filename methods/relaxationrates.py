import sympy as sp
from lbmpy.moments import isShearMoment, getOrder


def relaxationRateFromLatticeViscosity(nu):
    """Computes relaxation rate from lattice viscosity: :math:`\omega = \frac{1}{3\nu_L + \frac{1}{2}}`"""
    return 1.0 / (3 * nu + 0.5)


def latticeViscosityFromRelaxationRate(omega):
    """Computes lattice viscosity from relaxation rate: 
    :math:`\nu_L=\frac{1}{3}\left(\frac{1}{\omega}-\frac{1}{2}\right)`"""
    return (1/omega - 1/2) / 3


def relaxationRateFromMagicNumber(hydrodynamicRelaxationRate, magicNumber=sp.Rational(3, 16)):
    """
    Computes second TRT relaxation rate from magic number
    """
    omega = hydrodynamicRelaxationRate
    return (4 - 2 * omega) / (4 * magicNumber * omega + 2 - omega)


def getShearRelaxationRate(method):
    """
    Assumes that all shear moments are relaxed with same rate - returns this rate
    Shear moments in 3D are: x*y, x*z and y*z - in 2D its only x*y
    The shear relaxation rate determines the viscosity in hydrodynamic LBM schemes
    """
    relaxationRates = set()
    for moment, relaxInfo in method.relaxationInfoDict.items():
        if isShearMoment(moment):
            relaxationRates.add(relaxInfo.relaxationRate)
    if len(relaxationRates) == 1:
        return relaxationRates.pop()
    else:
        if len(relaxationRates) > 1:
            raise ValueError("Shear moments are relaxed with different relaxation times: %s" % (relaxationRates,))
        else:
            raise NotImplementedError("Shear moments seem to be not relaxed separately - "
                                      "Can not determine their relaxation rate automatically")


def defaultRelaxationRateNames():
    nextIndex = [0]

    def result(momentList):
        shearMomentInside = False
        allConservedMoments = True
        for m in momentList:
            if isShearMoment(m):
                shearMomentInside = True
            if not (getOrder(m) == 0 or getOrder(m) == 1):
                allConservedMoments = False

        if shearMomentInside:
            return sp.Symbol("omega")
        elif allConservedMoments:
            return 0
        else:
            nextIndex[0] += 1
            return sp.Symbol("omega_%d" % (nextIndex[0],))

    return result