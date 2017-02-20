

def relaxationRateFromLatticeViscosity(nu):
    return 1.0 / (3 * nu + 0.5)


def latticeViscosityFromRelaxationRate(omega):
    return (1/omega - 1/2) / 3
