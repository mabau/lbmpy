from lbmpy.boundaries import NoSlip
from lbmpy.plot2d import phasePlot
from pystencils import makeSlice


def dropsBetweenTwoPhase():
    import lbmpy.plot2d as plt
    import sympy as sp
    from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
    from lbmpy.phasefield.analytical import freeEnergyFunctionalNPhasesPenaltyTerm
    c = sp.symbols("c_:4")
    F = freeEnergyFunctionalNPhasesPenaltyTerm(c, 1, [0.01, 0.02, 0.001, 0.02], 0.0001)
    sc = PhaseFieldStep(F, c, domainSize=(2*200, 2*70), openMP=4, hydroDynamicRelaxationRate=1.9)
    sc.setConcentration(makeSlice[:, 0.5:], [1, 0, 0, 0])
    sc.setConcentration(makeSlice[:, :0.5], [0, 1, 0, 0])
    sc.setConcentration(makeSlice[0.2:0.4, 0.3:0.7], [0, 0, 1, 0])
    sc.setConcentration(makeSlice[0.7:0.8, 0.3:0.7], [0, 0, 0, 1])
    sc.setPdfFieldsFromMacroscopicValues()

    for i in range(500000):
        print("Step", i)
        plt.figure(figsize=(14, 7))
        phasePlot(sc, makeSlice[:, 5:-5], linewidth=0.1)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('step%03d.png' % (i,), bbox_inches='tight')
        plt.clf()
        sc.run(25)
    plt.show()


def fallingDrop():
    import lbmpy.plot2d as plt
    import sympy as sp
    from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
    from lbmpy.phasefield.analytical import freeEnergyFunctionalNPhasesPenaltyTerm
    c = sp.symbols("c_:3")
    F = freeEnergyFunctionalNPhasesPenaltyTerm(c, 1, [0.001, 0.001, 0.0005])
    gravity = -0.5e-5
    sc = PhaseFieldStep(F, c, domainSize=(160, 200), openMP=4,
                        hydroDynamicRelaxationRate=1.9,
                        orderParameterForce={2: (0, gravity), 1: (0, 0)})
    sc.setConcentration(makeSlice[:, 0.4:], [1, 0, 0])
    sc.setConcentration(makeSlice[:, :0.4], [0, 1, 0])
    sc.setConcentration(makeSlice[0.45:0.55, 0.8:0.9], [0, 0, 1])
    sc.hydroLbmStep.boundaryHandling.setBoundary(NoSlip(), makeSlice[:, 0])
    #sc.hydroLbmStep.boundaryHandling.setBoundary(NoSlip(), makeSlice[:, -1])

    sc.setPdfFieldsFromMacroscopicValues()
    for i in range(650):
        print("Step", i)
        plt.figure(figsize=(14, 10))
        plt.subplot(1, 2, 1)
        phasePlot(sc, makeSlice[:, 5:-15], linewidth=0.1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.vectorField(sc.velocity[:, 5:-15, :])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('fallingDrop_boundary2_%05d.png' % (i,), bbox_inches='tight')
        plt.clf()
        sc.run(200)
    plt.show()


if __name__ == '__main__':
    fallingDrop()
