import numpy as np
from itertools import cycle

from lbmpy.boundaries import NoSlip
from pystencils import makeSlice


def setPhase(pfStep, phaseIndex, sliceObj, value=1.0, others=0.0):
    for b in pfStep.dataHandling.iterate(sliceObj):
        b[pfStep.phiFieldName][..., phaseIndex].fill(value)
        if others is not None:
            for i in range(pfStep.numOrderParameters):
                if i == phaseIndex:
                    continue
                else:
                    b[pfStep.phiFieldName][..., i].fill(others)


def boxBetweenPhases(pfStep, phaseIndices=(0, 1, 2), values=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                     boxX=(0.25, 0.75), boxY=(0.25, 0.75)):
    assert all(idx < pfStep.numOrderParameters for idx in phaseIndices)
    assert pfStep.dataHandling.dim == 2

    domainSize = pfStep.dataHandling.shape
    x1, x2 = (int(c * domainSize[0]) for c in boxX)
    y1, y2 = (int(c * domainSize[1]) for c in boxY)
    
    pfStep.reset()
    
    for b in pfStep.dataHandling.iterate():
        x, y = b.cellIndexArrays[0:2]
        phi = b[pfStep.phiFieldName]

        innerBox = np.logical_and(np.logical_and(x < x2, x > x1),
                                  np.logical_and(y < y2, y > y1))

        upperValues, lowerValues, midValues = values
        for phaseIdx, val in zip(phaseIndices, lowerValues):
            phi[y <= domainSize[1] //2, phaseIdx] = val
        for phaseIdx, val in zip(phaseIndices, upperValues):
            phi[y > domainSize[1] // 2, phaseIdx] = val
        for phaseIdx, val in zip(phaseIndices, midValues):
            phi[innerBox, phaseIdx] = val

    pfStep.setPdfFieldsFromMacroscopicValues()


def initThreeBoxes(pfStep, phaseIndices=(0, 1, 2), values=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    assert all(idx < pfStep.numOrderParameters for idx in phaseIndices)
    assert pfStep.dataHandling.dim == 2

    domainSize = pfStep.dataHandling.shape
    x1, x2 = (domainSize[0] // 3), (domainSize[0] // 3) * 2
    y1, y2 = (domainSize[1] // 3), (domainSize[1] // 3) * 2

    pfStep.reset()

    for b in pfStep.dataHandling.iterate():
        x, y = b.cellIndexArrays[0:2]
        phi = b[pfStep.phiFieldName]
        xMid = np.logical_and(x1 < x, x < x2)
        yMid = np.logical_and(y1 < y, y < y2)

        center = np.logical_and(xMid, yMid)
        side = np.logical_and(np.logical_not(xMid),yMid)
        topBottom = np.logical_not(yMid)

        midValues, sideValues, topBottomValues = values
        for phaseIdx, val in zip(phaseIndices, midValues):
            phi[center, phaseIdx] = val
        for phaseIdx, val in zip(phaseIndices, sideValues):
            phi[side, phaseIdx] = val
        for phaseIdx, val in zip(phaseIndices, topBottomValues):
            phi[topBottom, phaseIdx] = val

    pfStep.setPdfFieldsFromMacroscopicValues()


def phasePlot(pfStep, sliceObj=makeSlice[:, :], linewidth=1.0):
    import lbmpy.plot2d as plt
    colors = ['#fe0002', '#00fe00', '#0000ff', '#ffa800', '#f600ff']
    colorCycle = cycle(colors)

    for i in range(pfStep.numOrderParameters):
        s = sliceObj + (i,)
        plt.scalarFieldAlphaValue(pfStep.phi[s], next(colorCycle), clip=True, interpolation='bilinear')
    for i in range(pfStep.numOrderParameters):
        s = sliceObj + (i,)
        plt.scalarFieldContour(pfStep.phi[s], levels=[0.5], colors='k', linewidth=linewidth)


def dropsBetweenTwoPhase():
    import lbmpy.plot2d as plt
    import sympy as sp
    from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
    from lbmpy.phasefield.analytical import freeEnergyFunctionalNPhasesPenaltyTerm
    c = sp.symbols("c_:4")
    F = freeEnergyFunctionalNPhasesPenaltyTerm(c, 1, [0.01, 0.02, 0.001, 0.02], 0.0001)
    sc = PhaseFieldStep(F, c, domainSize=(2*200, 2*70), openMP=4, hydroDynamicRelaxationRate=1.9)
    setPhase(sc, 0, makeSlice[:, 0.5:])
    setPhase(sc, 1, makeSlice[:, :0.5])
    setPhase(sc, 2, makeSlice[0.2:0.4, 0.3:0.7])
    setPhase(sc, 3, makeSlice[0.7:0.8, 0.3:0.7])
    sc.setPdfFieldsFromMacroscopicValues()
    #sc.run(1)
    #for i in range(3):
    #    plt.subplot(3, 1, i+1)
    #    plt.scalarField(sc.phi[:, :, i])
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
    F = freeEnergyFunctionalNPhasesPenaltyTerm(c, 1, [0.0008, 0.0008, 0.00002])
    gravity = -1e-5
    sc = PhaseFieldStep(F, c, domainSize=(160, 200), openMP=4,
                        hydroDynamicRelaxationRate=1.9,
                        orderParameterForce={2: (0, gravity), 1: (0, gravity)})
    setPhase(sc, 0, makeSlice[:, 0.4:])
    setPhase(sc, 1, makeSlice[:, :0.4])
    setPhase(sc, 2, makeSlice[0.45:0.55, 0.8:0.9])
    sc.hydroLbmStep.boundaryHandling.setBoundary(NoSlip(), makeSlice[:, 0])
    sc.hydroLbmStep.boundaryHandling.setBoundary(NoSlip(), makeSlice[:, -1])

    sc.setPdfFieldsFromMacroscopicValues()
    #sc.run(1)
    #for i in range(3):
    #    plt.subplot(3, 1, i+1)
    #    plt.scalarField(sc.phi[:, :, i])
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