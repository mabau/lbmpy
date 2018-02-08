import numpy as np
from pystencils import makeSlice


def plotStatus(phaseFieldStep, fromX=None, toX=None):
    import lbmpy.plot2d as plt

    domainSize = phaseFieldStep.dataHandling.shape
    assert len(domainSize) == 2 and domainSize[1] == 1, "Not a 1D scenario"

    dh = phaseFieldStep.dataHandling

    numPhases = phaseFieldStep.numOrderParameters

    plt.subplot(1, 3, 1)
    plt.title('φ')
    phiName = phaseFieldStep.phiFieldName
    for i in range(numPhases):
        plt.plot(dh.gatherArray(phiName, makeSlice[fromX:toX, 0, i]), marker='x', label='φ_%d' % (i,))
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.title("μ")
    muName = phaseFieldStep.muFieldName
    for i in range(numPhases):
        plt.plot(dh.gatherArray(muName, makeSlice[fromX:toX, 0, i]), marker='x', label='μ_%d' % (i,));
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.title("Force and Velocity")
    plt.plot(dh.gatherArray(phaseFieldStep.forceFieldName, makeSlice[fromX:toX, 0, 0]), label='F', marker='x')
    plt.plot(dh.gatherArray(phaseFieldStep.velFieldName, makeSlice[fromX:toX, 0, 0]), label='u', marker='v')
    plt.legend()


def initSharpInterface(pfStep, phaseIdx1=1, phaseIdx2=2, x1=None, x2=None):
    domainSize = pfStep.dataHandling.shape
    if x1 is None:
        x1 = domainSize[0] // 4
    if x2 is None:
        x2 = 3 * x1

    for b in pfStep.dataHandling.iterate():
        x = b.cellIndexArrays[0]
        mid = np.logical_and(x1 < x, x < x2)

        phi = b[pfStep.phiFieldName]

        if phaseIdx1 is not None:
            phi[..., phaseIdx1].fill(0)
            phi[mid, phaseIdx1] = 1

        if phaseIdx2 is not None:
            phi[..., phaseIdx2].fill(1)
            phi[mid, phaseIdx2] = 0
    pfStep.setPdfFieldsFromMacroscopicValues()
