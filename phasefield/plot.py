from lbmpy.plot2d import *


def _drawAngles(ax, groupedPoints):
    from matplotlib.lines import Line2D
    from matplotlib.text import Annotation
    from lbmpy.phasefield.postprocessing import getAngle

    xData = [groupedPoints[1][0], groupedPoints[0][0], groupedPoints[2][0]]
    yData = [groupedPoints[1][1], groupedPoints[0][1], groupedPoints[2][1]]

    v = [p - groupedPoints[0] for p in groupedPoints[1:]]
    v = [p / np.linalg.norm(p, ord=2) for p in v]
    direction = v[0] + v[1]

    if direction[1] > 1:
        ha = 'left'
    elif direction[1] < -1:
        ha = 'right'
    else:
        ha = 'center'

    if direction[0] > 1:
        va = 'bottom'
    elif direction[0] < -1:
        va = 'top'
    else:
        va = 'center'

    textPos = groupedPoints[0] + 10 * direction
    lines = Line2D(yData, xData, axes=ax, linewidth=3, color='k')
    ax.add_line(lines)

    angle = getAngle(*groupedPoints)
    annotation = Annotation('%.1f°' % (angle,), (textPos[1], textPos[0]), axes=ax, fontsize=15, ha=ha, va=va)
    ax.add_artist(annotation)


def phaseIndices(phi, **kwargs):
    singlePhaseArrays = [phi[..., i] for i in range(phi.shape[-1])]
    lastPhase = 1 - sum(singlePhaseArrays)
    singlePhaseArrays.append(lastPhase)
    idxArray = np.array(singlePhaseArrays).argmax(axis=0)
    return scalarField(idxArray, **kwargs)


def angles(phi0, phi1, phi2, branchingDistance=0.5, branchingPointFilter=3, onlyFirst=True):
    from lbmpy.phasefield.postprocessing import getTriplePointInfos

    levels = [0.5, ]
    scalarFieldContour(phi0, levels=levels)
    scalarFieldContour(phi1, levels=levels)
    scalarFieldContour(phi2, levels=levels)

    if onlyFirst:
        angleInfo = getTriplePointInfos(phi0, phi1, phi2, branchingDistance, branchingPointFilter)
    else:
        angleInfo = getTriplePointInfos(phi0, phi1, phi2, branchingDistance, branchingPointFilter) +\
                    getTriplePointInfos(phi1, phi0, phi2, branchingDistance, branchingPointFilter) +\
                    getTriplePointInfos(phi2, phi1, phi0, branchingDistance, branchingPointFilter)

    for points in angleInfo:
        _drawAngles(gca(), points)

