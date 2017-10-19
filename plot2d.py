from pystencils.plot2d import *
from pystencils.slicing import normalizeSlice


def plotBoundaryHandling(boundaryHandling, slice=None, boundaryNameToColor=None, legend=True):
    """
    Shows boundary cells

    :param boundaryHandling: instance of :class:`lbmpy.boundaries.BoundaryHandling`
    :param boundaryNameToColor: optional dictionary mapping boundary names to colors
    """
    import matplotlib
    import matplotlib.pyplot as plt

    boundaryHandling.prepare()

    if len(boundaryHandling.flagField.shape) != 2 and slice is None:
        raise ValueError("To plot 3D boundary handlings a slice has to be passed")

    if boundaryNameToColor:
        fixedColors = boundaryNameToColor
    else:
        fixedColors = {
            'fluid': '#56b4e9',
            'NoSlip': '#999999',
            'UBB': '#d55e00',
            'FixedDensity': '#009e73',
        }

    boundaryNames = []
    flagValues = []
    for name, flag in sorted(boundaryHandling.getBoundaryNameToFlagDict().items(), key=lambda l: l[1]):
        boundaryNames.append(name)
        flagValues.append(flag)
    defaultCycler = matplotlib.rcParams['axes.prop_cycle']
    colorValues = [fixedColors[name] if name in fixedColors else cycle['color']
                   for cycle, name in zip(defaultCycler, boundaryNames)]

    cmap = matplotlib.colors.ListedColormap(colorValues)
    bounds = np.array(flagValues, dtype=float) - 0.5
    bounds = list(bounds) + [bounds[-1] + 1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    flagField = boundaryHandling.flagField
    if slice:
        flagField = flagField[normalizeSlice(slice, flagField.shape)]
    flagField = flagField.swapaxes(0, 1)
    plt.imshow(flagField, interpolation='none', origin='lower',
               cmap=cmap, norm=norm)

    patches = [matplotlib.patches.Patch(color=color, label=name) for color, name in zip(colorValues, boundaryNames)]
    plt.axis('equal')
    if legend:
        plt.legend(handles=patches, bbox_to_anchor=(1.02, 0.5), loc=2, borderaxespad=0.)