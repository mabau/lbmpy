import sympy as sp


def getStencil(name, ordering='walberla'):
    """
    Stencils are tuples of discrete velocities. They are commonly labeled in the 'DxQy' notation, where d is the
    dimension (length of the velocity tuples) and y is number of discrete velocities.

    :param name: DxQy notation
    :param ordering: the LBM literature does not use a common order of the discrete velocities, therefore here
                     different common orderings are available. All orderings lead to the same method, it just has
                     to be used consistently. Here more orderings are available to compare intermediate results with
                     the literature.
    """
    try:
        return getStencil.data[ordering.lower()][name.upper()]
    except KeyError:
        errMsg = ""
        for orderingName, stencils in getStencil.data.items():
            errMsg += "  %s: %s\n" % (orderingName, ", ".join(stencils.keys()))

        raise ValueError("No such stencil available. Available stencils: <orderingName>( <stencilNames> )\n" + errMsg)
getStencil.data = {

    'walberla': {
        'D2Q9': ((0, 0),
                 (0, 1), (0, -1), (-1, 0), (1, 0),
                 (-1, 1), (1, 1), (-1, -1), (1, -1),),
        'D3Q15': ((0, 0, 0),
                  (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
                  (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)),
        'D3Q19': ((0, 0, 0),
                  (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
                  (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0),
                  (0, 1, 1), (0, -1, 1), (-1, 0, 1), (1, 0, 1),
                  (0, 1, -1), (0, -1, -1), (-1, 0, -1), (1, 0, -1)),
        'D3Q27': ((0, 0, 0),
                  (0, 1, 0), (0, -1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1),
                  (-1, 1, 0), (1, 1, 0), (-1, -1, 0), (1, -1, 0),
                  (0, 1, 1), (0, -1, 1), (-1, 0, 1), (1, 0, 1),
                  (0, 1, -1), (0, -1, -1), (-1, 0, -1), (1, 0, -1),
                  (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1), (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1))
    },

    'standard': {
        'D2Q9': ((0, 0),
                 (1, 0), (0, 1), (-1, 0), (0, -1),
                 (1, 1), (-1, 1), (-1, -1), (1, -1)),
    },

    'braunschweig': {
        'D2Q9': ((0, 0),
                 (-1, 1), (-1, 0), (-1, -1), (0, -1),
                 (1, -1), (1, 0), (1, 1), (0, 1)),
        'D3Q19': ((0, 0, 0),

                  (1, 0, 0), (-1, 0, 0),
                  (0, 1, 0), (0, -1, 0),
                  (0, 0, 1), (0, 0, -1),

                  (1, 1, 0), (-1, -1, 0),
                  (1, -1, 0), (-1, 1, 0),
                  (1, 0, 1), (-1, 0, -1),
                  (1, 0, -1), (-1, 0, 1),
                  (0, 1, 1), (0, -1, -1),
                  (0, 1, -1), (0, -1, 1)),
    },
    'premnath': {
        'D3Q15': ((0, 0, 0),
                  (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                  (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
                  (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)),
        'D3Q19': ((0, 0, 0),
                  (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                  (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                  (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                  (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)),
        'D3Q27': ((0, 0, 0),
                  (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                  (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                  (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                  (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
                  (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
                  (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)),
    }
}


def inverseDirection(direction):
    """Returns inverse i.e. negative of given direction tuple"""
    return tuple([-i for i in direction])


def isValidStencil(stencil, maxNeighborhood=None):
    """
    Tests if a nested sequence is a valid stencil i.e. all the inner sequences have the same length.
    If maxNeighborhood is specified, it is also verified that the stencil does not contain any direction components
    with absolute value greater than the maximal neighborhood.
    """
    expectedDim = len(stencil[0])
    for d in stencil:
        if len(d) != expectedDim:
            return False
        if maxNeighborhood is not None:
            for d_i in d:
                if abs(d_i) > maxNeighborhood:
                    return False
    return True


def isSymmetricStencil(stencil):
    """Tests for every direction d, that -d is also in the stencil"""
    for d in stencil:
        if inverseDirection(d) not in stencil:
            return False
    return True


def stencilsHaveSameEntries(s1, s2):
    if len(s1) != len(s2):
        return False
    return len(set(s1) - set(s2)) == 0


# -------------------------------------- Visualization -----------------------------------------------------------------

def visualizeStencil(stencil, **kwargs):
    dim = len(stencil[0])
    if dim == 2:
        visualizeStencil2D(stencil, **kwargs)
    else:
        slicing = False
        if 'slice' in kwargs:
            slicing = kwargs['slice']
            del kwargs['slice']

        if slicing:
            visualizeStencil3DBySlicing(stencil, **kwargs)
        else:
            visualizeStencil3D(stencil, **kwargs)


def visualizeStencil2D(stencil, axes=None, data=None, textsize='12', **kwargs):
    """
    Creates a matplotlib 2D plot of the stencil

    :param stencil: sequence of directions
    :param axes: optional matplotlib axes
    :param data: data to annotate the directions with, if none given, the indices are used
    :param textsize: size of annotation text
    """
    from matplotlib.patches import BoxStyle
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gca()

    text_box_style = BoxStyle("Round", pad=0.3)
    head_length = 0.1
    text_offset = 1.25

    if data is None:
        data = list(range(len(stencil)))

    for dir, annotation in zip(stencil, data):
        assert len(dir) == 2, "Works only for 2D stencils"

        if not(dir[0] == 0 and dir[1] == 0):
            axes.arrow(0, 0, dir[0], dir[1], head_width=0.08, head_length=head_length, color='k')

        if isinstance(annotation, sp.Basic):
            annotation = "$" + sp.latex(annotation) + "$"
        else:
            annotation = str(annotation)
        axes.text(dir[0]*text_offset, dir[1]*text_offset, annotation, verticalalignment='center', zorder=30,
                  horizontalalignment='center',
                  size=textsize, bbox=dict(boxstyle=text_box_style, facecolor='#00b6eb', alpha=0.85, linewidth=0))

    axes.set_axis_off()
    axes.set_aspect('equal')
    axes.set_xlim([-text_offset*1.1, text_offset*1.1])
    axes.set_ylim([-text_offset * 1.1, text_offset * 1.1])


def visualizeStencil3DBySlicing(stencil, sliceAxis=2, data=None, **kwargs):
    """
    Visualizes a 3D, first-neighborhood stencil by plotting 3 slices along a given axis

    :param stencil: stencil as sequence of directions
    :param sliceAxis: 0, 1, or 2 indicating the axis to slice through
    :param data: optional data to print as text besides the arrows
    :return:
    """
    import matplotlib.pyplot as plt

    for dir in stencil:
        for element in dir:
            assert element == -1 or element == 0 or element == 1, "This function can only first neighborhood stencils"

    f, axes = plt.subplots(1, 3)
    splittedDirections = [[], [], []]
    splittedData = [[], [], []]
    axesNames = ['x', 'y', 'z']

    for i, dir in enumerate(stencil):
        splitIdx = dir[sliceAxis] + 1
        reducedDir = tuple([element for j, element in enumerate(dir) if j != sliceAxis])
        splittedDirections[splitIdx].append(reducedDir)
        splittedData[splitIdx].append(i if data is None else data[i])

    for i in range(3):
        visualizeStencil2D(splittedDirections[i], axes[i], splittedData[i], **kwargs)
    for i in [-1, 0, 1]:
        axes[i+1].set_title("Cut at %s=%d" % (axesNames[sliceAxis], i))


def visualizeStencil3D(stencil, axes=None, data=None, textsize='8'):
    """
    Draws 3D stencil into a 3D coordinate system, parameters are similar to :func:`visualizeStencil2D`
    If data is None, no labels are drawn. To draw the labels as in the 2D case, use ``data=list(range(len(stencil)))``
    """
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.pyplot as plt
    from matplotlib.patches import BoxStyle
    from itertools import product, combinations
    import numpy as np

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    if axes is None:
        axes = plt.figure().gca(projection='3d')
        axes.set_aspect("equal")

    if data is None:
        data = [None] * len(stencil)

    text_offset = 1.25
    text_box_style = BoxStyle("Round", pad=0.3)

    # Draw cell (cube)
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == r[1] - r[0]:
            axes.plot3D(*zip(s, e), color="k", alpha=0.5)

    for dir, annotation in zip(stencil, data):
        assert len(dir) == 3, "Works only for 3D stencils"
        if not(dir[0] == 0 and dir[1] == 0 and dir[2] == 0):
            if dir[0] == 0:
                color = '#348abd'
            elif dir[1] == 0:
                color = '#fac364'
            elif sum([abs(d) for d in dir]) == 2:
                color = '#95bd50'
            else:
                color = '#808080'

            a = Arrow3D([0, dir[0]], [0, dir[1]], [0, dir[2]], mutation_scale=20, lw=2, arrowstyle="-|>", color=color)
            axes.add_artist(a)

        if annotation:
            if isinstance(annotation, sp.Basic):
                annotation = "$" + sp.latex(annotation) + "$"
            else:
                annotation = str(annotation)

            axes.text(dir[0]*text_offset, dir[1]*text_offset, dir[2]*text_offset,
                      annotation, verticalalignment='center', zorder=30,
                      size=textsize, bbox=dict(boxstyle=text_box_style, facecolor='#777777', alpha=0.6, linewidth=0))

    axes.set_xlim([-text_offset*1.1, text_offset*1.1])
    axes.set_ylim([-text_offset * 1.1, text_offset * 1.1])
    axes.set_zlim([-text_offset * 1.1, text_offset * 1.1])
    axes.set_axis_off()

