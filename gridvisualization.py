import matplotlib.patches as patches


class Grid(object):
    """Visualizes a 2D LBM grid with matplotlib by drawing cells and pdf arrows"""

    def __init__(self, xCells, yCells):
        """Create a new grid with the given number of cells in x (horizontal) and y (vertical) direction"""

        self._xCells = xCells
        self._yCells = yCells

        self._patches = []
        for x in range(xCells):
            for y in range(yCells):
                self._patches.append(patches.Rectangle((x, y), 1.0, 1.0, fill=False, linewidth=3, color='#bbbbbb'))

        self._cellBoundaries = dict()  # mapping cell to rectangle patch
        self._arrows = dict()  # mapping (cell, direction) tuples to arrow patches

    def addCellBoundary(self, cell, **kwargs):
        """Draws a rectangle around a single cell. Keyword arguments are passed to the matplotlib Rectangle patch"""
        if 'fill' not in kwargs: kwargs['fill'] = False
        if 'linewidth' not in kwargs: kwargs['linewidth'] = 3
        if 'color' not in kwargs: kwargs['#bbbbbb']
        self._cellBoundaries[cell] = patches.Rectangle(cell, 1.0, 1.0, **kwargs)

    def addCellBoundaries(self, **kwargs):
        """Draws a rectangle around all cells. Keyword arguments are passed to the matplotlib Rectangle patch"""
        for x in range(self._xCells):
            for y in range(self._yCells):
                self.addCellBoundary((x, y), **kwargs)

    def addArrow(self, cell, arrowPosition, arrowDirection, **kwargs):
        """
        Draws an arrow in a cell. If an arrow exists already at this position, it is replaced.

        :param cell: cell coordinate as tuple (x,y)
        :param arrowPosition: each cell has 9 possible positions specified as tuple e.g. upper left (-1, 1)
        :param arrowDirection: direction of the arrow as (x,y) tuple
        :param kwargs: arguments passed directly to the FancyArrow patch of matplotlib
        """
        cellMidpoint = (0.5 + cell[0], 0.5 + cell[1])

        if 'width' not in kwargs: kwargs['width'] = 0.005
        if 'color' not in kwargs: kwargs['color'] = 'k'

        if arrowPosition == (0, 0):
            del kwargs['width']
            self._arrows[(cell, arrowPosition)] = patches.Circle(cellMidpoint, radius=0.03, **kwargs)
        else:
            arrowMidpoint = (cellMidpoint[0] + arrowPosition[0] * 0.25,
                             cellMidpoint[1] + arrowPosition[1] * 0.25)
            length = 0.75
            arrowStart = (arrowMidpoint[0] - arrowDirection[0] * 0.25 * length,
                          arrowMidpoint[1] - arrowDirection[1] * 0.25 * length)

            patch = patches.FancyArrow(arrowStart[0], arrowStart[1],
                                       0.25 * length * arrowDirection[0],
                                       0.25 * length * arrowDirection[1],
                                       **kwargs)
            self._arrows[(cell, arrowPosition)] = patch

    def fillWithDefaultArrows(self, **kwargs):
        """Fills the complete grid with the default pdf arrows"""
        for x in range(self._xCells):
            for y in range(self._yCells):
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 'color' not in kwargs: kwargs['color'] = '#bbbbbb'
                        if 'width' not in kwargs: kwargs['width'] = 0.006

                        self.addArrow((x, y), (dx, dy), (dx, dy), **kwargs)

    def draw(self, ax):
        """Draw the grid into a given matplotlib axes object"""

        for p in self._patches:
            ax.add_patch(p)

        for arrowPatch in self._arrows.values():
            ax.add_patch(arrowPatch)

        offset = 0.1
        ax.set_xlim(-offset, self._xCells+offset)
        ax.set_xlim(-offset, self._xCells + offset)
        ax.set_ylim(-offset, self._yCells + offset)
        ax.set_aspect('equal')
        ax.set_axis_off()
