# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from lbmpy.stencils import get_stencil

import numpy as np
import math
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

maxValue = 2
X = np.arange(-maxValue, maxValue, 0.1)
Y = np.arange(-maxValue, maxValue, 0.1)
X, Y = np.meshgrid(X, Y)


def maxwell_boltzmann(x, y):
    rho = 1
    m = 1
    k_B = 0.5
    pi = math.pi
    T = 1.2
    return rho * (m / (2 * k_B * T * pi)) ** (3 / 2) * np.exp(- m / (k_B * T) * (x ** 2 + y ** 2) / 2)


if __name__ == "__main__":
    from mpl_toolkits.mplot3d import Axes3D
    MB = maxwell_boltzmann(X, Y)

    surf = ax.plot_surface(X, Y, MB, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    for dir in get_stencil("D2Q9"):
        a = Arrow3D([0, dir[0]], [0, dir[1]], [0, 0], mutation_scale=20, lw=1, arrowstyle="-|>", color="b")
        ax.add_artist(a)

        h = maxwell_boltzmann(dir[0], dir[1])
        a = Arrow3D([dir[0], dir[0]], [dir[1], dir[1]], [0, h], mutation_scale=20, lw=2, arrowstyle="wedge", color="k")
        ax.add_artist(a)

    plt.show()
