from collections import defaultdict

import numpy as np
import sympy as sp

from lbmpy.phasefield.nphase_nestler import create_model


def get_parameters(num_phases, sigma, alpha=1):
    res_a = {}
    res_eps = {}
    for i in range(num_phases):
        for j in range(i):
            res_a[(i, j)] = alpha / (72 * sigma[(j, i)])
            res_eps[(i, j)] = sp.sqrt(alpha * sigma[(j, i)])
    return res_a, res_eps


def test_main():
    num_phases = 3
    alpha = 2
    sigma = defaultdict(lambda: 0.001)
    sigma[(1, 0)] = 0.005 / 4
    sigma[(0, 1)] = 0.005 / 4
    sigma[(0, 2)] = 0.0083 / 4
    sigma[(2, 0)] = 0.0083 / 4
    sigma[(1, 2)] = 0.01 / 4
    sigma[(2, 1)] = 0.01 / 4

    a_dict, epsilon_dict = get_parameters(num_phases, sigma, alpha)
    dh, init, run = create_model([50, 50], num_phases, a_dict,
                                 epsilon_dict, 0.0005, alpha, penalty_factor=0.0,
                                 simplex_projection=True)

    c_arr = dh.cpu_arrays['c']
    nx, ny = dh.shape

    c_arr[:, :int(0.5 * nx), 0] = 1
    c_arr[:, int(0.5 * nx):, 1] = 1

    c_arr[int(0.3 * nx):int(0.7 * nx), int(0.3 * ny):int(0.7 * ny), 0] = 0
    c_arr[int(0.3 * nx):int(0.7 * nx), int(0.3 * ny):int(0.7 * ny), 1] = 0
    c_arr[int(0.3 * nx):int(0.7 * nx), int(0.3 * ny):int(0.7 * ny), 2] = 1

    init()

    res = run(100)
    assert np.isfinite(np.max(res))
