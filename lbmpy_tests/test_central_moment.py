import numpy as np
import sympy as sp

from lbmpy.creationfunctions import create_lb_method
from lbmpy.forcemodels import Luo
from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.moments import get_default_moment_set_for_stencil, moment_matrix, set_up_shift_matrix
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.stencils import get_stencil

from lbmpy.methods.momentbased.moment_transforms import FastCentralMomentTransform


def test_central_moment_ldc():
    sc_central_moment = create_lid_driven_cavity((20, 20), method='central_moment',
                                                 relaxation_rates=[1.8, 1, 1], equilibrium_order=4,
                                                 compressible=True, force=(-1e-10, 0))

    sc_central_mometn_3D = create_lid_driven_cavity((20, 20, 3), method='central_moment',
                                                    relaxation_rates=[1.8, 1, 1, 1, 1], equilibrium_order=4,
                                                    compressible=True, force=(-1e-10, 0, 0))

    sc_central_moment.run(1000)
    sc_central_mometn_3D.run(1000)
    assert np.isfinite(np.max(sc_central_moment.velocity[:, :]))
    assert np.isfinite(np.max(sc_central_mometn_3D.velocity[:, :, :]))


def test_central_moment_class():
    stencil = get_stencil("D2Q9")

    method = create_lb_method(stencil=stencil, method='central_moment',
                              relaxation_rates=[1.2, 1.3, 1.4], equilibrium_order=4, compressible=True)

    srt = create_lb_method(stencil=stencil, method='srt',
                           relaxation_rate=1.2, equilibrium_order=4, compressible=True)

    rho = method.zeroth_order_equilibrium_moment_symbol
    u = method.first_order_equilibrium_moment_symbols
    cs_sq = sp.Rational(1, 3)

    force_model = Luo(force=sp.symbols(f"F_:{2}"))

    eq = (rho, 0, 0, rho * cs_sq, rho * cs_sq, 0, 0, 0, rho * cs_sq ** 2)

    default_moments = get_default_moment_set_for_stencil(stencil)

    assert method.central_moment_transform_class == FastCentralMomentTransform
    assert method.conserved_quantity_computation.zeroth_order_moment_symbol == rho
    assert method.conserved_quantity_computation.first_order_moment_symbols == u
    assert method.moment_equilibrium_values == eq

    assert method.force_model == None
    method.set_force_model(force_model)
    assert method.force_model == force_model

    assert method.relaxation_matrix[0, 0] == 0
    assert method.relaxation_matrix[1, 1] == 0
    assert method.relaxation_matrix[2, 2] == 0

    method.set_conserved_moments_relaxation_rate(1.9)

    assert method.relaxation_matrix[0, 0] == 1.9
    assert method.relaxation_matrix[1, 1] == 1.9
    assert method.relaxation_matrix[2, 2] == 1.9

    moments = list()
    for i in method.relaxation_info_dict:
        moments.append(i)

    assert moments == default_moments

    for i in range(len(stencil)):
        assert method.relaxation_rates[i] == method.relaxation_matrix[i, i]

    M = method.moment_matrix
    N = method.shift_matrix

    assert M == moment_matrix(moments, stencil=stencil)
    assert N == set_up_shift_matrix(moments, stencil=stencil)

    assert get_weights(stencil) == method.weights

    eq_terms_central = method.get_equilibrium_terms()
    eq_terms_srt = srt.get_equilibrium_terms()

    for i in range(len(stencil)):
        assert sp.simplify(eq_terms_central[i] - eq_terms_srt[i]) == 0

    method = create_lb_method(stencil="D2Q9", method='central_moment',
                              relaxation_rates=[1.7, 1.8, 1.2, 1.3, 1.4], equilibrium_order=4, compressible=True)

    assert method.relaxation_matrix[0, 0] == 1.7
    assert method.relaxation_matrix[1, 1] == 1.8
    assert method.relaxation_matrix[2, 2] == 1.8

    method = create_lb_method(stencil="D2Q9", method='central_moment',
                              relaxation_rates=[1.3] * 9, equilibrium_order=4, compressible=True)

    assert sum(method.relaxation_rates) == 1.3 * 9