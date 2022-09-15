import numpy as np
import sympy as sp

from lbmpy.creationfunctions import create_lb_method, LBMConfig
from lbmpy.enums import Method, Stencil
from lbmpy.forcemodels import Luo
from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.moments import moment_matrix, set_up_shift_matrix
from lbmpy.methods.creationfunctions import cascaded_moment_sets_literature
from lbmpy.scenarios import create_lid_driven_cavity
from lbmpy.stencils import LBStencil

from lbmpy.moment_transforms import BinomialChimeraTransform


def test_central_moment_ldc():
    sc_central_moment = create_lid_driven_cavity((20, 20), method=Method.CENTRAL_MOMENT,
                                                 relaxation_rate=1.8, equilibrium_order=4,
                                                 compressible=True, force=(-1e-10, 0))

    sc_central_moment_3d = create_lid_driven_cavity((20, 20, 3), method=Method.CENTRAL_MOMENT,
                                                    relaxation_rate=1.8, equilibrium_order=4,
                                                    compressible=True, force=(-1e-10, 0, 0))

    sc_central_moment.run(1000)
    sc_central_moment_3d.run(1000)
    assert np.isfinite(np.max(sc_central_moment.velocity[:, :]))
    assert np.isfinite(np.max(sc_central_moment_3d.velocity[:, :, :]))


def test_central_moment_class():
    stencil = LBStencil(Stencil.D2Q9)
    lbm_config = LBMConfig(stencil=stencil, method=Method.CENTRAL_MOMENT, relaxation_rates=[1.2, 1.3, 1.4, 1.5],
                           equilibrium_order=4, compressible=True, zero_centered=True)

    method = create_lb_method(lbm_config=lbm_config)

    lbm_config = LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.2,
                           equilibrium_order=4, compressible=True, zero_centered=True)
    srt = create_lb_method(lbm_config=lbm_config)

    rho = method.zeroth_order_equilibrium_moment_symbol
    u = method.first_order_equilibrium_moment_symbols
    cs_sq = sp.Rational(1, 3)

    force_model = Luo(force=sp.symbols(f"F_:{2}"))

    eq = (rho, 0, 0, 0, 0, 2 * rho * cs_sq, 0, 0, rho * cs_sq ** 2)

    default_moments = cascaded_moment_sets_literature(stencil)
    default_moments = [item for sublist in default_moments for item in sublist]

    assert method.central_moment_transform_class == BinomialChimeraTransform
    assert method.conserved_quantity_computation.density_symbol == rho
    assert method.conserved_quantity_computation.velocity_symbols == u
    assert method.moment_equilibrium_values == eq

    assert method.force_model is None
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

    cqc = method.conserved_quantity_computation
    subs = {cqc.density_deviation_symbol : cqc.density_symbol - cqc.background_density}

    eq_terms_central = method.get_equilibrium_terms()
    eq_terms_srt = srt.get_equilibrium_terms()

    assert (eq_terms_central - eq_terms_srt).subs(subs).expand() == sp.Matrix((0,) * stencil.Q)

    method = create_lb_method(lbm_config=LBMConfig(stencil=LBStencil(Stencil.D2Q9), method=Method.CENTRAL_MOMENT,
                                                   relaxation_rates=[1.7, 1.8, 1.2, 1.3, 1.4], equilibrium_order=4,
                                                   compressible=True))

    assert method.relaxation_matrix[0, 0] == 0
    assert method.relaxation_matrix[1, 1] == 0
    assert method.relaxation_matrix[2, 2] == 0

    method = create_lb_method(lbm_config=LBMConfig(stencil=LBStencil(Stencil.D2Q9), method=Method.CENTRAL_MOMENT,
                                                   relaxation_rates=[1.3] * 9, equilibrium_order=4,
                                                   compressible=True))

    np.testing.assert_almost_equal(sum(method.relaxation_rates), 1.3 * 9)
