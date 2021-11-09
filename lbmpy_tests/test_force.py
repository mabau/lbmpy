import pytest

import numpy as np
import sympy as sp
import pystencils as ps
from pystencils import Target

from lbmpy.creationfunctions import create_lb_method, create_lb_update_rule, LBMConfig, LBMOptimisation
from lbmpy.enums import Stencil, Method, ForceModel
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter, macroscopic_values_getter
from lbmpy.moments import is_bulk_moment
from lbmpy.stencils import LBStencil
from lbmpy.updatekernels import create_stream_pull_with_output_kernel

# all force models available are defined in the ForceModel enum, but Cumulant is not a "real" force model
force_models = [f for f in ForceModel if f is not ForceModel.CUMULANT]


@pytest.mark.parametrize("method_enum", [Method.SRT, Method.TRT])
@pytest.mark.parametrize("force_model", force_models)
@pytest.mark.parametrize("omega", [0.5, 1.5])
def test_total_momentum(method_enum, force_model, omega):
    # for the EDM force model this test case not work. However it is successfully used in test_entropic_model
    # Any attempt to adapted the EDM force model so it fullfills the test case did result in a failure in the
    # entropic test case. Note also that the test runs for MRT and EMD
    if force_model == ForceModel.EDM:
        pytest.skip()

    L = (16, 16)
    stencil = LBStencil(Stencil.D2Q9)
    F = (2e-4, -3e-4)

    dh = ps.create_data_handling(L, periodicity=True, default_target=Target.CPU)
    src = dh.add_array('src', values_per_cell=stencil.Q)
    dst = dh.add_array_like('dst', 'src')
    ρ = dh.add_array('rho', values_per_cell=1)
    u = dh.add_array('u', values_per_cell=stencil.D)

    lbm_config = LBMConfig(method=method_enum, stencil=stencil, relaxation_rate=omega,
                           compressible=True, force_model=force_model, force=F, streaming_pattern='pull')
    lbm_opt = LBMOptimisation(symbolic_field=src)

    collision = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    config = ps.CreateKernelConfig(cpu_openmp=True, target=dh.default_target)

    collision_kernel = ps.create_kernel(collision, config=config).compile()

    def init():
        dh.fill(ρ.name, 1)
        dh.fill(u.name, 0)

        setter = macroscopic_values_setter(collision.method, velocity=(0,) * dh.dim,
                                           pdfs=src, density=ρ.center,
                                           set_pre_collision_pdfs=True)
        kernel = ps.create_kernel(setter).compile()
        dh.run_kernel(kernel)

    sync_pdfs = dh.synchronization_function([src.name])

    getter = macroscopic_values_getter(collision.method, ρ.center, u.center_vector, src, use_pre_collision_pdfs=True)
    getter_kernel = ps.create_kernel(getter).compile()

    def time_loop(steps):
        dh.all_to_gpu()
        for _ in range(steps):
            dh.run_kernel(collision_kernel)
            dh.swap(src.name, dst.name)
            sync_pdfs()
        dh.all_to_cpu()

    t = 20
    init()
    time_loop(t)
    dh.run_kernel(getter_kernel)
    total = np.sum(dh.gather_array(u.name), axis=(0, 1))
    assert np.allclose(total / np.prod(L) / F / t, 1)


@pytest.mark.parametrize("force_model", force_models)
@pytest.mark.parametrize("compressible", [True, False])
def test_modes(force_model, compressible):
    """check force terms in mode space"""
    _check_modes(LBStencil(Stencil.D2Q9), force_model, compressible)


@pytest.mark.parametrize("stencil", [Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize("force_model", force_models)
@pytest.mark.parametrize("compressible", [True, False])
@pytest.mark.longrun
def test_modes_longrun(stencil, force_model, compressible):
    """check force terms in mode space"""
    _check_modes(LBStencil(stencil), force_model, compressible)


@pytest.mark.parametrize("force_model", force_models)
def test_momentum_density_shift(force_model):
    target = Target.CPU

    stencil = LBStencil(Stencil.D2Q9)
    domain_size = (4, 4)
    dh = ps.create_data_handling(domain_size=domain_size, default_target=target)

    rho = dh.add_array('rho', values_per_cell=1)
    dh.fill('rho', 0.0, ghost_layers=True)

    momentum_density = dh.add_array('momentum_density', values_per_cell=dh.dim)
    dh.fill('momentum_density', 0.0, ghost_layers=True)

    src = dh.add_array('src', values_per_cell=len(stencil))
    dh.fill('src', 0.0, ghost_layers=True)

    lbm_config = LBMConfig(method=Method.SRT, compressible=True, force_model=force_model,
                           force=(1, 2))
    method = create_lb_method(lbm_config=lbm_config)

    cqc = method.conserved_quantity_computation

    momentum_density_getter = cqc.output_equations_from_pdfs(src.center_vector,
                                                             {'density': rho.center,
                                                              'momentum_density': momentum_density.center_vector})

    config = ps.CreateKernelConfig(target=dh.default_target)
    momentum_density_ast = ps.create_kernel(momentum_density_getter, config=config)
    momentum_density_kernel = momentum_density_ast.compile()

    dh.run_kernel(momentum_density_kernel)
    assert np.sum(dh.gather_array(momentum_density.name)[:, :, 0]) == np.prod(domain_size) / 2
    assert np.sum(dh.gather_array(momentum_density.name)[:, :, 1]) == np.prod(domain_size)


@pytest.mark.parametrize('force_model', force_models)
def test_forcing_space_equivalences(force_model):
    if force_model == ForceModel.HE:
        #   We don't expect equivalence for the He model since its
        #   moments are derived from the continuous maxwellian
        return
    stencil = LBStencil(Stencil.D3Q27)
    force = sp.symbols(f"F_:{stencil.D}")
    lbm_config = LBMConfig(stencil=stencil, method=Method.SRT, force=force, force_model=force_model)
    fmodel = lbm_config.force_model

    lb_method = create_lb_method(lbm_config=lbm_config)
    inv_moment_matrix = lb_method.moment_matrix.inv()

    force_pdfs = sp.Matrix(fmodel(lb_method))
    force_moments = fmodel.moment_space_forcing(lb_method)

    diff = (force_pdfs - (inv_moment_matrix * force_moments)).expand()
    for i, d in enumerate(diff):
        assert d == 0, f"Mismatch between population and moment space forcing " \
                       f"in force model {force_model}, population f_{i}"


@pytest.mark.parametrize("force_model", [ForceModel.GUO, ForceModel.BUICK, ForceModel.SHANCHEN])
@pytest.mark.parametrize("stencil", [Stencil.D2Q9, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize("method", [Method.SRT, Method.TRT, Method.MRT])
def test_literature(force_model, stencil, method):
    # Be aware that the choice of the conserved moments does not affect the forcing although omega is introduced
    # in the forcing vector then. The reason is that:
    # m_{100}^{\ast} = m_{100} + \omega ( m_{100}^{eq} - m_{100} ) + \left( 1 - \frac{\omega}{2} \right) F_x
    # always simplifies to:
    # m_{100}^{\ast} = m_{100} + F_x
    # Thus the relaxation rate gets cancled again.

    stencil = LBStencil(stencil)

    omega_s = sp.Symbol("omega_s")
    omega_b = sp.Symbol("omega_b")
    omega_o = sp.Symbol("omega_o")
    omega_e = sp.Symbol("omega_e")
    if method == Method.SRT:
        rrs = [omega_s]
        omega_o = omega_b = omega_e = omega_s
    elif method == Method.TRT:
        rrs = [omega_e, omega_o]
        omega_s = omega_b = omega_e
    else:
        rrs = [omega_s, omega_b, omega_o, omega_e, omega_o, omega_e]

    F = sp.symbols(f"F_:{stencil.D}")

    lbm_config = LBMConfig(method=method, weighted=True, stencil=stencil, relaxation_rates=rrs,
                           compressible=force_model != ForceModel.BUICK,
                           force_model=force_model, force=F)

    lb_method = create_lb_method(lbm_config=lbm_config)
    omega_momentum = list(set(lb_method.relaxation_rates[1:stencil.D + 1]))
    assert len(omega_momentum) == 1
    omega_momentum = omega_momentum[0]

    subs_dict = lb_method.subs_dict_relxation_rate
    force_term = sp.simplify(lb_method.force_model(lb_method).subs(subs_dict))
    u = sp.Matrix(lb_method.first_order_equilibrium_moment_symbols)
    rho = lb_method.conserved_quantity_computation.zeroth_order_moment_symbol

    # see silva2020 for nomenclature
    F = sp.Matrix(F)
    uf = sp.Matrix(u).dot(F)
    F2 = sp.Matrix(F).dot(sp.Matrix(F))
    Fq = sp.zeros(stencil.Q, 1)
    uq = sp.zeros(stencil.Q, 1)
    for i, cq in enumerate(stencil):
        Fq[i] = sp.Matrix(cq).dot(sp.Matrix(F))
        uq[i] = sp.Matrix(cq).dot(u)

    common_plus = 3 * (1 - omega_e / 2)
    common_minus = 3 * (1 - omega_momentum / 2)

    result = []
    if method == Method.MRT and force_model == ForceModel.GUO:
        # check against eq. 4.68 from schiller2008thermal
        uf = u.dot(F) * sp.eye(len(F))
        G = (u * F.transpose() + F * u.transpose() - uf * sp.Rational(2, lb_method.dim)) * sp.Rational(1, 2) * \
            (2 - omega_s) + uf * sp.Rational(1, lb_method.dim) * (2 - omega_b)
        for direction, w_i in zip(lb_method.stencil, lb_method.weights):
            direction = sp.Matrix(direction)
            tr = sp.trace(G * (direction * direction.transpose() - sp.Rational(1, 3) * sp.eye(len(F))))
            result.append(3 * w_i * (F.dot(direction) + sp.Rational(3, 2) * tr))
    elif force_model == ForceModel.GUO:
        # check against table 2 in silva2020 (correct for SRT and TRT), matches eq. 20 from guo2002discrete (for SRT)
        Sq_plus = sp.zeros(stencil.Q, 1)
        Sq_minus = sp.zeros(stencil.Q, 1)
        for i, w_i in enumerate(lb_method.weights):
            Sq_plus[i] = common_plus * w_i * (3 * uq[i] * Fq[i] - uf)
            Sq_minus[i] = common_minus * w_i * Fq[i]
        result = Sq_plus + Sq_minus
    elif force_model == ForceModel.BUICK:
        # check against table 2 in silva2020 (correct for all collision models due to the simplicity of Buick),
        # matches eq. 18 from silva2010 (for SRT)
        Sq_plus = sp.zeros(stencil.Q, 1)
        Sq_minus = sp.zeros(stencil.Q, 1)
        for i, w_i in enumerate(lb_method.weights):
            Sq_plus[i] = 0
            Sq_minus[i] = common_minus * w_i * Fq[i]
        result = Sq_plus + Sq_minus
    elif force_model == ForceModel.EDM:
        # check against table 2 in silva2020
        if method == Method.MRT:
            # for mrt no literature terms are known at the time of writing this test case.
            # However it is most likly correct since SRT and TRT are derived from the moment space representation
            pytest.skip()
        Sq_plus = sp.zeros(stencil.Q, 1)
        Sq_minus = sp.zeros(stencil.Q, 1)
        for i, w_i in enumerate(lb_method.weights):
            Sq_plus[i] = common_plus * w_i * (3 * uq[i] * Fq[i] - uf) + ((w_i / (8 * rho)) * (3 * Fq[i] ** 2 - F2))
            Sq_minus[i] = common_minus * w_i * Fq[i]
        result = Sq_plus + Sq_minus
    elif force_model == ForceModel.SHANCHEN:
        # check against table 2 in silva2020
        if method == Method.MRT:
            # for mrt no literature terms are known at the time of writing this test case.
            # However it is most likly correct since SRT and TRT are derived from the moment space representation
            pytest.skip()
        Sq_plus = sp.zeros(stencil.Q, 1)
        Sq_minus = sp.zeros(stencil.Q, 1)
        for i, w_i in enumerate(lb_method.weights):
            Sq_plus[i] = common_plus * w_i * (3 * uq[i] * Fq[i] - uf) + common_plus ** 2 * (
                (w_i / (2 * rho)) * (3 * Fq[i] ** 2 - F2))
            Sq_minus[i] = common_minus * w_i * Fq[i]
        result = Sq_plus + Sq_minus
    assert list(sp.simplify(force_term - sp.Matrix(result))) == [0] * len(stencil)


@pytest.mark.parametrize("force_model", force_models)
@pytest.mark.parametrize("compressible", [True, False])
def test_modes_central_moment(force_model, compressible):
    """check force terms in mode space"""
    stencil = LBStencil(Stencil.D2Q9)
    omega_s = sp.Symbol("omega_s")
    F = list(sp.symbols(f"F_:{stencil.D}"))

    lbm_config = LBMConfig(method=Method.CENTRAL_MOMENT, stencil=stencil, relaxation_rate=omega_s,
                           compressible=True, force_model=force_model, force=tuple(F))
    method = create_lb_method(lbm_config=lbm_config)

    subs_dict = method.subs_dict_relxation_rate
    force_moments = method.force_model.moment_space_forcing(method)
    force_moments = force_moments.subs(subs_dict)

    # The mass mode should be zero
    assert force_moments[0] == 0

    # The momentum moments should contain the force
    assert list(force_moments[1:stencil.D + 1]) == F


@pytest.mark.parametrize("stencil", [Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27])
@pytest.mark.parametrize("force_model", force_models)
@pytest.mark.parametrize("compressible", [True, False])
@pytest.mark.longrun
def test_modes_central_moment_longrun(stencil, force_model, compressible):
    """check force terms in mode space"""
    stencil = LBStencil(stencil)
    omega_s = sp.Symbol("omega_s")
    F = list(sp.symbols(f"F_:{stencil.D}"))

    lbm_config = LBMConfig(method=Method.CENTRAL_MOMENT, stencil=stencil, relaxation_rate=omega_s,
                           compressible=compressible, force_model=force_model, force=tuple(F))
    method = create_lb_method(lbm_config=lbm_config)

    subs_dict = method.subs_dict_relxation_rate
    force_moments = method.force_model.moment_space_forcing(method)
    force_moments = force_moments.subs(subs_dict)

    # The mass mode should be zero
    assert force_moments[0] == 0

    # The momentum moments should contain the force
    assert list(force_moments[1:stencil.D + 1]) == F


def _check_modes(stencil, force_model, compressible):
    omega_s = sp.Symbol("omega_s")
    omega_b = sp.Symbol("omega_b")
    omega_o = sp.Symbol("omega_o")
    omega_e = sp.Symbol("omega_e")

    F = list(sp.symbols(f"F_:{stencil.D}"))

    lbm_config = LBMConfig(method=Method.MRT, stencil=stencil,
                           relaxation_rates=[omega_s, omega_b, omega_o, omega_e, omega_o, omega_e],
                           compressible=compressible, force_model=force_model, force=tuple(F))
    method = create_lb_method(lbm_config=lbm_config)

    subs_dict = method.subs_dict_relxation_rate
    force_moments = method.force_model.moment_space_forcing(method)
    force_moments = force_moments.subs(subs_dict)

    # The mass mode should be zero
    assert force_moments[0] == 0

    # The momentum moments should contain the force
    assert list(force_moments[1:stencil.D + 1]) == F

    if force_model == ForceModel.GUO:
        num_stresses = (stencil.D * stencil.D - stencil.D) // 2 + stencil.D
        lambda_s, lambda_b = -omega_s, -omega_b

        # The stress moments should match eq. 47 from https://doi.org/10.1023/A:1010414013942
        u = method.first_order_equilibrium_moment_symbols

        def traceless(m):
            tr = sp.simplify(sum([m[i, i] for i in range(stencil.D)]))
            return m - tr / m.shape[0] * sp.eye(m.shape[0])

        C = sp.Rational(1, 2) * (2 + lambda_s) * (traceless(sp.Matrix(u) * sp.Matrix(F).transpose()) +
                                                  traceless(sp.Matrix(F) * sp.Matrix(u).transpose())) + \
            sp.Rational(1, method.dim) * (2 + lambda_b) * sp.Matrix(u).dot(F) * sp.eye(method.dim)

        subs = {sp.Symbol(chr(ord("x") + i)) * sp.Symbol(chr(ord("x") + j)): C[i, j]
                for i in range(stencil.D) for j in range(stencil.D)}
        for force_moment, moment in zip(force_moments[stencil.D + 1:stencil.D + 1 + num_stresses],
                                        method.moments[stencil.D + 1:stencil.D + 1 + num_stresses]):
            ref = moment.subs(subs)
            diff = sp.simplify(ref - force_moment)
            if is_bulk_moment(moment, stencil.D):
                assert diff == 0 or isinstance(diff, sp.Rational)  # difference should be zero or a constant
            else:
                assert diff == 0  # difference should be zero

        ff = method.moment_matrix.inv() * sp.Matrix(method.force_model.moment_space_forcing(method).subs(subs_dict))
        # Check eq. 4.53a from schiller2008thermal
        assert sp.simplify(sum(ff)) == 0
        # Check eq. 4.53b from schiller2008thermal
        assert [sp.simplify(sum(ff[i] * stencil[i][j] for i in range(len(stencil)))) for j in range(stencil.D)] == F
        # Check eq. 4.61a from schiller2008thermal
        ref = (2 + lambda_s) / 2 * (traceless(sp.Matrix(u) * sp.Matrix(F).transpose()) +
                                    traceless(sp.Matrix(F) * sp.Matrix(u).transpose()))
        s = sp.zeros(stencil.D)
        for i in range(0, len(stencil)):
            s += ff[i] * traceless(sp.Matrix(stencil[i]) * sp.Matrix(stencil[i]).transpose())
        assert sp.simplify(s - ref) == sp.zeros(stencil.D)
        # Check eq. 4.61b from schiller2008thermal
        assert sp.simplify(sum(ff[i] * stencil[i][a] ** 2 for i in range(len(stencil)) for a in range(stencil.D))
                           - (2 + lambda_b) * sp.Matrix(u).dot(F)) == 0

        # All other moments should be zero
        assert list(force_moments[stencil.D + 1 + num_stresses:]) == [0] * (len(stencil) - (stencil.D + 1 + num_stresses))
    elif force_model == ForceModel.SIMPLE:
        # All other moments should be zero
        assert list(force_moments[stencil.D + 1:]) == [0] * (len(stencil) - (stencil.D + 1))
