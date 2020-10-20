"""Tests velocity and stress fluctuations for thermalized LB"""


import pystencils as ps
from lbmpy.creationfunctions import *
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
import numpy as np
from lbmpy.moments import is_bulk_moment, is_shear_moment, get_order


def single_component_maxwell(x1, x2, kT, mass):
    """Integrate the probability density from x1 to x2 using the trapezoidal rule"""
    x = np.linspace(x1, x2, 1000)
    return np.trapz(np.exp(-mass * x**2 / (2. * kT)), x) / np.sqrt(2. * np.pi * kT/mass)


def rr_getter(moment_group):
    """Maps a group of moments to a relaxation rate (shear, bulk, even, odd)
    in the 4 relaxation time thermalized LB model
    """
    is_shear = [is_shear_moment(m, 3) for m in moment_group]
    is_bulk = [is_bulk_moment(m, 3) for m in moment_group]
    order = [get_order(m) for m in moment_group]
    assert min(order) == max(order)
    order = order[0]

    if order < 2:
        return 0
    elif any(is_bulk):
        assert all(is_bulk)
        return sp.Symbol("omega_bulk")
    elif any(is_shear):
        assert all(is_shear)
        return sp.Symbol("omega_shear")
    elif order % 2 == 0:
        assert order > 2
        return sp.Symbol("omega_even")
    else:
        return sp.Symbol("omega_odd")


def second_order_moment_tensor_assignments(function_values, stencil, output_field):
    """Assignments for calculating the pressure tensor"""
    assert len(function_values) == len(stencil)
    dim = len(stencil[0])
    return [ps.Assignment(output_field(i, j),
                          sum(c[i] * c[j] * f for f, c in zip(function_values, stencil)))
            for i in range(dim) for j in range(dim)]


def add_pressure_output_to_collision_rule(collision_rule, pressure_field):
    pressure_ouput = second_order_moment_tensor_assignments(collision_rule.method.pre_collision_pdf_symbols,
                                                            collision_rule.method.stencil, pressure_field)
    collision_rule.main_assignments = collision_rule.main_assignments + pressure_ouput


def get_fluctuating_lb(size=None, kT=None, omega_shear=None, omega_bulk=None, omega_odd=None, omega_even=None, rho_0=None, target=None):

    # Parameters
    stencil = get_stencil('D3Q19')

    # Setup data handling
    dh = ps.create_data_handling(
        [size]*3, periodicity=True, default_target=target)
    src = dh.add_array('src', values_per_cell=len(stencil), layout='f')
    dst = dh.add_array_like('dst', 'src')
    rho = dh.add_array('rho', layout='f', latex_name='\\rho')
    u = dh.add_array('u', values_per_cell=dh.dim, layout='f')
    pressure_field = dh.add_array('pressure', values_per_cell=(
        3, 3), layout='f', gpu=target == 'gpu')
    force_field = dh.add_array(
        'force', values_per_cell=3, layout='f', gpu=target == 'gpu')

    # Method setup
    method = create_mrt_orthogonal(
        stencil=get_stencil('D3Q19'),
        compressible=True,
        weighted=True,
        relaxation_rate_getter=rr_getter,
        force_model=force_model_from_string('schiller', force_field.center_vector))
    collision_rule = create_lb_collision_rule(
        method,
        fluctuating={
            'temperature': kT
        },
        optimization={'cse_global': True}
    )

    add_pressure_output_to_collision_rule(collision_rule, pressure_field)

    collision = create_lb_update_rule(collision_rule=collision_rule,
                                      stencil=stencil,
                                      method=method,
                                      compressible=True,
                                      kernel_type='collide_only',
                                      optimization={'symbolic_field': src})
    stream = create_stream_pull_with_output_kernel(collision.method, src, dst,
                                                   {'density': rho, 'velocity': u})

    opts = {'cpu_openmp': True,
            'cpu_vectorize_info': None,
            'target': dh.default_target}

    # Compile kernels
    stream_kernel = ps.create_kernel(stream, **opts).compile()
    collision_kernel = ps.create_kernel(collision, **opts).compile()

    sync_pdfs = dh.synchronization_function([src.name])

    # Initialization
    init = macroscopic_values_setter(collision.method, velocity=(0,)*dh.dim,
                                     pdfs=src.center_vector, density=rho.center)
    init_kernel = ps.create_kernel(init, ghost_layers=0).compile()

    dh.fill(rho.name, rho_0)
    dh.fill(u.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
    dh.fill(u.name, 0)
    dh.fill(force_field.name, np.nan,
            ghost_layers=True, inner_ghost_layers=True)
    dh.fill(force_field.name, 0)
    dh.run_kernel(init_kernel)

    # time loop
    def time_loop(start, steps):
        dh.all_to_gpu()
        for i in range(start, start+steps):
            dh.run_kernel(collision_kernel, omega_shear=omega_shear, omega_bulk=omega_bulk,
                          omega_odd=omega_odd, omega_even=omega_even, seed=42, time_step=i)

            sync_pdfs()
            dh.run_kernel(stream_kernel)

            dh.swap(src.name, dst.name)
        return start+steps

    return dh, time_loop


def test_resting_fluid(target="cpu"):
    rho_0 = 0.86
    kT = 4E-4
    L = [60]*3
    dh, time_loop = get_fluctuating_lb(size=L[0], target=target,
                                       rho_0=rho_0, kT=kT,
                                       omega_shear=0.8, omega_bulk=0.5, omega_even=.04, omega_odd=0.3)

    # Test
    t = 0
    # warm up
    t = time_loop(t, 10)

    # Measurement
    for i in range(10):
        t = time_loop(t, 5)

        res_u = dh.gather_array("u").reshape((-1, 3))
        res_rho = dh.gather_array("rho").reshape((-1,))

        # mass conservation
        np.testing.assert_allclose(np.mean(res_rho), rho_0, atol=3E-12)

        # momentum conservation
        momentum = np.dot(res_rho, res_u)
        np.testing.assert_allclose(momentum, [0, 0, 0], atol=1E-10)

        # temperature
        kinetic_energy = 1/2*np.dot(res_rho, res_u*res_u)/np.product(L)
        np.testing.assert_allclose(
            kinetic_energy, [kT/2]*3, atol=kT*0.01)

        # velocity distribution
        v_hist, v_bins = np.histogram(
            res_u, bins=11, range=(-.075, .075), density=True)

        # Calculate expected values from single
        v_expected = []
        for j in range(len(v_hist)):
            # Maxwell distribution
            res = 1./(v_bins[j+1]-v_bins[j]) * \
                single_component_maxwell(
                    v_bins[j], v_bins[j+1], kT, rho_0)
            v_expected.append(res)
        v_expected = np.array(v_expected)

        # 10% accuracy on the entire histogram
        np.testing.assert_allclose(v_hist, v_expected, rtol=0.1)
        # 1% accuracy on the middle part
        remove = 3
        np.testing.assert_allclose(
            v_hist[remove:-remove], v_expected[remove:-remove], rtol=0.01)

        # pressure tensor against expressions from
        # Duenweg, Schiller, Ladd, https://arxiv.org/abs/0707.1581

        res_pressure = dh.gather_array("pressure").reshape((-1, 3, 3))

        c_s = np.sqrt(1/3)  # speed of sound

        # average of pressure tensor
        # Diagonal elements are rho c_s^22 +<u,u>. When the fluid is
        # thermalized, the expectation value of <u,u> = kT due to the
        # equi-partition theorem.
        p_av_expected = np.diag([rho_0*c_s**2 + kT]*3)
        np.testing.assert_allclose(
            np.mean(res_pressure, axis=0), p_av_expected, atol=c_s**2/2000)


def test_point_force(target="cpu"):
    """Test momentum balance for thermalized fluid with applied poitn forces"""
    rho_0 = 0.86
    kT = 4E-4
    L = [8]*3
    dh, time_loop = get_fluctuating_lb(size=L[0], target=target,
                                       rho_0=rho_0, kT=kT,
                                       omega_shear=0.8, omega_bulk=0.5, omega_even=.04, omega_odd=0.3)

    # Test
    t = 0
    # warm up
    t = time_loop(t, 100)

    introduced_momentum = np.zeros(3)
    for i in range(100):
        point_force = 1E-5*(np.random.random(3) - .5)
        introduced_momentum += point_force

        # Note that ghost layers are included in the indexing
        force_pos = np.random.randint(1, L[0]-2, size=3)

        dh.cpu_arrays["force"][force_pos[0],
                               force_pos[1], force_pos[2]] = point_force
        t = time_loop(t, 1)
        res_u = dh.gather_array("u").reshape((-1, 3))
        res_rho = dh.gather_array("rho").reshape((-1,))

        # mass conservation
        np.testing.assert_allclose(np.mean(res_rho), rho_0, atol=3E-12)

        # momentum conservation
        momentum = np.dot(res_rho, res_u)
        np.testing.assert_allclose(
            momentum, introduced_momentum + 0.5 * point_force, atol=1E-10)
        dh.cpu_arrays["force"][force_pos[0],
                               force_pos[1], force_pos[2]] = np.zeros(3)
