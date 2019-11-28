"""
This tests that for the thermalized LB (MRT with 15 equal relaxation times),
the correct temperature is obtained and the velocity distribution matches
the Maxwell-Boltzmann distribution
"""


import pystencils as ps
from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.creationfunctions import create_lb_collision_rule
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity, relaxation_rate_from_magic_number
import numpy as np
import pickle
import gzip
from time import time


def single_component_maxwell(x1, x2, kT):
    """Integrate the probability density from x1 to x2 using the trapezoidal rule"""
    x = np.linspace(x1, x2, 1000)
    return np.trapz(np.exp(-x**2 / (2. * kT)), x) / np.sqrt(2. * np.pi * kT)


def run_scenario(scenario, steps):
    scenario.pre_run()
    for t in range(scenario.time_steps_run, scenario.time_steps_run + steps):
        scenario.kernel_params['time_step'] = t
        scenario.time_step()
    scenario.post_run()
    scenario.time_steps_run += steps


def create_scenario(domain_size, temperature=None, viscosity=None, seed=2, target='cpu', openmp=4, num_rel_rates=None):
    rr = [relaxation_rate_from_lattice_viscosity(viscosity)]
    rr = rr*num_rel_rates
    cr = create_lb_collision_rule(
        stencil='D3Q19', compressible=True,
        method='mrt', weighted=True, relaxation_rates=rr,
        fluctuating={'temperature': temperature, 'seed': seed},
        optimization={'cse_global': True, 'split': False,
                      'cse_pdfs': True, 'vectorization': True}
    )
    return LatticeBoltzmannStep(periodicity=(True, True, True), domain_size=domain_size, compressible=True, stencil='D3Q19', collision_rule=cr, optimization={'target': target, 'openmp': openmp})


def test_fluctuating_mrt():
    # Unit conversions (MD to lattice) for parameters known to work with Espresso
    agrid = 1.
    m = 1.  # mass per node
    tau = 0.01  # time step
    temperature = 4. / (m * agrid**2/tau**2)
    viscosity = 3. * tau / agrid**2
    n = 8
    sc = create_scenario((n, n, n), viscosity=viscosity, temperature=temperature,
                         target='cpu', openmp=4, num_rel_rates=15)
    assert np.average(sc.velocity[:, :, :]) == 0.

    # Warmup
    run_scenario(sc, steps=500)
    # sampling:
    steps = 20000
    v = np.zeros((steps, n, n, n, 3))
    for i in range(steps):
        run_scenario(sc, steps=2)
        v[i, :, :, :, :] = np.copy(sc.velocity[:, :, :, :])

    v = v.reshape((steps*n*n*n, 3))
    np.testing.assert_allclose(np.mean(v, axis=0), [0, 0, 0], atol=6E-7)
    np.testing.assert_allclose(
        np.var(v, axis=0), [temperature, temperature, temperature], rtol=1E-2)
    v_hist, v_bins = np.histogram(v, bins=11, range=(-.08, .08), density=True)

    # Calculate expected values from single
    v_expected = []
    for i in range(len(v_hist)):
        # Maxwell distribution
        res = np.exp(-v_bins[i]**2/(2.*temperature)) / \
            np.sqrt(2*np.pi*temperature)
        res = 1./(v_bins[i+1]-v_bins[i]) * \
            single_component_maxwell(v_bins[i], v_bins[i+1], temperature)
        v_expected.append(res)
    v_expected = np.array(v_expected)

    # 8% accuracy on the entire histogram
    np.testing.assert_allclose(v_hist, v_expected, rtol=0.08)
    # 0.5% accuracy on the middle part
    remove = 3
    np.testing.assert_allclose(
        v_hist[remove:-remove], v_expected[remove:-remove], rtol=0.005)

