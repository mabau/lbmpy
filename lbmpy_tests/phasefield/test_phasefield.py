import pytest
import sympy as sp
import numpy as np

from lbmpy.phasefield.analytical import (
    analytic_interface_profile, chemical_potentials_from_free_energy, cosh_integral,
    force_from_phi_and_mu, force_from_pressure_tensor, free_energy_functional_n_phases,
    pressure_tensor_from_free_energy, substitute_laplacian_by_sum, symbolic_order_parameters,
    symmetric_symbolic_surface_tension)
from pystencils.fd import evaluate_diffs, expand_diff_full
from pystencils.enums import Target

from lbmpy.phasefield.experiments2D import liquid_lens_setup
from lbmpy.phasefield.contact_angle_circle_fitting import liquid_lens_neumann_angles
from lbmpy.phasefield.post_processing import analytic_neumann_angles


def test_analytic_interface_solution():
    """Ensures that the tanh is an analytical solution for the prescribed free energy / chemical potential
    """
    num_phases = 4
    phi = symbolic_order_parameters(num_phases - 1)
    free_energy = free_energy_functional_n_phases(num_phases, order_parameters=phi).subs({p: 0 for p in phi[1:]})
    mu_diff_eq = chemical_potentials_from_free_energy(free_energy, [phi[0]])[0]

    x = sp.Symbol("x")
    sol = analytic_interface_profile(x)

    inserted = mu_diff_eq.subs(phi[0], sol)
    assert sp.expand(evaluate_diffs(inserted, x)) == 0


def test_surface_tension_derivation():
    """Computes the excess free energy per unit area of an interface transition between two phases
    which should give exactly the surface tension parameter"""
    num_phases = 4
    eta = sp.Symbol("eta")

    free_energy = free_energy_functional_n_phases(num_phases, interface_width=eta)
    phi = symbolic_order_parameters(num_phases)

    x = sp.Symbol("x")
    sol = analytic_interface_profile(x, interface_width=eta)

    for a, b in [(1, 3), (0, 1)]:
        substitutions = {phi[a]: sol}
        if b < len(phi) - 1:
            substitutions[phi[b]] = 1 - sol
        for i, phi_i in enumerate(phi[:-1]):
            if i not in (a, b):
                substitutions[phi_i] = 0

        free_energy_2_phase = sp.simplify(evaluate_diffs(free_energy.subs(substitutions), x))
        result = cosh_integral(free_energy_2_phase, x)
        assert result == symmetric_symbolic_surface_tension(a, b)


def test_pressure_tensor():
    """
    Checks that the following ways are equivalent:
    1) phi -> mu -> force
    2) phi -> pressure tensor -> force
    """
    dim = 3
    c = symbolic_order_parameters(3)
    f = free_energy_functional_n_phases(order_parameters=c)

    mu = chemical_potentials_from_free_energy(f, c)
    mu = substitute_laplacian_by_sum(mu, dim)
    force_chem_pot = expand_diff_full(force_from_phi_and_mu(c, dim, mu), functions=c)

    p = pressure_tensor_from_free_energy(f, c, dim)
    force_pressure_tensor = force_from_pressure_tensor(p, functions=c)

    for f1_i, f2_i in zip(force_chem_pot, force_pressure_tensor):
        assert sp.expand(f1_i - f2_i) == 0


def test_neumann_angle():
    pytest.importorskip('skimage')
    kappa3 = 0.03
    alpha = 1

    sc = liquid_lens_setup(domain_size=(150, 60), optimization={'target': Target.CPU},
                           kappas=(0.01, 0.02, kappa3),
                           cahn_hilliard_relaxation_rates=[np.nan, 1, 3 / 2],
                           cahn_hilliard_gammas=[1, 1, 1 / 3],
                           alpha=alpha)

    sc.run(10000)

    angles = liquid_lens_neumann_angles(sc.concentration[:, :, :])
    np.testing.assert_almost_equal(sum(angles), 360)

    analytic_angles = analytic_neumann_angles([0.01, 0.02, kappa3])
    for ref, simulated in zip(analytic_angles, angles):
        assert np.abs(ref - simulated) < 8

    # to show the phasefield use:
    # plt.phase_plot_for_step(sc)
