import os
from tempfile import TemporaryDirectory

from lbmpy.boundaries import NoSlip
from lbmpy.phasefield.experiments2D import (
    create_two_drops_between_phases, write_phase_field_picture_sequence,
    write_phase_velocity_picture_sequence)
from lbmpy.phasefield.scenarios import *
from pystencils import make_slice


def create_falling_drop(domain_size=(160, 200), omega=1.9, kappas=(0.001, 0.001, 0.0005), **kwargs):
    c = sp.symbols("c_:3")
    free_energy = free_energy_functional_n_phases_penalty_term(c, 1, kappas)
    gravity = -0.1e-5
    if 'optimization' not in kwargs:
        kwargs['optimization'] = {'openmp': 4}
    sc = PhaseFieldStep(free_energy, c, domain_size=domain_size, hydro_dynamic_relaxation_rate=omega,
                        order_parameter_force={2: (0, gravity), 1: (0, 0), 0: (0, 0)}, **kwargs)
    sc.set_concentration(make_slice[:, 0.4:], [1, 0, 0])
    sc.set_concentration(make_slice[:, :0.4], [0, 1, 0])
    sc.set_concentration(make_slice[0.45:0.55, 0.8:0.9], [0, 0, 1])
    sc.hydro_lbm_step.boundary_handling.set_boundary(NoSlip(), make_slice[:, 0])

    sc.set_pdf_fields_from_macroscopic_values()
    return sc


def test_drops_between_phases():
    sc = create_two_drops_between_phases()
    with TemporaryDirectory() as tmp_dir:
        file_pattern = os.path.join(tmp_dir, "output_%d.png")
        write_phase_field_picture_sequence(sc, file_pattern, total_steps=200)
    assert np.isfinite(np.max(sc.phi[:, :, :]))


def test_falling_drop():
    sc = create_falling_drop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TemporaryDirectory() as tmp_dir:
            file_pattern = os.path.join(tmp_dir, "output_%d.png")
            write_phase_velocity_picture_sequence(sc, file_pattern, total_steps=200)
        assert np.isfinite(np.max(sc.phi[:, :, :]))


def test_setup():
    domain_size = (30, 15)

    scenarios = [
        create_three_phase_model(domain_size=domain_size, include_rho=True),
        # create_three_phase_model(domain_size=domain_size, include_rho=False),
        create_n_phase_model_penalty_term(domain_size=domain_size, num_phases=4),
    ]
    for i, sc in enumerate(scenarios):
        print(f"Testing scenario {i}")
        sc.set_concentration(make_slice[:, :0.5], [1, 0, 0])
        sc.set_concentration(make_slice[:, 0.5:], [0, 1, 0])
        sc.set_concentration(make_slice[0.4:0.6, 0.4:0.6], [0, 0, 1])
        sc.set_pdf_fields_from_macroscopic_values()
        sc.run(10)


def test_fd_cahn_hilliard():
    sc = create_n_phase_model_penalty_term(domain_size=(100, 50), num_phases=3,
                                           solve_cahn_hilliard_with_finite_differences=True)
    sc.set_concentration(make_slice[:, 0.5:], [1, 0, 0])
    sc.set_concentration(make_slice[:, :0.5], [0, 1, 0])
    sc.set_concentration(make_slice[0.3:0.7, 0.3:0.7], [0, 0, 1])
    sc.set_pdf_fields_from_macroscopic_values()
    sc.run(100)
    assert np.isfinite(np.max(sc.concentration[:, :]))
