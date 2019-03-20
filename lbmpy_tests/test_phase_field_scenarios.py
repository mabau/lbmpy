import os
import warnings
from tempfile import TemporaryDirectory
import numpy as np
import sympy as sp
from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
from lbmpy.phasefield.analytical import free_energy_functional_n_phases_penalty_term
from pystencils import make_slice
from lbmpy.boundaries import NoSlip
from lbmpy.phasefield.experiments2D import create_two_drops_between_phases, write_phase_field_picture_sequence, \
    write_phase_velocity_picture_sequence


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
