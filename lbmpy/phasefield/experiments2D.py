import sympy as sp

from lbmpy.phasefield.analytical import free_energy_functional_n_phases_penalty_term
from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
from lbmpy.phasefield.scenarios import create_three_phase_model
from pystencils import make_slice


def write_phase_field_picture_sequence(sc, filename='two_drops_%05d.png',
                                       time_steps_between_frames=25, total_steps=9000):
    import lbmpy.plot as plt
    outer_iterations = total_steps // time_steps_between_frames
    for i in range(outer_iterations):
        plt.figure(figsize=(14, 7))
        plt.phase_plot(sc.phi[:, :], linewidth=0.1)
        plt.axis('off')
        plt.savefig(filename % (i,), bbox_inches='tight')
        plt.clf()
        sc.run(time_steps_between_frames)


def write_phase_velocity_picture_sequence(sc, filename='falling_drop_%05d.png',
                                          time_steps_between_frames=25, total_steps=9000):
    import lbmpy.plot as plt
    outer_iterations = total_steps // time_steps_between_frames
    for i in range(outer_iterations):
        plt.figure(figsize=(14, 10))
        plt.subplot(1, 2, 1)
        plt.phase_plot(sc.phi[:, :], linewidth=0.1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.vector_field(sc.velocity[:, 5:-15, :])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename % (i,), bbox_inches='tight')
        plt.clf()
        sc.run(time_steps_between_frames)
    plt.show()


def liquid_lens_setup(domain_size=(200, 120), omega=1.2, kappas=(0.01, 0.02, 0.05), alpha=1, **kwargs):
    """Sets up a liquid lens scenario with the 3 phase model.

              ---------
        (0)   |       |
    ----------|  (2)  |------
        (1)   |       |
              ---------

    """
    sc = create_three_phase_model(domain_size=domain_size, alpha=alpha, kappa=kappas,
                                  hydro_dynamic_relaxation_rate=omega, **kwargs)
    sc.set_concentration(make_slice[:, 0.5:], [1, 0, 0])
    sc.set_concentration(make_slice[:, :0.5], [0, 1, 0])
    sc.set_concentration(make_slice[0.3:0.7, 0.3:0.7], [0, 0, 1])
    sc.set_pdf_fields_from_macroscopic_values()
    return sc


def create_two_drops_between_phases(domain_size=(400, 150), omega=1.7, kappas=(0.01, 0.02, 0.001, 0.02),
                                    penalty_term_factor=0.0001, **kwargs):
    """Sets up a fully periodic scenario with two drops at the boundary between two phases.

              ---------          ---------
        (0)   |       |          |       |
    ----------|  (2)  |----------|  (3)  |-------
        (1)   |       |          |       |
              ---------          ---------

    The bracketed numbers denote the phase indices.

    This set up can be used to validate the Neumann angles between the phases as
    function of the κ surface tension parameters.

    Args:
        domain_size: size of 2D simulation domain
        omega: relaxation rate for the hydrodynamic lattice Boltzmann, determining viscosity of fluid
        kappas: parameters for phase field model, determining surface tensions between phases
        penalty_term_factor: the n-phase model uses a penalty term ensuring that the phase parameters sum up to one.
                             This penalty term is weighted with the given factor.
        kwargs: passed along to constructor of PhaseFieldStep

    Returns:
        a scenario as PhaseFieldStep instance
    """
    assert len(domain_size) == 2
    c = sp.symbols("c_:4")
    free_energy = free_energy_functional_n_phases_penalty_term(c, 1, kappas, penalty_term_factor)
    if 'optimization' not in kwargs:
        kwargs['optimization'] = {'openmp': 4}
    sc = PhaseFieldStep(free_energy, c, domain_size=domain_size, hydro_dynamic_relaxation_rate=omega, **kwargs)
    sc.set_concentration(make_slice[:, 0.5:], [1, 0, 0, 0])
    sc.set_concentration(make_slice[:, :0.5], [0, 1, 0, 0])
    sc.set_concentration(make_slice[0.2:0.4, 0.3:0.7], [0, 0, 1, 0])
    sc.set_concentration(make_slice[0.7:0.8, 0.3:0.7], [0, 0, 0, 1])
    sc.set_pdf_fields_from_macroscopic_values()
    return sc
