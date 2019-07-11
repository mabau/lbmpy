from lbmpy.phasefield.scenarios import *
from pystencils import make_slice


def test_setup():
    domain_size = (30, 15)

    scenarios = [
        create_three_phase_model(domain_size=domain_size, include_rho=True),
        #create_three_phase_model(domain_size=domain_size, include_rho=False),
        create_n_phase_model_penalty_term(domain_size=domain_size, num_phases=4),
    ]
    for i, sc in enumerate(scenarios):
        print("Testing scenario", i)
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
