from lbmpy.boundaries import NoSlip
from lbmpy.plot2d import phase_plot
from pystencils import make_slice


def drops_between_two_phase():
    import lbmpy.plot2d as plt
    import sympy as sp
    from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
    from lbmpy.phasefield.analytical import free_energy_functional_n_phases_penalty_term
    c = sp.symbols("c_:4")
    F = free_energy_functional_n_phases_penalty_term(c, 1, [0.01, 0.02, 0.001, 0.02], 0.0001)
    sc = PhaseFieldStep(F, c, domain_size=(2 * 200, 2 * 70), openmp=4, hydro_dynamic_relaxation_rate=1.9)
    sc.set_concentration(make_slice[:, 0.5:], [1, 0, 0, 0])
    sc.set_concentration(make_slice[:, :0.5], [0, 1, 0, 0])
    sc.set_concentration(make_slice[0.2:0.4, 0.3:0.7], [0, 0, 1, 0])
    sc.set_concentration(make_slice[0.7:0.8, 0.3:0.7], [0, 0, 0, 1])
    sc.set_pdf_fields_from_macroscopic_values()

    for i in range(500000):
        print("Step", i)
        plt.figure(figsize=(14, 7))
        phase_plot(sc, make_slice[:, 5:-5], linewidth=0.1)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('step%03d.png' % (i,), bbox_inches='tight')
        plt.clf()
        sc.run(25)
    plt.show()


def falling_drop():
    import lbmpy.plot2d as plt
    import sympy as sp
    from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
    from lbmpy.phasefield.analytical import free_energy_functional_n_phases_penalty_term
    c = sp.symbols("c_:3")
    F = free_energy_functional_n_phases_penalty_term(c, 1, [0.001, 0.001, 0.0005])
    gravity = -0.5e-5
    sc = PhaseFieldStep(F, c, domain_size=(160, 200), openmp=4,
                        hydro_dynamic_relaxation_rate=1.9,
                        order_parameter_force={2: (0, gravity), 1: (0, 0)})
    sc.set_concentration(make_slice[:, 0.4:], [1, 0, 0])
    sc.set_concentration(make_slice[:, :0.4], [0, 1, 0])
    sc.set_concentration(make_slice[0.45:0.55, 0.8:0.9], [0, 0, 1])
    sc.hydroLbmStep.boundary_handling.set_boundary(NoSlip(), make_slice[:, 0])
    #sc.hydroLbmStep.boundary_handling.set_boundary(NoSlip(), make_slice[:, -1])

    sc.set_pdf_fields_from_macroscopic_values()
    for i in range(650):
        print("Step", i)
        plt.figure(figsize=(14, 10))
        plt.subplot(1, 2, 1)
        phase_plot(sc, make_slice[:, 5:-15], linewidth=0.1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.vector_field(sc.velocity[:, 5:-15, :])
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('fallingDrop_boundary2_%05d.png' % (i,), bbox_inches='tight')
        plt.clf()
        sc.run(200)
    plt.show()
