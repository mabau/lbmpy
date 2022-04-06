try:
    import pyximport

    pyximport.install(language_level=3)
    from lbmpy.phasefield.simplex_projection import simplex_projection_2d  # NOQA
except ImportError:
    try:
        from lbmpy.phasefield.simplex_projection import simplex_projection_2d  # NOQA
    except ImportError:
        raise ImportError("neither pyximport nor binary module simplex_projection_2d available.")

import sympy as sp

from lbmpy.creationfunctions import create_lb_update_rule, LBMConfig, LBMOptimisation
from lbmpy.enums import Stencil
from lbmpy.macroscopic_value_kernels import pdf_initialization_assignments
from lbmpy.phasefield.analytical import chemical_potentials_from_free_energy, force_from_phi_and_mu
from lbmpy.phasefield.cahn_hilliard_lbm import cahn_hilliard_lb_method
from lbmpy.stencils import LBStencil
from pystencils import Assignment, create_data_handling, create_kernel
from pystencils.fd import Diff, discretize_spatial, expand_diff_full
from pystencils.fd.derivation import FiniteDifferenceStencilDerivation


def forth_order_isotropic_discretize(field):
    second_neighbor_stencil = [(i, j)
                               for i in (-2, -1, 0, 1, 2)
                               for j in (-2, -1, 0, 1, 2)
                               ]
    x_diff = FiniteDifferenceStencilDerivation((0,), second_neighbor_stencil)
    x_diff.set_weight((2, 0), sp.Rational(1, 10))
    x_diff.assume_symmetric(0, anti_symmetric=True)
    x_diff.assume_symmetric(1)
    x_diff_stencil = x_diff.get_stencil(isotropic=True)

    y_diff = FiniteDifferenceStencilDerivation((1,), second_neighbor_stencil)
    y_diff.set_weight((0, 2), sp.Rational(1, 10))
    y_diff.assume_symmetric(1, anti_symmetric=True)
    y_diff.assume_symmetric(0)
    y_diff_stencil = y_diff.get_stencil(isotropic=True)

    substitutions = {}
    for i in range(field.index_shape[0]):
        substitutions.update({Diff(field(i), 0): x_diff_stencil.apply(field(i)),
                              Diff(field(i), 1): y_diff_stencil.apply(field(i))})
    return substitutions


def create_model(domain_size, num_phases, coeff_a, coeff_epsilon, gabd, alpha=1, penalty_factor=0.01,
                 simplex_projection=False):
    def lapl(e):
        return sum(Diff(Diff(e, i), i) for i in range(dh.dim))

    def interfacial_chemical_potential(c):
        result = []
        n = len(c)
        for i in range(n):
            entry = 0
            for k in range(n):
                if i == k:
                    continue
                eps = coeff_epsilon[(k, i)] if i < k else coeff_epsilon[(i, k)]
                entry += alpha ** 2 * eps ** 2 * (c[k] * lapl(c[i]) - c[i] * lapl(c[k]))
            result.append(entry)
        return -sp.Matrix(result)

    def bulk(c):
        result = 0
        for i in range(num_phases):
            for j in range(i):
                result += (c[i] ** 2 * c[j] ** 2) / (4 * coeff_a[i, j])
        for i in range(num_phases):
            for j in range(i):
                for k in range(j):
                    result += gabd * c[i] * c[j] * c[k]
        return result

    # -------------- Data ------------------
    dh = create_data_handling(domain_size, periodicity=(True, True), default_ghost_layers=2)

    c = dh.add_array("c", values_per_cell=num_phases)
    rho = dh.add_array("rho", values_per_cell=1)
    mu = dh.add_array("mu", values_per_cell=num_phases, latex_name="\\mu")
    force = dh.add_array("F", values_per_cell=dh.dim)
    u = dh.add_array("u", values_per_cell=dh.dim)

    # Distribution functions for each order parameter
    pdf_field = []
    pdf_dst_field = []
    for i in range(num_phases):
        pdf_field_local = dh.add_array(f"pdf_ch_{i}", values_per_cell=9)  # 9 for D2Q9
        pdf_dst_field_local = dh.add_array(f"pdfs_ch_{i}_dst", values_per_cell=9)
        pdf_field.append(pdf_field_local)
        pdf_dst_field.append(pdf_dst_field_local)

    # Distribution functions for the hydrodynamics
    pdf_hydro_field = dh.add_array("pdfs", values_per_cell=9)
    pdf_hydro_dst_field = dh.add_array("pdfs_dst", values_per_cell=9)

    # ------------- Compute kernels --------
    c_vec = c.center_vector
    f_penalty = penalty_factor * (1 - sum(c_vec[i] for i in range(num_phases))) ** 2
    f_bulk = bulk(c_vec) + f_penalty
    print(f_bulk)
    mu_eq = chemical_potentials_from_free_energy(f_bulk, order_parameters=c_vec)
    mu_eq += interfacial_chemical_potential(c_vec)
    mu_eq = [expand_diff_full(mu_i, functions=c) for mu_i in mu_eq]
    mu_assignments = [Assignment(mu(i), discretize_spatial(mu_i, dx=1, stencil='isotropic'))
                      for i, mu_i in enumerate(mu_eq)]
    mu_compute_kernel = create_kernel(mu_assignments).compile()

    mu_discretize_substitutions = forth_order_isotropic_discretize(mu)
    force_rhs = force_from_phi_and_mu(order_parameters=c_vec, dim=dh.dim, mu=mu.center_vector)
    force_rhs = force_rhs.subs(mu_discretize_substitutions)
    force_assignments = [Assignment(force(i), force_rhs[i]) for i in range(dh.dim)]
    force_kernel = create_kernel(force_assignments).compile()

    ch_collide_kernels = []
    ch_methods = []
    for i in range(num_phases):
        ch_method = cahn_hilliard_lb_method(LBStencil(Stencil.D2Q9), mu(i),
                                            relaxation_rate=1.0, gamma=1.0)
        ch_methods.append(ch_method)

        lbm_config = LBMConfig(lb_method=ch_method, kernel_type='collide_only', density_input=c(i),
                               velocity_input=u.center_vector, compressible=True, zero_centered=False)
        lbm_opt = LBMOptimisation(symbolic_field=pdf_field[i])
        ch_update_rule = create_lb_update_rule(lbm_config=lbm_config,
                                               lbm_optimisation=lbm_opt)

        ch_assign = ch_update_rule.all_assignments
        ch_kernel = create_kernel(ch_assign).compile()
        ch_collide_kernels.append(ch_kernel)

    ch_stream_kernels = []
    for i in range(num_phases):
        ch_method = ch_methods[i]

        lbm_config = LBMConfig(lb_method=ch_method, kernel_type='stream_pull_only',
                               temporary_field_name=pdf_dst_field[i].name, zero_centered=False)
        lbm_opt = LBMOptimisation(symbolic_field=pdf_field[i])
        ch_update_rule = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

        ch_assign = ch_update_rule.all_assignments
        ch_kernel = create_kernel(ch_assign).compile()
        ch_stream_kernels.append(ch_kernel)

    # Defining the initialisation kernels for the C-H pdfs
    init_kernels = []
    for i in range(num_phases):
        ch_method = ch_methods[i]
        init_assign = pdf_initialization_assignments(lb_method=ch_method,
                                                     density=c_vec[i],
                                                     velocity=(0, 0),
                                                     pdfs=pdf_field[i].center_vector)
        init_kernel = create_kernel(init_assign).compile()
        init_kernels.append(init_kernel)

    getter_kernels = []
    for i in range(num_phases):
        cqc = ch_methods[i].conserved_quantity_computation
        output_assign = cqc.output_equations_from_pdfs(pdf_field[i].center_vector,
                                                       {'density': c(i)})
        getter_kernel = create_kernel(output_assign).compile()
        getter_kernels.append(getter_kernel)

    lbm_config = LBMConfig(kernel_type='collide_only', relaxation_rate=1.0, force=force,
                           compressible=True, zero_centered=False)
    lbm_opt = LBMOptimisation(symbolic_field=pdf_hydro_field)
    collide_assign = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    collide_kernel = create_kernel(collide_assign).compile()

    lbm_config = LBMConfig(kernel_type='stream_pull_only', temporary_field_name=pdf_hydro_dst_field.name,
                           zero_centered=False, output={"density": rho, "velocity": u})
    lbm_opt = LBMOptimisation(symbolic_field=pdf_hydro_field)
    stream_assign = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)

    stream_kernel = create_kernel(stream_assign).compile()

    method_collide = collide_assign.method
    init_hydro_assign = pdf_initialization_assignments(lb_method=method_collide,
                                                       density=rho.center,
                                                       velocity=u.center_vector,
                                                       pdfs=pdf_hydro_field.center_vector)
    init_hydro_kernel = create_kernel(init_hydro_assign).compile()

    output_hydro_assign = cqc.output_equations_from_pdfs(pdf_hydro_field.center_vector,
                                                         {'density': rho.center,
                                                          'velocity': u.center_vector}).all_assignments
    # Creating getter kernel to extract quantities
    getter_hydro_kernel = create_kernel(output_hydro_assign).compile()  # getter kernel

    # Setting values of arrays
    dh.cpu_arrays[c.name].fill(0)
    dh.cpu_arrays[u.name].fill(0)
    dh.cpu_arrays[rho.name].fill(1)
    dh.cpu_arrays[mu.name].fill(0)
    dh.cpu_arrays[force.name].fill(0)

    def init():
        for k in init_kernels:
            dh.run_kernel(k)
        dh.run_kernel(init_hydro_kernel)

    pdf_sync_fns = []
    for i in range(num_phases):
        sync_fn = dh.synchronization_function([pdf_field[i].name])
        pdf_sync_fns.append(sync_fn)
    hydro_sync_fn = dh.synchronization_function([pdf_hydro_field.name])
    c_sync_fn = dh.synchronization_function([c.name])
    mu_sync = dh.synchronization_function([mu.name])

    def run(steps):
        for t in range(steps):
            # μ and P
            c_sync_fn()
            dh.run_kernel(mu_compute_kernel)
            mu_sync()
            dh.run_kernel(force_kernel)

            # Hydrodynamic LB
            dh.run_kernel(collide_kernel)  # running collision kernel
            hydro_sync_fn()
            dh.run_kernel(stream_kernel)  # running streaming kernel
            dh.swap(pdf_hydro_field.name, pdf_hydro_dst_field.name)
            dh.run_kernel(getter_hydro_kernel)

            # Cahn-Hilliard LBs
            for i in range(num_phases):
                dh.run_kernel(ch_collide_kernels[i])
                pdf_sync_fns[i]()
                dh.run_kernel(ch_stream_kernels[i])
                dh.swap(pdf_field[i].name, pdf_dst_field[i].name)
                dh.run_kernel(getter_kernels[i])
            if simplex_projection:
                simplex_projection_2d(dh.cpu_arrays[c.name])
        return dh.cpu_arrays[c.name][1:-1, 1:-1, :]

    return dh, init, run
