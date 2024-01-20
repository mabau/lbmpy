from pystencils import Assignment, AssignmentCollection
from pystencils.sympyextensions import scalar_product
from pystencils.simp.subexpression_insertion import insert_constants

from lbmpy.phasefield_allen_cahn.derivatives import isotropic_gradient_symbolic, laplacian_symbolic

import sympy as sp

VELOCITY_SYMBOLS = sp.symbols(f"u_:{3}")
GRAD_T_SYMBOLS = sp.symbols(f"gratT_:{3}")
GRAD_K_SYMBOLS = sp.symbols(f"gratK_:{3}")
LAPLACIAN_SYMBOL = sp.Symbol("lap")


def get_runge_kutta_update_assignments(stencil, phase_field, temperature_field, velocity_field, runge_kutta_fields,
                                       conduction_h, conduction_l, heat_capacity_h, heat_capacity_l,
                                       density, stabiliser=1):
    dimensions = len(stencil[0])

    grad_temperature = isotropic_gradient_symbolic(temperature_field, stencil)
    lap_temperature = laplacian_symbolic(temperature_field, stencil)
    grad_conduction = _get_conduction_gradient(stencil, phase_field, conduction_h, conduction_l)

    grad_rk = [isotropic_gradient_symbolic(rk, stencil) for rk in runge_kutta_fields]
    lap_rk = [laplacian_symbolic(rk, stencil) for rk in runge_kutta_fields]

    dot_u_grad_t = scalar_product(VELOCITY_SYMBOLS[:dimensions], GRAD_T_SYMBOLS[:dimensions])
    dot_grad_k_grad_t = scalar_product(GRAD_K_SYMBOLS[:dimensions], GRAD_T_SYMBOLS[:dimensions])

    conduction = conduction_l + phase_field.center * sp.nsimplify(conduction_h - conduction_l)
    conduction_times_lap = conduction * LAPLACIAN_SYMBOL

    heat_capacity = heat_capacity_l + phase_field.center * sp.nsimplify(heat_capacity_h - heat_capacity_l)

    rho_cp = 1.0 / (density * heat_capacity)
    end_term = dot_grad_k_grad_t + conduction_times_lap

    update_stage_1 = temperature_field.center + stabiliser * 0.5 * (-1.0 * dot_u_grad_t + rho_cp * end_term)
    subexpressions_1 = _get_stage(dimensions, velocity_field, grad_temperature, grad_conduction, lap_temperature)
    stage_1 = AssignmentCollection(main_assignments=[Assignment(runge_kutta_fields[0].center, update_stage_1)],
                                   subexpressions=subexpressions_1)

    if len(runge_kutta_fields) == 1:
        update_stage_2 = temperature_field.center + stabiliser * (-1.0 * dot_u_grad_t + rho_cp * end_term)
        subexpressions_2 = _get_stage(dimensions, velocity_field, grad_rk[0], grad_conduction, lap_rk[0])
        stage_2 = AssignmentCollection(main_assignments=[Assignment(temperature_field.center, update_stage_2)],
                                       subexpressions=subexpressions_2)

        return [insert_constants(ac) for ac in [stage_1, stage_2]]

    update_stage_2 = temperature_field.center + stabiliser * 0.5 * (-1.0 * dot_u_grad_t + rho_cp * end_term)
    subexpressions_2 = _get_stage(dimensions, velocity_field, grad_rk[0], grad_conduction, lap_rk[0])
    stage_2 = AssignmentCollection(main_assignments=[Assignment(runge_kutta_fields[1].center, update_stage_2)],
                                   subexpressions=subexpressions_2)

    update_stage_3 = temperature_field.center + stabiliser * 1.0 * (-1.0 * dot_u_grad_t + rho_cp * end_term)
    subexpressions_3 = _get_stage(dimensions, velocity_field, grad_rk[1], grad_conduction, lap_rk[1])
    stage_3 = AssignmentCollection(main_assignments=[Assignment(runge_kutta_fields[2].center, update_stage_3)],
                                   subexpressions=subexpressions_3)

    update_stage_4 = stabiliser * 1.0 * (-1.0 * dot_u_grad_t + rho_cp * end_term)
    rk_update = 2.0 * runge_kutta_fields[0].center + 4.0 * runge_kutta_fields[1].center + 2.0 * runge_kutta_fields[
        2].center
    update_stage_4 = (1.0 - 4.0 / 3.0) * temperature_field.center + (rk_update - update_stage_4) / 6.0
    subexpressions_4 = _get_stage(dimensions, velocity_field, grad_rk[2], grad_conduction, lap_rk[2])
    stage_4 = AssignmentCollection(main_assignments=[Assignment(temperature_field.center, update_stage_4)],
                                   subexpressions=subexpressions_4)

    return [insert_constants(ac) for ac in [stage_1, stage_2, stage_3, stage_4]]


def get_initialiser_assignments(temperature_field, runge_kutta_fields):
    result = []
    for i in range(len(runge_kutta_fields)):
        result.append(Assignment(runge_kutta_fields[i].center, temperature_field.center))

    return result


def _get_conduction_gradient(stencil, phase_field, conduction_h, conduction_l):
    dimensions = len(stencil[0])
    grad_phase = isotropic_gradient_symbolic(phase_field, stencil)

    free_symbols = set()
    for i in range(dimensions):
        free_symbols.update(grad_phase[i].free_symbols)

    subs_dict = {}

    for f in free_symbols:
        subs_dict[f] = interpolate_field_access(f, conduction_h, conduction_l)

    result = list()
    for i in range(dimensions):
        eq = grad_phase[i].subs(subs_dict)
        # replace very small numbers by zero
        eq = eq.xreplace(dict([(n, 0) for n in eq.atoms(sp.Float) if abs(n) < 1e-16]))
        result.append(eq)

    return result


def interpolate_field_access(field_access, upper, lower):
    return lower + field_access * sp.nsimplify(upper - lower)


def _get_stage(dimensions, velocity_field, gradient_t, gradient_k, laplacian):
    result = list()

    for i in range(dimensions):
        result.append(Assignment(VELOCITY_SYMBOLS[i], velocity_field.center_vector[i]))

    for i in range(dimensions):
        result.append(Assignment(GRAD_T_SYMBOLS[i], gradient_t[i]))

    for i in range(dimensions):
        result.append(Assignment(GRAD_K_SYMBOLS[i], gradient_k[i]))

    result.append(Assignment(LAPLACIAN_SYMBOL, laplacian))

    return result
