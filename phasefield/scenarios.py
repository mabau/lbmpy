import sympy as sp
import numpy as np
from lbmpy.phasefield.phasefieldstep import PhaseFieldStep
from lbmpy.phasefield.analytical import free_energy_functional_3_phases, free_energy_functional_n_phases, \
    symbolic_order_parameters, free_energy_functional_n_phases_penalty_term


def create_three_phase_model(alpha=1, kappa=(0.015, 0.015, 0.015), include_rho=True, **kwargs):
    parameters = {sp.Symbol("alpha"): alpha,
                  sp.Symbol("kappa_0"): kappa[0],
                  sp.Symbol("kappa_1"): kappa[1],
                  sp.Symbol("kappa_2"): kappa[2]}
    if include_rho:
        order_parameters = sp.symbols("rho phi psi")
        free_energy, transformation_matrix = free_energy_functional_3_phases(order_parameters)
        free_energy = free_energy.subs(parameters)
        op_transformation = np.array(transformation_matrix).astype(float)
        op_transformation_inv = np.array(transformation_matrix.inv()).astype(float)

        def concentration_to_order_parameters(c):
            phi = np.tensordot(c, op_transformation, axes=([-1], [1]))
            return phi

        return PhaseFieldStep(free_energy, order_parameters, density_order_parameter=order_parameters[0],
                              concentration_to_order_parameters=concentration_to_order_parameters,
                              order_parameters_to_concentrations=lambda phi: np.tensordot(phi, op_transformation_inv,
                                                                                          axes=([-1], [1])),
                              **kwargs)
    else:
        order_parameters = sp.symbols("phi psi")
        free_energy, transformation_matrix = free_energy_functional_3_phases((1,) + order_parameters)
        free_energy = free_energy.subs(parameters)
        op_transformation = transformation_matrix.copy()
        op_transformation.row_del(0)  # rho is assumed to be 1 - is not required
        op_transformation = np.array(op_transformation).astype(float)
        reverse = transformation_matrix.inv() * sp.Matrix(sp.symbols("rho phi psi"))
        op_transformation_inv = sp.lambdify(sp.symbols("phi psi"), reverse.subs(sp.Symbol("rho"), 1))

        def order_parameters_to_concentrations(phi):
            phi = np.array(phi)
            transformed = op_transformation_inv(phi[..., 0], phi[..., 1])
            return np.moveaxis(transformed[:, 0], 0, -1)

        def concentration_to_order_parameters(c):
            phi = np.tensordot(c, op_transformation, axes=([-1], [1]))
            return phi

        return PhaseFieldStep(free_energy, order_parameters, density_order_parameter=None,
                              concentration_to_order_parameters=concentration_to_order_parameters,
                              order_parameters_to_concentrations=order_parameters_to_concentrations,
                              **kwargs)


def create_n_phase_model(alpha=1, num_phases=4,
                         surface_tensions=lambda i, j: 0.005 if i != j else 0,
                         f1=lambda c: c ** 2 * (1 - c) ** 2,
                         f2=lambda c: c ** 2 * (1 - c) ** 2,
                         **kwargs):
    order_parameters = symbolic_order_parameters(num_phases - 1)
    free_energy = free_energy_functional_n_phases(num_phases, surface_tensions, alpha,
                                                  order_parameters, f1=f1, f2=f2)
    return PhaseFieldStep(free_energy, order_parameters, **kwargs)


def create_n_phase_model_penalty_term(alpha=1, num_phases=4, kappa=0.015, penalty_term_factor=0.01, **kwargs):
    order_parameters = symbolic_order_parameters(num_phases)
    free_energy = free_energy_functional_n_phases_penalty_term(order_parameters, alpha, kappa, penalty_term_factor)
    return PhaseFieldStep(free_energy, order_parameters, **kwargs)
