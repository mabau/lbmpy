import sympy as sp
from dataclasses import dataclass

from lbmpy.methods.abstractlbmethod import LbmCollisionRule
from pystencils import Assignment, AssignmentCollection
from pystencils.field import Field


@dataclass
class PSMConfig:
    fraction_field: Field = None
    """
    Fraction field for PSM 
    """

    object_velocity_field: Field = None
    """
    Object velocity field for PSM 
    """

    SC: int = 1
    """
    Solid collision option for PSM
    """

    MaxParticlesPerCell: int = 1
    """
    Maximum number of particles overlapping with a cell 
    """

    individual_fraction_field: Field = None
    """
    Fraction field for each overlapping particle in PSM 
    """

    particle_force_field: Field = None
    """
    Force field for each overlapping particle in PSM 
    """


def add_psm_to_collision_rule(collision_rule, psm_config):
    if psm_config.individual_fraction_field is None:
        psm_config.individual_fraction_field = psm_config.fraction_field

    method = collision_rule.method
    pre_collision_pdf_symbols = method.pre_collision_pdf_symbols
    stencil = method.stencil

    # Get equilibrium from object velocity for solid collision
    forces_rhs = [0] * psm_config.MaxParticlesPerCell * stencil.D
    solid_collisions = [0] * stencil.Q
    for p in range(psm_config.MaxParticlesPerCell):
        equilibrium_fluid = method.get_equilibrium_terms()
        equilibrium_solid = []
        for eq in equilibrium_fluid:
            eq_sol = eq
            for i in range(stencil.D):
                eq_sol = eq_sol.subs(sp.Symbol("u_" + str(i)),
                                     psm_config.object_velocity_field.center(p * stencil.D + i), )
            equilibrium_solid.append(eq_sol)

        # Build solid collision
        for i, (eqFluid, eqSolid, f, offset) in enumerate(
                zip(equilibrium_fluid, equilibrium_solid, pre_collision_pdf_symbols, stencil)):
            inverse_direction_index = stencil.stencil_entries.index(stencil.inverse_stencil_entries[i])
            if psm_config.SC == 1:
                solid_collision = psm_config.individual_fraction_field.center(p) * (
                    (
                        pre_collision_pdf_symbols[inverse_direction_index]
                        - equilibrium_fluid[inverse_direction_index]
                    )
                    - (f - eqSolid)
                )
            elif psm_config.SC == 2:
                # TODO get relaxation rate vector from method and use the right relaxation rate [i]
                solid_collision = psm_config.individual_fraction_field.center(p) * (
                    (eqSolid - f) + (1.0 - method.relaxation_rates[0]) * (f - eqFluid)
                )
            elif psm_config.SC == 3:
                solid_collision = psm_config.individual_fraction_field.center(p) * (
                    (
                        pre_collision_pdf_symbols[inverse_direction_index]
                        - equilibrium_solid[inverse_direction_index]
                    )
                    - (f - eqSolid)
                )
            else:
                raise ValueError("Only SC=1, SC=2 and SC=3 are supported.")
            solid_collisions[i] += solid_collision
            for j in range(stencil.D):
                forces_rhs[p * stencil.D + j] -= solid_collision * int(offset[j])

    # Add solid collision to main assignments of collision rule
    collision_assignments = []
    for main, sc in zip(collision_rule.main_assignments, solid_collisions):
        collision_assignments.append(Assignment(main.lhs, main.rhs + sc))

    # Add hydrodynamic force calculations to collision assignments if two-way coupling is used
    # (i.e., force field is not None)
    if psm_config.particle_force_field is not None:
        for p in range(psm_config.MaxParticlesPerCell):
            for i in range(stencil.D):
                collision_assignments.append(
                    Assignment(
                        psm_config.particle_force_field.center(p * stencil.D + i),
                        forces_rhs[p * stencil.D + i],
                    )
                )

    collision_assignments = AssignmentCollection(collision_assignments)
    ac = LbmCollisionRule(method, collision_assignments, collision_rule.subexpressions,
                          collision_rule.simplification_hints)
    ac.topological_sort()
    return ac
