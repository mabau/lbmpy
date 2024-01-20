# -*- coding: utf-8 -*-
"""Force can be computed from pressure tensor or directly from φ and μ. The pressure tensor version is numerically
more stable. Mathematically they should be equivalent.
This test ensures that the complicated pressure tensor formulation yields the same force as computed directly."""

import sympy as sp

from lbmpy.phasefield.analytical import (
    chemical_potentials_from_free_energy, force_from_phi_and_mu, force_from_pressure_tensor,
    pressure_tensor_interface_component_new, substitute_laplacian_by_sum)
from pystencils.fd import Diff, expand_diff_full, normalize_diff_order


def force_computation_equivalence(dim=3, num_phases=4):

    def Λ(i, j):
        if i > j:
            i, j = j, i
        return sp.Symbol("Lambda_{}{}".format(i, j))

    φ = sp.symbols("φ_:{}".format(num_phases))
    f_if = sum(Λ(α, β) / 2 * Diff(φ[α]) * Diff(φ[β])
               for α in range(num_phases) for β in range(num_phases))
    μ = chemical_potentials_from_free_energy(f_if, order_parameters=φ)
    μ = substitute_laplacian_by_sum(μ, dim=dim)

    p = sp.Matrix(dim, dim,
                  lambda i, j: pressure_tensor_interface_component_new(f_if, φ, dim, i, j))
    force_from_p = force_from_pressure_tensor(p, functions=φ)

    for d in range(dim):
        t1 = normalize_diff_order(force_from_p[d])
        t2 = expand_diff_full(force_from_phi_and_mu(φ, dim=dim, mu=μ)[d], functions=φ).expand()
        assert t1 - t2 == 0
    print("Success")


if __name__ == '__main__':
    force_computation_equivalence()
