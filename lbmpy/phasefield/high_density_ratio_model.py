import sympy as sp

from lbmpy.phasefield.eos import free_energy_from_eos
from pystencils.fd import Diff
from pystencils.sympyextensions import remove_small_floats


def free_energy_high_density_ratio(eos, density, density_gas, density_liquid, c_liquid_1, c_liquid_2, lambdas, kappas):
    """ Free energy for a ternary system with high density ratio :cite:`Wohrwag2018`

    Args:
        eos: equation of state, has to depend on exactly on symbol, the density
        density: symbol for density
        density_gas: numeric value for gas density (can be obtained by `maxwell_construction`)
        density_liquid: numeric value for liquid density (can be obtained by `maxwell_construction`)
        c_liquid_1: symbol for concentration of first liquid phase
        c_liquid_2: symbol for concentration of second liquid phase
        lambdas: pre-factors of bulk terms, lambdas[0] multiplies the density term, lambdas[1] the first liquid and
                 lambdas[2] the second liquid phase
        kappas: pre-factors of interfacial terms, order same as for lambdas

    Returns:
        free energy expression
    """
    assert eos.atoms(sp.Symbol) == {density}
    # ---- Part 1: contribution of equation of state, ψ_eos
    symbolic_integration_constant = sp.Dummy(real=True)
    psi_eos = free_energy_from_eos(eos, density, symbolic_integration_constant)
    # accuracy problems in free_energy_from_eos can lead to complex solutions for integration constant
    psi_eos = remove_small_floats(psi_eos, 1e-14)

    # integration constant is determined from the condition ψ(ρ_gas) == ψ(ρ_liquid)
    equal_psi_condition = psi_eos.subs(density, density_gas) - psi_eos.subs(density, density_liquid)
    solve_res = sp.solve(equal_psi_condition, symbolic_integration_constant)
    assert len(solve_res) == 1
    integration_constant = solve_res[0]
    psi_eos = psi_eos.subs(symbolic_integration_constant, integration_constant)

    # energy is shifted by ψ_0 = ψ(ρ_gas) which is also ψ(ρ_liquid) by construction
    psi_0 = psi_eos.subs(density, density_gas)

    # ---- Part 2: standard double well potential as bulk term, and gradient squares as interface term
    def f(c):
        return c ** 2 * (1 - c) ** 2

    f_bulk = (lambdas[0] / 2 * (psi_eos - psi_0)
              + lambdas[1] / 2 * f(c_liquid_1)
              + lambdas[2] / 2 * f(c_liquid_2))
    f_interface = (kappas[0] / 2 * Diff(density) ** 2
                   + kappas[1] / 2 * Diff(c_liquid_1) ** 2
                   + kappas[2] / 2 * Diff(c_liquid_2) ** 2)
    return f_bulk + f_interface
