import sympy as sp


def eos_from_free_energy(free_energy, density):
    """Compute equation of state from free energy"""
    chemical_potential = sp.diff(free_energy, density)
    return density * chemical_potential - free_energy


def free_energy_from_eos(eos, density, integration_constant):
    """Compute free energy from equation of state by integration
    Args:
        eos: equation of state
        density: symbolic! density parameter
        integration_constant:
    """
    return (sp.integrate(eos / (density**2), density) + integration_constant) * density


# ---------------------------------- Equations of state ----------------------------------------------------------------


def maxwell_construction(eos, tolerance=1e-4):
    """Numerical Maxwell construction to find ρ_gas and ρ_liquid for a given equation of state.

    Args:
        eos: equation of state, that has only one symbol (the density) in it
        tolerance: internally a bisection algorithm is used to find pressure such that areas below and
                   above are equal. The tolerance parameter refers to the pressure. If the integral is smaller than
                   the tolerance the bisection algorithm is stopped.

    Returns:
        (gas density, liquid density)
    """
    # The following solve and integrate calls can make use of the fact that the density is a positive, real number
    dofs = eos.atoms(sp.Symbol)
    assert len(dofs) == 1
    density = dofs.pop()
    rho = sp.Dummy(real=True, positive=True)
    eos = eos.subs(density, rho)

    # pre-compute integral once - then it is evaluated in every bisection iteration
    symbolic_offset = sp.Dummy(real=True)
    integral = sp.integrate(sp.nsimplify((eos - symbolic_offset) / (rho ** 2)), rho)
    upper_bound, lower_bound = sp.Dummy(real=True), sp.Dummy(real=True)
    symbolic_deviation = integral.subs(rho, upper_bound) - integral.subs(rho, lower_bound)
    get_deviation = sp.lambdify((lower_bound, upper_bound, symbolic_offset), symbolic_deviation)

    critical_points = sorted(sp.solve(sp.diff(eos, rho)))
    max_rho, min_rho, _ = critical_points
    max_p, min_p = eos.subs(rho, max_rho), eos.subs(rho, min_rho)
    shift_max = max_p * 0.999
    shift_min = max(0, min_p)
    
    c = (shift_max + shift_min) / 2
    deviation = tolerance * 2
    while abs(deviation) > tolerance:
        print("Deviation", deviation, "Shift", c)
        zeros = sp.solve(eos - c)
        integral_bounds = (min(zeros), max(zeros))
        deviation = get_deviation(float(integral_bounds[0]), float(integral_bounds[1]), float(c))
        if deviation > 0:
            shift_min = c
        else:
            shift_max = c
        c = (shift_max + shift_min) / 2

    return integral_bounds


# To get final free energy:
# - from maxwell construciton $\rho_{min}$ and $\rho_{max}$
# - remove slope from free energy function: C determined by $C = - \frac{d}{dρ} F(C=0)  $
# - energy shift = $F(ρ_{liquid})$  or $F(ρ_{gas})$ (should be equal)
# - final free energy := $F - F(ρ_{liquid})$


def carnahan_starling_eos(density, gas_constant, temperature, a, b):
    """Carnahan Starling equation of state.

    a, b are parameters specific to this equation of state
    for details see: Equations of state in a lattice Boltzmann model, by Yuan and Schaefer, 2006
    """
    e = b * density / 4
    fraction = (1 + e + e**2 - e**3) / (1 - e)**3
    return density * gas_constant * temperature * fraction - a * density ** 2


def carnahan_starling_critical_temperature(a, b, gas_constant):
    return 0.3773 * a / b / gas_constant
