import sympy as sp

from pystencils.cache import disk_cache
# ---------------------------------- Equations of state ----------------------------------------------------------------
from pystencils.sympyextensions import remove_small_floats


def carnahan_starling_eos(density, gas_constant, temperature, a, b):
    """Carnahan Starling equation of state.

    a, b are parameters specific to this equation of state
    for details see: Equations of state in a lattice Boltzmann model, by Yuan and Schaefer, 2006
    """
    e = b * density / 4
    fraction = (1 + e + e ** 2 - e ** 3) / (1 - e) ** 3
    return density * gas_constant * temperature * fraction - a * density ** 2


def carnahan_starling_critical_temperature(a, b, gas_constant):
    return 0.3773 * a / b / gas_constant


def van_der_walls_eos(density, gas_constant, temperature, a, b):
    pressure = sp.Symbol("P")
    vdw = sp.Equality((pressure + a * density ** 2) * (1 / density - b), gas_constant * temperature)
    return sp.solve(vdw, pressure)[0]


def van_der_walls_critical_temperature(a, b, gas_constant):
    return 8 * a / 27 / b / gas_constant


# ----------------------------- Functions operating on equation of states ----------------------------------------------


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
    return (sp.integrate(eos / (density ** 2), density) + integration_constant) * density


@disk_cache
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
    dofs = eos.atoms(sp.Symbol)
    assert len(dofs) == 1
    density = dofs.pop()
    v = sp.Dummy()
    eos = eos.subs(density, 1 / v)

    # pre-compute integral once - then it is evaluated in every bisection iteration
    symbolic_offset = sp.Dummy(real=True)
    integral = sp.integrate(sp.nsimplify(eos + symbolic_offset), v)
    upper_bound, lower_bound = sp.Dummy(real=True), sp.Dummy(real=True)
    symbolic_deviation = integral.subs(v, upper_bound) - integral.subs(v, lower_bound)
    get_deviation = sp.lambdify((lower_bound, upper_bound, symbolic_offset), symbolic_deviation)

    critical_points = sp.solve(sp.diff(eos, v))
    critical_points = [remove_small_floats(e, 1e-14) for e in critical_points]
    critical_points = [e for e in critical_points if e.is_real]
    *_, v_min, v_max = critical_points

    assert sp.diff(eos, v, v).subs(v, v_min) > 0
    assert sp.diff(eos, v, v).subs(v, v_max) < 0

    assert sp.limit(eos, v, sp.oo) == 0
    # shift has to be negative, since equation of state approaches zero for v -> oo
    shift_min, shift_max = -eos.subs(v, v_max), 0
    c = (shift_max + shift_min) / 2
    deviation = tolerance * 2

    while abs(deviation) > tolerance:
        solve_res = sp.solve(eos + c)
        solve_res = [remove_small_floats(e, 1e-14) for e in solve_res]
        zeros = sorted([e for e in solve_res if e.is_real])
        integral_bounds = (zeros[-3], zeros[-1])
        deviation = get_deviation(float(integral_bounds[0]), float(integral_bounds[1]), float(c))
        if deviation > 0:
            shift_max = c
        else:
            shift_min = c
        c = (shift_max + shift_min) / 2

    return 1 / integral_bounds[1], 1 / integral_bounds[0]
