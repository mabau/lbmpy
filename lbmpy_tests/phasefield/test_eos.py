from lbmpy.phasefield.eos import *


def test_eos_to_free_energy_conversion():
    rho, r, t, a, b, c = sp.symbols("rho, r, t, a, b, c")
    eos = carnahan_starling_eos(rho, r, t, a, b)
    f = free_energy_from_eos(eos, rho, c)

    assert sp.simplify(eos_from_free_energy(f, rho) - eos) == 0


def test_maxwell_construction_cs():
    """Test of Maxwell construction routine with parameters from paper
    Ternary free-energy entropic lattice Boltzmann model with high density ratio
    """
    rho = sp.Symbol("rho")
    a = 0.037
    b = 0.2
    reduced_temperature = 0.69

    eos = carnahan_starling_eos(rho,
                                gas_constant=1,
                                temperature=carnahan_starling_critical_temperature(a, b, 1) * reduced_temperature,
                                a=a, b=b)
    r = maxwell_construction(eos, tolerance=1e-3)
    assert abs(r[0] - 0.17) < 0.01
    assert abs(r[1] - 7.26) < 0.01


def test_maxwell_construction_vw():
    """Test of Maxwell construction routine with parameters from paper
    Ternary free-energy entropic lattice Boltzmann model with high density ratio
    """
    rho = sp.Symbol("rho")
    a = 2. / 49.0
    b = 2.0 / 21.0
    reduced_temperature = 0.69

    eos = van_der_walls_eos(rho, gas_constant=1,
                            temperature=van_der_walls_critical_temperature(a, b, 1) * reduced_temperature,
                            a=a, b=b)
    r = maxwell_construction(eos, tolerance=1e-3)
    assert abs(r[0] - 0.419) < 0.01
    assert abs(r[1] - 7.556) < 0.01
