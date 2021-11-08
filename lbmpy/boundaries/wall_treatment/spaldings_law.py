import sympy as sp


def spaldings_law(u_plus, y_plus, kappa=0.41, B=5.5):
    """
    Returns a symbolic expression for spaldings law

    Args:
        u_plus: velocity nondimensionalized by the friction velocity u_tau
        y_plus: distances from the wall nondimensionalized by the friction velocity u_tau
        kappa: free parameter
        B: free parameter
    """
    k_times_u = kappa * u_plus
    fraction_1 = (k_times_u ** 2) / 2
    fraction_2 = (k_times_u ** 3) / 6
    return u_plus + sp.exp(-kappa * B) * (sp.exp(k_times_u) - 1 - k_times_u - fraction_1 - fraction_2) - y_plus
