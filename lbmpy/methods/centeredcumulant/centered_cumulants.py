from lbmpy.stencils import get_stencil
import sympy as sp

from pystencils.stencil import have_same_entries

from lbmpy.moments import MOMENT_SYMBOLS, moment_sort_key, exponent_to_polynomial_representation


def statistical_quantity_symbol(name, exponents):
    return sp.Symbol(f'{name}_{"".join(str(i) for i in exponents)}')


def exponent_tuple_sort_key(x):
    return moment_sort_key(exponent_to_polynomial_representation(x))


def get_default_polynomial_cumulants_for_stencil(stencil):
    """
    Returns default groups of cumulants to be relaxed with common relaxation rates as stated in literature.
    Groups are ordered like this:
     - First group is density
     - Second group are the momentum modes
     - Third group are the shear modes
     - Fourth group is the bulk mode
     - Remaining groups do not govern hydrodynamic properties
    """
    x, y, z = MOMENT_SYMBOLS
    if have_same_entries(stencil, get_stencil("D2Q9")):
        #   Cumulants of the D2Q9 stencil up to third order are equal to
        #   the central moments; only the fourth-order cumulant x**2 * y**2
        #   has a more complicated form. They can be arranged into groups
        #   for the preservation of rotational invariance as described by
        #   Martin Geier in his dissertation.
        #
        #   Reference: Martin Geier. Ab inito derivation of the cascaded Lattice Boltzmann
        #   Automaton. Dissertation. University of Freiburg. 2006.
        return [
            [sp.sympify(1)],        # density is conserved
            [x, y],                 # momentum is relaxed for cumulant forcing

            [x * y, x**2 - y**2],   # shear

            [x**2 + y**2],          # bulk

            [x**2 * y, x * y**2],
            [x**2 * y**2]
        ]

    elif have_same_entries(stencil, get_stencil("D3Q19")):
        #   D3Q19 cumulants are obtained by pruning the D3Q27 cumulant set as
        #   described by Coreixas, 2019.
        return [
            [sp.sympify(1)],                # density is conserved
            [x, y, z],                      # momentum might be affected by forcing

            [x * y,
             x * z,
             y * z,
             x ** 2 - y ** 2,
             x ** 2 - z ** 2],              # shear

            [x ** 2 + y ** 2 + z ** 2],     # bulk

            [x * y ** 2 + x * z ** 2,
             x ** 2 * y + y * z ** 2,
             x ** 2 * z + y ** 2 * z],

            [x * y ** 2 - x * z ** 2,
             x ** 2 * y - y * z ** 2,
             x ** 2 * z - y ** 2 * z],

            [x ** 2 * y ** 2 - 2 * x ** 2 * z ** 2 + y ** 2 * z ** 2,
             x ** 2 * y ** 2 + x ** 2 * z ** 2 - 2 * y ** 2 * z ** 2],

            [x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2]
        ]

    elif have_same_entries(stencil, get_stencil("D3Q27")):
        #   Cumulants grouped to preserve rotational invariance as described by Geier et al, 2015
        return [
            [sp.sympify(1)],                # density is conserved
            [x, y, z],                      # momentum might be affected by forcing

            [x * y,
             x * z,
             y * z,
             x ** 2 - y ** 2,
             x ** 2 - z ** 2],              # shear

            [x ** 2 + y ** 2 + z ** 2],     # bulk

            [x * y ** 2 + x * z ** 2,
             x ** 2 * y + y * z ** 2,
             x ** 2 * z + y ** 2 * z],

            [x * y ** 2 - x * z ** 2,
             x ** 2 * y - y * z ** 2,
             x ** 2 * z - y ** 2 * z],

            [x * y * z],

            [x ** 2 * y ** 2 - 2 * x ** 2 * z ** 2 + y ** 2 * z ** 2,
             x ** 2 * y ** 2 + x ** 2 * z ** 2 - 2 * y ** 2 * z ** 2],

            [x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2],

            [x ** 2 * y * z,
             x * y ** 2 * z,
             x * y * z ** 2],

            [x ** 2 * y ** 2 * z,
             x ** 2 * y * z ** 2,
             x * y ** 2 * z ** 2],

            [x ** 2 * y ** 2 * z ** 2]
        ]
    else:
        raise ValueError("No default set of cumulants is available for this stencil. "
                         "Please specify your own set of polynomial cumulants.")
