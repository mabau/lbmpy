import sympy as sp

from lbmpy.enums import Stencil
from lbmpy.moments import MOMENT_SYMBOLS, sort_moments_into_groups_of_same_order
from lbmpy.stencils import LBStencil
from pystencils.stencil import have_same_entries


def cascaded_moment_sets_literature(stencil):
    """
    Returns default groups of central moments or cumulants to be relaxed with common relaxation rates 
    as stated in literature.
    Groups are ordered like this:

    - First group is density
    - Second group are the momentum modes
    - Third group are the shear modes
    - Fourth group is the bulk mode
    - Remaining groups do not govern hydrodynamic properties

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`. Can be D2Q9, D3Q7, D3Q15, D3Q19 or D3Q27
    """
    x, y, z = MOMENT_SYMBOLS
    if have_same_entries(stencil, LBStencil(Stencil.D2Q9)):
        #   Cumulants of the D2Q9 stencil up to third order are equal to
        #   the central moments; only the fourth-order cumulant x**2 * y**2
        #   has a more complicated form. They can be arranged into groups
        #   for the preservation of rotational invariance as described by
        #   Martin Geier in his dissertation.
        #
        #   Reference: Martin Geier. Ab inito derivation of the cascaded Lattice Boltzmann
        #   Automaton. Dissertation. University of Freiburg. 2006.
        return [
            [sp.sympify(1)],  # density is conserved
            [x, y],  # momentum is relaxed for cumulant forcing

            [x * y, x ** 2 - y ** 2],  # shear

            [x ** 2 + y ** 2],  # bulk

            [x ** 2 * y, x * y ** 2],
            [x ** 2 * y ** 2]
        ]

    elif have_same_entries(stencil, LBStencil(Stencil.D3Q7)):
        # D3Q7 moments: https://arxiv.org/ftp/arxiv/papers/1611/1611.03329.pdf
        return [
            [sp.sympify(1)],  # density is conserved
            [x, y, z],  # momentum might be affected by forcing

            [x ** 2 - y ** 2,
             x ** 2 - z ** 2],  # shear

            [x ** 2 + y ** 2 + z ** 2],  # bulk
        ]

    elif have_same_entries(stencil, LBStencil(Stencil.D3Q15)):
        #   D3Q15 central moments by Premnath et al. https://arxiv.org/pdf/1202.6081.pdf.
        return [
            [sp.sympify(1)],  # density is conserved
            [x, y, z],  # momentum might be affected by forcing

            [x * y,
             x * z,
             y * z,
             x ** 2 - y ** 2,
             x ** 2 - z ** 2],  # shear

            [x ** 2 + y ** 2 + z ** 2],  # bulk

            [x * (x ** 2 + y ** 2 + z ** 2),
             y * (x ** 2 + y ** 2 + z ** 2),
             z * (x ** 2 + y ** 2 + z ** 2)],

            [x * y * z],

            [x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2]
        ]

    elif have_same_entries(stencil, LBStencil(Stencil.D3Q19)):
        #   D3Q19 cumulants are obtained by pruning the D3Q27 cumulant set as
        #   described by Coreixas, 2019.
        return [
            [sp.sympify(1)],  # density is conserved
            [x, y, z],  # momentum might be affected by forcing

            [x * y,
             x * z,
             y * z,
             x ** 2 - y ** 2,
             x ** 2 - z ** 2],  # shear

            [x ** 2 + y ** 2 + z ** 2],  # bulk

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

    elif have_same_entries(stencil, LBStencil(Stencil.D3Q27)):
        #   Cumulants grouped to preserve rotational invariance as described by Geier et al, 2015
        return [
            [sp.sympify(1)],  # density is conserved
            [x, y, z],  # momentum might be affected by forcing

            [x * y,
             x * z,
             y * z,
             x ** 2 - y ** 2,
             x ** 2 - z ** 2],  # shear

            [x ** 2 + y ** 2 + z ** 2],  # bulk

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
        raise ValueError("No default set of cascaded moments is available for this stencil. "
                         "Please specify your own set of cascaded moments.")


def mrt_orthogonal_modes_literature(stencil, is_weighted):
    """
    Returns a list of lists of modes, grouped by common relaxation times.
    This is for commonly used MRT models found in literature.

    Args:
        stencil: instance of :class:`lbmpy.stencils.LBStencil`. Can be D2Q9, D3Q15, D3Q19 or D3Q27
        is_weighted: whether to use weighted or unweighted orthogonality

    MRT schemes as described in the following references are used
    """
    x, y, z = MOMENT_SYMBOLS
    one = sp.Rational(1, 1)

    if have_same_entries(stencil, LBStencil(Stencil.D2Q9)) and is_weighted:
        # Reference:
        # Duenweg, B., Schiller, U. D., & Ladd, A. J. (2007). Statistical mechanics of the fluctuating
        # lattice Boltzmann equation. Physical Review E, 76(3)
        sq = x ** 2 + y ** 2
        all_moments = [one, x, y, 3 * sq - 2, 2 * x ** 2 - sq, x * y,
                       (3 * sq - 4) * x, (3 * sq - 4) * y, 9 * sq ** 2 - 15 * sq + 2]
        nested_moments = list(sort_moments_into_groups_of_same_order(all_moments).values())
        return nested_moments
    elif have_same_entries(stencil, LBStencil(Stencil.D3Q15)) and is_weighted:
        sq = x ** 2 + y ** 2 + z ** 2
        nested_moments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 9 * sq + 4],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 10, 11, 12, 13]
            [x * y * z]
        ]
    elif have_same_entries(stencil, LBStencil(Stencil.D3Q19)) and is_weighted:
        # This MRT variant mentioned in the dissertation of Ulf Schiller
        # "Thermal fluctuations and boundary conditions in the lattice Boltzmann method" (2008), p. 24ff
        # There are some typos in the moment matrix on p.27
        # The here implemented ordering of the moments is however different from that reference (Eq. 2.61-2.63)
        # The moments are weighted-orthogonal (Eq. 2.58)

        # Further references:
        # Duenweg, B., Schiller, U. D., & Ladd, A. J. (2007). Statistical mechanics of the fluctuating
        # lattice Boltzmann equation. Physical Review E, 76(3)
        # Chun, B., & Ladd, A. J. (2007). Interpolated boundary condition for lattice Boltzmann simulations of
        # flows in narrow gaps. Physical review E, 75(6)
        sq = x ** 2 + y ** 2 + z ** 2
        nested_moments = [
            [one, x, y, z],  # [0, 3, 5, 7]
            [sq - 1],  # [1]
            [3 * sq ** 2 - 6 * sq + 1],  # [2]
            [(3 * sq - 5) * x, (3 * sq - 5) * y, (3 * sq - 5) * z],  # [4, 6, 8]
            [3 * x ** 2 - sq, y ** 2 - z ** 2, x * y, y * z, x * z],  # [9, 11, 13, 14, 15]
            [(2 * sq - 3) * (3 * x ** 2 - sq), (2 * sq - 3) * (y ** 2 - z ** 2)],  # [10, 12]
            [(y ** 2 - z ** 2) * x, (z ** 2 - x ** 2) * y, (x ** 2 - y ** 2) * z]  # [16, 17, 18]
        ]
    elif have_same_entries(stencil, LBStencil(Stencil.D3Q27)) and not is_weighted:
        xsq, ysq, zsq = x ** 2, y ** 2, z ** 2
        all_moments = [
            sp.Rational(1, 1),  # 0
            x, y, z,  # 1, 2, 3
            x * y, x * z, y * z,  # 4, 5, 6
            xsq - ysq,  # 7
            (xsq + ysq + zsq) - 3 * zsq,  # 8
            (xsq + ysq + zsq) - 2,  # 9
            3 * (x * ysq + x * zsq) - 4 * x,  # 10
            3 * (xsq * y + y * zsq) - 4 * y,  # 11
            3 * (xsq * z + ysq * z) - 4 * z,  # 12
            x * ysq - x * zsq,  # 13
            xsq * y - y * zsq,  # 14
            xsq * z - ysq * z,  # 15
            x * y * z,  # 16
            3 * (xsq * ysq + xsq * zsq + ysq * zsq) - 4 * (xsq + ysq + zsq) + 4,  # 17
            3 * (xsq * ysq + xsq * zsq - 2 * ysq * zsq) - 2 * (2 * xsq - ysq - zsq),  # 18
            3 * (xsq * ysq - xsq * zsq) - 2 * (ysq - zsq),  # 19
            3 * (xsq * y * z) - 2 * (y * z),  # 20
            3 * (x * ysq * z) - 2 * (x * z),  # 21
            3 * (x * y * zsq) - 2 * (x * y),  # 22
            9 * (x * ysq * zsq) - 6 * (x * ysq + x * zsq) + 4 * x,  # 23
            9 * (xsq * y * zsq) - 6 * (xsq * y + y * zsq) + 4 * y,  # 24
            9 * (xsq * ysq * z) - 6 * (xsq * z + ysq * z) + 4 * z,  # 25
            27 * (xsq * ysq * zsq) - 18 * (xsq * ysq + xsq * zsq + ysq * zsq) + 12 * (xsq + ysq + zsq) - 8,  # 26
        ]
        nested_moments = list(sort_moments_into_groups_of_same_order(all_moments).values())
    else:
        raise NotImplementedError("No MRT model is available (yet) for this stencil. "
                                  "Create a custom MRT using 'create_with_discrete_maxwellian_equilibrium'")

    return nested_moments
