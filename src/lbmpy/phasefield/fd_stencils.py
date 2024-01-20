import sympy as sp

from pystencils.fd.spatial import fd_stencils_standard


def fd_stencils_isotropic_high_density_code(indices, dx, fa):
    dim = fa.field.spatial_dimensions
    if dim == 1:
        return fd_stencils_standard(indices, dx, fa)
    elif dim == 2:
        order = len(indices)

        if order == 1:
            idx = indices[0]
            assert 0 <= idx < 2
            other_idx = 1 if indices[0] == 0 else 0
            weights = {-1: sp.Rational(1, 12) / dx,
                       0: sp.Rational(1, 3) / dx,
                       1: sp.Rational(1, 12) / dx}
            upper_terms = sum(fa.neighbor(idx, +1).neighbor(other_idx, off) * w for off, w in weights.items())
            lower_terms = sum(fa.neighbor(idx, -1).neighbor(other_idx, off) * w for off, w in weights.items())
            return upper_terms - lower_terms
        elif order == 2:
            if indices[0] == indices[1]:
                idx = indices[0]
                diagonals = sp.Rational(1, 8) * sum(fa.neighbor(0, i).neighbor(1, j) for i in (-1, 1) for j in (-1, 1))
                div_direction = sp.Rational(1, 2) * sum(fa.neighbor(idx, i) for i in (-1, 1))
                center = - sp.Rational(3, 2) * fa
                return (diagonals + div_direction + center) / (dx ** 2)
            else:
                return fd_stencils_standard(indices, dx, fa)
    raise NotImplementedError("Supports only derivatives up to order 2 for 1D and 2D setups")
