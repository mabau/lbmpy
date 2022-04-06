import sympy as sp
import pytest

from lbmpy.stencils import LBStencil, Stencil
from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.equilibrium import default_background_distribution
from lbmpy.moments import get_default_moment_set_for_stencil

from lbmpy.moment_transforms import (
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform,
    PdfsToCentralMomentsByMatrix, PdfsToCentralMomentsByShiftMatrix,
    FastCentralMomentTransform
)

transforms = [
    PdfsToMomentsByMatrixTransform, PdfsToMomentsByChimeraTransform,
    PdfsToCentralMomentsByMatrix, PdfsToCentralMomentsByShiftMatrix,
    FastCentralMomentTransform
]


def check_shifts(stencil, transform_class):
    weights = get_weights(stencil)
    bd = default_background_distribution(stencil.D)
    rho = bd.density
    u = bd.velocity
    moments = get_default_moment_set_for_stencil(stencil)
    fs = sp.symbols(f'f_:{stencil.Q}')
    gs = sp.symbols(f'g_:{stencil.Q}')

    transform_unshifted = transform_class(stencil=stencil,
                                          equilibrium_density=rho,
                                          equilibrium_velocity=u,
                                          moment_polynomials=moments)
    transform_shifted = transform_class(stencil=stencil,
                                        equilibrium_density=rho,
                                        equilibrium_velocity=u,
                                        moment_polynomials=moments,
                                        background_distribution=bd)

    #   Test forward transforms
    fw_unshifted = transform_unshifted.forward_transform(fs).new_without_subexpressions()
    fw_shifted = transform_shifted.forward_transform(gs).new_without_subexpressions()

    fw_delta = [(a.rhs - b.rhs).expand() for a, b in zip(fw_unshifted, fw_shifted)]
    fw_subs = {f: w for f, w in zip(fs, weights)}
    fw_subs.update({g: sp.Integer(0) for g in gs})
    fw_delta = [eq.subs(fw_subs).expand() for eq in fw_delta]
    for i, eq in enumerate(fw_delta):
        assert eq == sp.Integer(0), f"Error at index {i}"

    #   Test backward transforms
    bw_unshifted = transform_unshifted.backward_transform(fs).new_without_subexpressions()
    bw_shifted = transform_shifted.backward_transform(fs).new_without_subexpressions()
    bw_delta = [(a.rhs - b.rhs).expand() for a, b in zip(bw_unshifted, bw_shifted)]
    assert bw_delta == weights


@pytest.mark.parametrize('stencil', [LBStencil(Stencil.D2Q9)])
@pytest.mark.parametrize('transform_class', transforms)
def test_shifted_transform_fast(stencil, transform_class):
    check_shifts(stencil, transform_class)


@pytest.mark.longrun
@pytest.mark.parametrize('stencil', [LBStencil(Stencil.D3Q19), LBStencil(Stencil.D3Q27)])
@pytest.mark.parametrize('transform_class', transforms)
def test_shifted_transform_long(stencil, transform_class):
    check_shifts(stencil, transform_class)
