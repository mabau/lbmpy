from lbmpy.methods import create_srt
from lbmpy.stencils import get_stencil
from pystencils.sympyextensions import remove_higher_order_terms


def test_weights():
    for stencil_name in ('D2Q9', 'D3Q19', 'D3Q27'):
        stencil = get_stencil(stencil_name)
        cumulant_method = create_srt(stencil, 1, cumulant=True, compressible=True,
                                     maxwellian_moments=True)
        moment_method = create_srt(stencil, 1, cumulant=False, compressible=True,
                                   maxwellian_moments=True)
        assert cumulant_method.weights == moment_method.weights


def test_equilibrium_equivalence():
    for stencil_name in ('D2Q9', 'D3Q19', 'D3Q27'):
        stencil = get_stencil(stencil_name)
        cumulant_method = create_srt(stencil, 1, cumulant=True, compressible=True,
                                     maxwellian_moments=True)
        moment_method = create_srt(stencil, 1, cumulant=False, compressible=True,
                                   maxwellian_moments=True)
        moment_eq = moment_method.get_equilibrium()
        cumulant_eq = cumulant_method.get_equilibrium()
        u = moment_method.first_order_equilibrium_moment_symbols
        for mom_eq, cum_eq in zip(moment_eq.main_assignments, cumulant_eq.main_assignments):
            diff = cum_eq.rhs - mom_eq.rhs
            assert remove_higher_order_terms(diff.expand(), order=2, symbols=u) == 0
