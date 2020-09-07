from pystencils.session import *
from lbmpy.session import *
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
import lbmpy.forcemodels
from lbmpy.moments import is_bulk_moment

import pytest
from contextlib import ExitStack as does_not_raise


force_models = [fm.lower() for fm in dir(lbmpy.forcemodels) if fm[0].isupper()]

def test_force_models_available():
    assert 'guo' in force_models
    assert 'luo' in force_models

@pytest.mark.parametrize("method", ["srt", "trt"])
@pytest.mark.parametrize("force_model", force_models)
@pytest.mark.parametrize("omega", [0.5, 1.5])
def test_total_momentum(method, force_model, omega):
    L = (16, 16)
    stencil = get_stencil("D2Q9")
    F = [2e-4, -3e-4]

    dh = ps.create_data_handling(L, periodicity=True, default_target='cpu')
    src = dh.add_array('src', values_per_cell=len(stencil))
    dst = dh.add_array_like('dst', 'src')
    ρ = dh.add_array('rho')
    u = dh.add_array('u', values_per_cell=dh.dim)

    expectation = does_not_raise()
    skip = False
    if force_model in ['guo', 'buick'] and method != 'srt':
        expectation = pytest.raises(AssertionError)
        skip = True
    with expectation:
        collision = create_lb_update_rule(method=method,
                                          stencil=stencil,
                                          relaxation_rate=omega, 
                                          compressible=True,
                                          force_model=force_model, 
                                          force=F,
                                          kernel_type='collide_only',
                                          optimization={'symbolic_field': src})
    if skip:
        return

    stream = create_stream_pull_with_output_kernel(collision.method, src, dst,
                                                   {'density': ρ, 'velocity': u})

    opts = {'cpu_openmp': True, 
            'cpu_vectorize_info': None,
            'target': dh.default_target}

    stream_kernel = ps.create_kernel(stream, **opts).compile()
    collision_kernel = ps.create_kernel(collision, **opts).compile()

    def init():
        dh.fill(ρ.name, 1)
        dh.fill(u.name, 0)

        setter = macroscopic_values_setter(collision.method, velocity=(0,)*dh.dim, 
                                       pdfs=src.center_vector, density=ρ.center)
        kernel = ps.create_kernel(setter, ghost_layers=0).compile()
        dh.run_kernel(kernel)

    sync_pdfs = dh.synchronization_function([src.name])
    def time_loop(steps):
        dh.all_to_gpu()
        for i in range(steps):
            dh.run_kernel(collision_kernel)
            sync_pdfs()
            dh.run_kernel(stream_kernel)
            dh.swap(src.name, dst.name)
        dh.all_to_cpu()

    t = 20
    init()
    time_loop(t)
    total = np.sum(dh.gather_array(u.name), axis=(0,1))
    assert np.allclose(total/np.prod(L)/F/t, 1)


@pytest.mark.parametrize("stencil", ["D2Q9", "D3Q15", "D3Q19", "D3Q27"])
@pytest.mark.parametrize("force_model", ["simple", "schiller"])
def test_modes(stencil, force_model):
    """check Schiller's force term in mode space"""
    stencil = get_stencil(stencil)
    dim = len(stencil[0])
    
    omega_s = sp.Symbol("omega_s")
    omega_b = sp.Symbol("omega_b")
    omega_o = sp.Symbol("omega_o")
    omega_e = sp.Symbol("omega_e")
    
    F = [sp.Symbol(f"F_{i}") for i in range(dim)]
    
    method = create_lb_method(method="mrt", weighted=True,
                              stencil=stencil,
                              relaxation_rates=[omega_s, omega_b, omega_o, omega_e, omega_o, omega_e], 
                              compressible=True,
                              force_model=force_model,
                              force=F)
    
    force_moments = sp.simplify(method.moment_matrix * sp.Matrix(method.force_model(method)))
    
    # The mass mode should be zero
    assert force_moments[0] == 0
    
    # The momentum moments should contain the force
    assert list(force_moments[1:dim+1]) == F
    
    if force_model == "schiller":
        num_stresses = (dim*dim-dim)//2+dim
        lambda_s, lambda_b = -omega_s, -omega_b

        # The stress moments should match eq. 47 from https://doi.org/10.1023/A:1010414013942
        u = method.first_order_equilibrium_moment_symbols
        def traceless(m):
            tr = sp.simplify(sp.Trace(m))
            return m - tr/m.shape[0]*sp.eye(m.shape[0])
        C = sp.Rational(1,2) * (2 + lambda_s) * (traceless(sp.Matrix(u) * sp.Matrix(F).transpose()) + \
                                                traceless(sp.Matrix(F) * sp.Matrix(u).transpose())) + \
            sp.Rational(1,method.dim) * (2 + lambda_b) * sp.Matrix(u).dot(F) * sp.eye(method.dim)

        subs = {sp.Symbol(chr(ord("x")+i)) * sp.Symbol(chr(ord("x")+j)) : C[i,j]
                for i in range(dim) for j in range(dim)}
        for force_moment, moment in zip(force_moments[dim+1:dim+1+num_stresses],
                                        method.moments[dim+1:dim+1+num_stresses]):
            ref = moment.subs(subs)
            diff = sp.simplify(ref - force_moment)
            if is_bulk_moment(moment, dim):
                assert diff == 0 or isinstance(diff, sp.Rational) # difference should be zero or a constant
            else:
                assert diff == 0 # difference should be zero

        ff = sp.Matrix(method.force_model(method))
        # Check eq. 4.53a from schiller2008thermal
        assert sp.simplify(sum(ff)) == 0
        # Check eq. 4.53b from schiller2008thermal
        assert [sp.simplify(sum(ff[i] * stencil[i][j] for i in range(len(stencil)))) for j in range(dim)] == F
        # Check eq. 4.61a from schiller2008thermal
        ref = (2 + lambda_s)/2 * (traceless(sp.Matrix(u) * sp.Matrix(F).transpose()) + \
                                 traceless(sp.Matrix(F) * sp.Matrix(u).transpose()))
        s = sp.zeros(dim)
        for i in range(0, len(stencil)):
            s += ff[i] * traceless(sp.Matrix(stencil[i]) * sp.Matrix(stencil[i]).transpose())
        assert sp.simplify(s-ref) == sp.zeros(dim)
        # Check eq. 4.61b from schiller2008thermal
        assert sp.simplify(sum(ff[i] * stencil[i][a]**2 for i in range(len(stencil)) for a in range(dim))
                           - (2+lambda_b)*sp.Matrix(u).dot(F)) == 0

        # All other moments should be zero
        assert list(force_moments[dim+1+num_stresses:]) == [0] * (len(stencil)-(dim+1+num_stresses))
    elif force_model == "simple":
        # All other moments should be zero
        assert list(force_moments[dim+1:]) == [0] * (len(stencil)-(dim+1))
