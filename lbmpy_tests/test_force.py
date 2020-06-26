from pystencils.session import *
from lbmpy.session import *
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
import lbmpy.forcemodels

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
