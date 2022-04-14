import numpy as np
import pytest
from types import MappingProxyType
from pystencils import Target, CreateKernelConfig

from lbmpy.scenarios import create_fully_periodic_flow, create_lid_driven_cavity

try:
    import pycuda.driver
    gpu_available = True
except ImportError:
    gpu_available = False

try:
    import waLBerla as wLB
    parallel_available = wLB.cpp_available
except ImportError:
    parallel_available = False
    wLB = None


def ldc_setup(**kwargs):
    ldc = create_lid_driven_cavity(relaxation_rate=1.7, **kwargs)
    ldc.run(50)
    return ldc.density_slice()


def test_data_handling_3d():
    print("--- LDC 3D test ---")
    results = []
    for parallel in [False, True] if parallel_available else [False]:
        for gpu in [False, True] if gpu_available else [False]:
            if parallel and gpu and not hasattr(wLB, 'cuda'):
                continue
            print(f"Testing parallel: {parallel}\tgpu: {gpu}")
            config = CreateKernelConfig(target=Target.GPU if gpu else Target.CPU,
                                        gpu_indexing_params=MappingProxyType({'block_size': (8, 4, 2)}))
            if parallel:
                from pystencils.datahandling import ParallelDataHandling
                blocks = wLB.createUniformBlockGrid(blocks=(2, 3, 4), cellsPerBlock=(5, 5, 5),
                                                    oneBlockPerProcess=False)
                dh = ParallelDataHandling(blocks, dim=3)
                rho = ldc_setup(data_handling=dh, config=config)
                results.append(rho)
            else:
                rho = ldc_setup(domain_size=(10, 15, 20), parallel=False, config=config)
                results.append(rho)
    for i, arr in enumerate(results[1:]):
        print("Testing equivalence version 0 with version %d" % (i + 1,))
        np.testing.assert_almost_equal(results[0], arr)


def test_data_handling_2d():
    print("--- LDC 2D test ---")
    results = []
    for parallel in [True, False] if parallel_available else [False]:
        for gpu in [True, False] if gpu_available else [False]:
            if parallel and gpu and not hasattr(wLB, 'cuda'):
                continue

            print(f"Testing parallel: {parallel}\tgpu: {gpu}")
            config = CreateKernelConfig(target=Target.GPU if gpu else Target.CPU,
                                        gpu_indexing_params=MappingProxyType({'block_size': (8, 4, 2)}))
            if parallel:
                from pystencils.datahandling import ParallelDataHandling
                blocks = wLB.createUniformBlockGrid(blocks=(2, 3, 1), cellsPerBlock=(5, 5, 1),
                                                    oneBlockPerProcess=False)
                dh = ParallelDataHandling(blocks, dim=2)
                rho = ldc_setup(data_handling=dh, config=config)
                results.append(rho)
            else:
                rho = ldc_setup(domain_size=(10, 15), parallel=False, config=config)
                results.append(rho)
    for i, arr in enumerate(results[1:]):
        print(f"Testing equivalence version 0 with version {i + 1}")
        np.testing.assert_almost_equal(results[0], arr)


def test_smagorinsky_setup():
    step = create_lid_driven_cavity((30, 30), smagorinsky=0.16, relaxation_rate=1.99)
    step.run(10)


def test_advanced_initialization():
    width, height = 100, 50
    velocity_magnitude = 0.05
    init_vel = np.zeros((width, height, 2))
    # fluid moving to the right everywhere...
    init_vel[:, :, 0] = velocity_magnitude
    # ...except at a stripe in the middle, where it moves left
    init_vel[:, height // 3: height // 3 * 2, 0] = -velocity_magnitude
    # small random y velocity component
    init_vel[:, :, 1] = 0.1 * velocity_magnitude * np.random.rand(width, height)
    shear_flow_scenario = create_fully_periodic_flow(initial_velocity=init_vel, relaxation_rate=1.99)
    with pytest.raises(ValueError) as e:
        shear_flow_scenario.run_iterative_initialization(max_steps=20000, check_residuum_after=500)
    assert 'did not converge' in str(e.value)

    shear_flow_scenario = create_fully_periodic_flow(initial_velocity=init_vel, relaxation_rate=1.6)
    shear_flow_scenario.run_iterative_initialization(max_steps=20000, check_residuum_after=500)
