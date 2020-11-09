import numpy as np
import pytest

from lbmpy.boundaries import UBB, NeumannByCopy, NoSlip, StreamInConstant
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.creationfunctions import create_lb_function
from lbmpy.geometry import add_box_boundary
from lbmpy.lbstep import LatticeBoltzmannStep
from pystencils import create_data_handling, make_slice


@pytest.mark.parametrize("target", ['cpu', 'gpu', 'opencl'])
def test_simple(target):
    if target == 'gpu':
        import pytest
        pytest.importorskip('pycuda')
    elif target == 'opencl':
        import pytest
        pytest.importorskip('pyopencl')
        import pystencils.opencl.autoinit

    dh = create_data_handling((4, 4), parallel=False, default_target=target)
    dh.add_array('pdfs', values_per_cell=9, cpu=True, gpu=target != 'cpu')
    for i in range(9):
        dh.fill("pdfs", i, value_idx=i, ghost_layers=True)

    if target == 'gpu' or target == 'opencl':
        dh.all_to_gpu()

    lb_func = create_lb_function(stencil='D2Q9',
                                 compressible=False,
                                 relaxation_rate=1.8,
                                 optimization={'target': target})

    bh = LatticeBoltzmannBoundaryHandling(lb_func.method, dh, 'pdfs', target=target)

    wall = NoSlip()
    moving_wall = UBB((1, 0))
    bh.set_boundary(wall, make_slice[0, :])
    bh.set_boundary(wall, make_slice[-1, :])
    bh.set_boundary(wall, make_slice[:, 0])
    bh.set_boundary(moving_wall, make_slice[:, -1])

    bh.prepare()
    bh()

    if target == 'gpu' or target == 'opencl':
        dh.all_to_cpu()
    # left lower corner
    assert (dh.cpu_arrays['pdfs'][0, 0, 6] == 7)

    assert (dh.cpu_arrays['pdfs'][0, 1, 4] == 3)
    assert (dh.cpu_arrays['pdfs'][0, 1, 6] == 7)

    assert (dh.cpu_arrays['pdfs'][1, 0, 1] == 2)
    assert (dh.cpu_arrays['pdfs'][1, 0, 6] == 7)

    # left side
    assert (all(dh.cpu_arrays['pdfs'][0, 2:4, 4] == 3))
    assert (all(dh.cpu_arrays['pdfs'][0, 2:4, 6] == 7))
    assert (all(dh.cpu_arrays['pdfs'][0, 2:4, 5] == 5))

    # left upper corner
    assert (dh.cpu_arrays['pdfs'][0, 4, 4] == 3)
    assert (dh.cpu_arrays['pdfs'][0, 4, 8] == 5)

    assert (dh.cpu_arrays['pdfs'][0, 5, 8] == 5 + 6 / 36)

    assert (dh.cpu_arrays['pdfs'][1, 5, 8] == 5 + 6 / 36)
    assert (dh.cpu_arrays['pdfs'][1, 5, 2] == 1)

    # top side
    assert (all(dh.cpu_arrays['pdfs'][2:4, 5, 2] == 1))
    assert (all(dh.cpu_arrays['pdfs'][2:4, 5, 7] == 6 - 6 / 36))
    assert (all(dh.cpu_arrays['pdfs'][2:4, 5, 8] == 5 + 6 / 36))

    # right upper corner
    assert (dh.cpu_arrays['pdfs'][4, 5, 2] == 1)
    assert (dh.cpu_arrays['pdfs'][4, 5, 7] == 6 - 6 / 36)

    assert (dh.cpu_arrays['pdfs'][5, 5, 7] == 6 - 6 / 36)

    assert (dh.cpu_arrays['pdfs'][5, 4, 3] == 4)
    assert (dh.cpu_arrays['pdfs'][5, 4, 7] == 6)

    # right side
    assert (all(dh.cpu_arrays['pdfs'][5, 2:4, 3] == 4))
    assert (all(dh.cpu_arrays['pdfs'][5, 2:4, 5] == 8))
    assert (all(dh.cpu_arrays['pdfs'][5, 2:4, 7] == 6))

    # right lower corner
    assert (dh.cpu_arrays['pdfs'][5, 1, 3] == 4)
    assert (dh.cpu_arrays['pdfs'][5, 1, 5] == 8)

    assert (dh.cpu_arrays['pdfs'][5, 0, 5] == 8)

    assert (dh.cpu_arrays['pdfs'][4, 0, 1] == 2)
    assert (dh.cpu_arrays['pdfs'][4, 0, 5] == 8)

    # lower side
    assert (all(dh.cpu_arrays['pdfs'][0, 2:4, 4] == 3))
    assert (all(dh.cpu_arrays['pdfs'][0, 2:4, 6] == 7))
    assert (all(dh.cpu_arrays['pdfs'][0, 2:4, 8] == 5))


def test_exotic_boundaries():
    step = LatticeBoltzmannStep((50, 50), relaxation_rate=1.8, compressible=False, periodicity=False)
    add_box_boundary(step.boundary_handling, NeumannByCopy())
    step.boundary_handling.set_boundary(StreamInConstant(0), make_slice[0, :])
    step.run(100)
    assert np.max(step.velocity[:, :, :]) < 1e-13
