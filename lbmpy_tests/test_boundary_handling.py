import numpy as np
import pytest

from lbmpy.boundaries import NoSlip, UBB, SimpleExtrapolationOutflow, ExtrapolationOutflow, \
    FixedDensity, DiffusionDirichlet, NeumannByCopy, StreamInConstant, FreeSlip
from lbmpy.boundaries.boundaryhandling import LatticeBoltzmannBoundaryHandling
from lbmpy.creationfunctions import create_lb_function, create_lb_method, LBMConfig
from lbmpy.enums import Stencil, Method
from lbmpy.geometry import add_box_boundary
from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.stencils import LBStencil
from pystencils import create_data_handling, make_slice, Target, CreateKernelConfig
from pystencils.slicing import slice_from_direction
from pystencils.stencil import inverse_direction


def mirror_stencil(direction, mirror_axis):
    for i, n in enumerate(mirror_axis):
        if n != 0:
            direction[i] = -direction[i]

    return tuple(direction)


@pytest.mark.parametrize("target", [Target.GPU, Target.CPU])
def test_simple(target):
    if target == Target.GPU:
        import pytest
        pytest.importorskip('pycuda')

    dh = create_data_handling((4, 4), parallel=False, default_target=target)
    dh.add_array('pdfs', values_per_cell=9, cpu=True, gpu=target != Target.CPU)
    for i in range(9):
        dh.fill("pdfs", i, value_idx=i, ghost_layers=True)

    if target == Target.GPU:
        dh.all_to_gpu()

    lbm_config = LBMConfig(stencil=LBStencil(Stencil.D2Q9), compressible=False, zero_centered=False,
                           delta_equilibrium=False, relaxation_rate=1.8)
    config = CreateKernelConfig(target=target)

    lb_func = create_lb_function(lbm_config=lbm_config, config=config)

    bh = LatticeBoltzmannBoundaryHandling(lb_func.method, dh, 'pdfs', target=target)

    wall = NoSlip()
    moving_wall = UBB((1, 0))
    bh.set_boundary(wall, make_slice[0, :])
    bh.set_boundary(wall, make_slice[-1, :])
    bh.set_boundary(wall, make_slice[:, 0])
    bh.set_boundary(moving_wall, make_slice[:, -1])

    bh.prepare()
    bh()

    if target == Target.GPU:
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


@pytest.mark.parametrize("given_normal", [True, False])
def test_free_slip(given_normal):
    # check if Free slip BC is applied correctly

    stencil = LBStencil(Stencil.D2Q9)
    dh = create_data_handling(domain_size=(4, 4),)
    src1 = dh.add_array('src1', values_per_cell=stencil.Q)
    dh.fill('src1', 0.0, ghost_layers=True)

    shape = dh.gather_array('src1', ghost_layers=True).shape

    num = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for direction in range(shape[2]):
                dh.cpu_arrays[src1.name][x, y, direction] = num
                num += 1

    method = create_lb_method(lbm_config=LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.8))

    bh = LatticeBoltzmannBoundaryHandling(method, dh, 'src1', name="bh1")
    if given_normal:
        free_slipN = FreeSlip(stencil=stencil, normal_direction=(0, -1))
        free_slipS = FreeSlip(stencil=stencil, normal_direction=(0, 1))
        free_slipE = FreeSlip(stencil=stencil, normal_direction=(-1, 0))
        free_slipW = FreeSlip(stencil=stencil, normal_direction=(1, 0))

        bh.set_boundary(free_slipN, slice_from_direction('N', dh.dim))
        bh.set_boundary(free_slipS, slice_from_direction('S', dh.dim))
        bh.set_boundary(free_slipE, slice_from_direction('E', dh.dim))
        bh.set_boundary(free_slipW, slice_from_direction('W', dh.dim))
    else:
        free_slip = FreeSlip(stencil=stencil)

        bh.set_boundary(free_slip, slice_from_direction('N', dh.dim))
        bh.set_boundary(free_slip, slice_from_direction('S', dh.dim))
        bh.set_boundary(free_slip, slice_from_direction('E', dh.dim))
        bh.set_boundary(free_slip, slice_from_direction('W', dh.dim))

    bh()

    mirrored_dirN = {6: 8, 1: 2, 5: 7}
    mirrored_dirS = {7: 5, 2: 1, 8: 6}
    mirrored_dirE = {6: 5, 4: 3, 8: 7}
    mirrored_dirW = {5: 6, 3: 4, 7: 8}

    # check North
    assert dh.cpu_arrays[src1.name][1, -1, mirrored_dirN[6]] == dh.cpu_arrays[src1.name][1, -2, 6]
    assert dh.cpu_arrays[src1.name][1, -1, mirrored_dirN[1]] == dh.cpu_arrays[src1.name][1, -2, 1]

    for i in range(2, 4):
        assert dh.cpu_arrays[src1.name][i, -1, mirrored_dirN[6]] == dh.cpu_arrays[src1.name][i, -2, 6]
        assert dh.cpu_arrays[src1.name][i, -1, mirrored_dirN[1]] == dh.cpu_arrays[src1.name][i, -2, 1]
        assert dh.cpu_arrays[src1.name][i, -1, mirrored_dirN[5]] == dh.cpu_arrays[src1.name][i, -2, 5]

    assert dh.cpu_arrays[src1.name][4, -1, mirrored_dirN[1]] == dh.cpu_arrays[src1.name][4, -2, 1]
    assert dh.cpu_arrays[src1.name][4, -1, mirrored_dirN[5]] == dh.cpu_arrays[src1.name][4, -2, 5]

    # check East
    assert dh.cpu_arrays[src1.name][-1, 1, mirrored_dirE[6]] == dh.cpu_arrays[src1.name][-2, 1, 6]
    assert dh.cpu_arrays[src1.name][-1, 1, mirrored_dirE[4]] == dh.cpu_arrays[src1.name][-2, 1, 4]

    for i in range(2, 4):
        assert dh.cpu_arrays[src1.name][-1, i, mirrored_dirE[6]] == dh.cpu_arrays[src1.name][-2, i, 6]
        assert dh.cpu_arrays[src1.name][-1, i, mirrored_dirE[4]] == dh.cpu_arrays[src1.name][-2, i, 4]
        assert dh.cpu_arrays[src1.name][-1, i, mirrored_dirE[8]] == dh.cpu_arrays[src1.name][-2, i, 8]

    assert dh.cpu_arrays[src1.name][-1, 4, mirrored_dirE[4]] == dh.cpu_arrays[src1.name][-2, 4, 4]
    assert dh.cpu_arrays[src1.name][-1, 4, mirrored_dirE[8]] == dh.cpu_arrays[src1.name][-2, 4, 8]

    # check South
    assert dh.cpu_arrays[src1.name][1, 0, mirrored_dirS[8]] == dh.cpu_arrays[src1.name][1, 1, 8]
    assert dh.cpu_arrays[src1.name][1, 0, mirrored_dirS[2]] == dh.cpu_arrays[src1.name][1, 1, 2]

    for i in range(2, 4):
        assert dh.cpu_arrays[src1.name][i, 0, mirrored_dirS[7]] == dh.cpu_arrays[src1.name][i, 1, 7]
        assert dh.cpu_arrays[src1.name][i, 0, mirrored_dirS[2]] == dh.cpu_arrays[src1.name][i, 1, 2]
        assert dh.cpu_arrays[src1.name][i, 0, mirrored_dirS[8]] == dh.cpu_arrays[src1.name][i, 1, 8]

    assert dh.cpu_arrays[src1.name][4, 0, mirrored_dirS[2]] == dh.cpu_arrays[src1.name][4, 1, 2]
    assert dh.cpu_arrays[src1.name][4, 0, mirrored_dirS[7]] == dh.cpu_arrays[src1.name][4, 1, 7]

    # check West
    assert dh.cpu_arrays[src1.name][0, 1, mirrored_dirW[5]] == dh.cpu_arrays[src1.name][1, 1, 5]
    assert dh.cpu_arrays[src1.name][0, 1, mirrored_dirW[3]] == dh.cpu_arrays[src1.name][1, 1, 3]

    for i in range(2, 4):
        assert dh.cpu_arrays[src1.name][0, i, mirrored_dirW[5]] == dh.cpu_arrays[src1.name][1, i, 5]
        assert dh.cpu_arrays[src1.name][0, i, mirrored_dirW[3]] == dh.cpu_arrays[src1.name][1, i, 3]
        assert dh.cpu_arrays[src1.name][0, i, mirrored_dirW[7]] == dh.cpu_arrays[src1.name][1, i, 7]

    assert dh.cpu_arrays[src1.name][0, 4, mirrored_dirW[3]] == dh.cpu_arrays[src1.name][1, 4, 3]
    assert dh.cpu_arrays[src1.name][0, 4, mirrored_dirW[7]] == dh.cpu_arrays[src1.name][1, 4, 7]

    if given_normal:
        # check corners --> determined by the last boundary applied there.
        # SouthWest --> West
        assert dh.cpu_arrays[src1.name][0, 0, mirrored_dirW[5]] == dh.cpu_arrays[src1.name][1, 0, 5]
        # NorthWest --> West
        assert dh.cpu_arrays[src1.name][0, -1, mirrored_dirW[7]] == dh.cpu_arrays[src1.name][1, -1, 7]
        # NorthEast --> East
        assert dh.cpu_arrays[src1.name][-1, -1, mirrored_dirE[8]] == dh.cpu_arrays[src1.name][-2, -1, 8]
        # SouthEast --> East
        assert dh.cpu_arrays[src1.name][-1, 0, mirrored_dirE[6]] == dh.cpu_arrays[src1.name][-2, 0, 6]
    else:
        # check corners --> this time the normals are calculated correctly in the corners
        # SouthWest --> Normal = (1, 1); dir 7 --> 6
        assert dh.cpu_arrays[src1.name][0, 0, 6] == dh.cpu_arrays[src1.name][1, 1, 7]
        # NorthWest --> Normal = (1, -1); dir 8 --> 5
        assert dh.cpu_arrays[src1.name][0, -1, 8] == dh.cpu_arrays[src1.name][1, -2, 5]
        # NorthEast --> Normal = (-1, -1); dir 7 --> 6
        assert dh.cpu_arrays[src1.name][-1, -1, 7] == dh.cpu_arrays[src1.name][-2, -2, 6]
        # SouthEast --> Normal = (-1, 1); dir 5 --> 8
        assert dh.cpu_arrays[src1.name][-1, 0, 5] == dh.cpu_arrays[src1.name][-2, 1, 8]


def test_free_slip_index_list():
    stencil = LBStencil(Stencil.D2Q9)
    dh = create_data_handling(domain_size=(4, 4), periodicity=(False, False))
    src = dh.add_array('src', values_per_cell=len(stencil), alignment=True)
    dh.fill('src', 0.0, ghost_layers=True)

    lbm_config = LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.8)
    method = create_lb_method(lbm_config=lbm_config)

    bh = LatticeBoltzmannBoundaryHandling(method, dh, 'src', name="bh")

    free_slip = FreeSlip(stencil=stencil)
    add_box_boundary(bh, free_slip)

    bh.prepare()
    for b in dh.iterate():
        for b_obj, idx_arr in b[bh._index_array_name].boundary_object_to_index_list.items():
            index_array = idx_arr

    # normal directions
    normal_west = (1, 0)
    normal_east = (-1, 0)
    normal_south = (0, 1)
    normal_north = (0, -1)

    normal_south_west = (1, 1)
    normal_north_west = (1, -1)
    normal_south_east = (-1, 1)
    normal_north_east = (-1, -1)

    for cell in index_array:
        direction = stencil[cell[2]]
        inv_dir = inverse_direction(direction)

        boundary_cell = (cell[0] + direction[0], cell[1] + direction[1])
        normal = (cell[3], cell[4])
        # the data is written on the inverse direction of the fluid cell near the boundary
        # the data is read from the mirrored direction of the inverse direction where the mirror axis is the normal
        assert cell[5] == stencil.index(mirror_stencil(list(inv_dir), normal))

        if boundary_cell[0] == 0 and 0 < boundary_cell[1] < 5:
            assert normal == normal_west

        if boundary_cell[0] == 5 and 0 < boundary_cell[1] < 5:
            assert normal == normal_east

        if 0 < boundary_cell[0] < 5 and boundary_cell[1] == 0:
            assert normal == normal_south

        if 0 < boundary_cell[0] < 5 and boundary_cell[1] == 5:
            assert normal == normal_north

        if boundary_cell == (0, 0):
            assert cell[2] == cell[5]
            assert normal == normal_south_west

        if boundary_cell == (5, 0):
            assert cell[2] == cell[5]
            assert normal == normal_south_east

        if boundary_cell == (0, 5):
            assert cell[2] == cell[5]
            assert normal == normal_north_west

        if boundary_cell == (5, 5):
            assert cell[2] == cell[5]
            assert normal == normal_north_east


def test_free_slip_index_list_convex_corner():
    stencil = LBStencil(Stencil.D2Q9)
    dh = create_data_handling(domain_size=(4, 4))
    src = dh.add_array('src', values_per_cell=len(stencil))
    dh.fill('src', 0.0, ghost_layers=True)

    lbm_config = LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.8)
    method = create_lb_method(lbm_config=lbm_config)

    def bh_callback(x, y):
        radius = 2
        x_mid = 2
        y_mid = 2
        return (x - x_mid) ** 2 + (y - y_mid) ** 2 > radius ** 2

    bh = LatticeBoltzmannBoundaryHandling(method, dh, 'src', name="bh")

    free_slip = FreeSlip(stencil=stencil)
    bh.set_boundary(free_slip, mask_callback=bh_callback)

    bh.prepare()
    for b in dh.iterate():
        for b_obj, idx_arr in b[bh._index_array_name].boundary_object_to_index_list.items():
            index_array = idx_arr

    # correct index array for this case with convex corners
    test = [(2, 1, 2, 0, 1, 2), (2, 1, 3, 1, 0, 3), (2, 1, 7, 1, 1, 7),
            (2, 1, 8, 0, 1, 7), (3, 1, 2, 0, 1, 2), (3, 1, 4, -1, 0, 4),
            (3, 1, 7, 0, 1, 8), (3, 1, 8, -1, 1, 8), (1, 2, 2, 0, 1, 2),
            (1, 2, 3, 1, 0, 3), (1, 2, 5, 1, 0, 7), (1, 2, 7, 1, 1, 7),
            (2, 2, 7, 1, 1, 7), (3, 2, 8, -1, 1, 8), (4, 2, 2, 0, 1, 2),
            (4, 2, 4, -1, 0, 4), (4, 2, 6, -1, 0, 8), (4, 2, 8, -1, 1, 8),
            (1, 3, 1, 0, -1, 1), (1, 3, 3, 1, 0, 3), (1, 3, 5, 1, -1, 5),
            (1, 3, 7, 1, 0, 5), (2, 3, 5, 1, -1, 5), (3, 3, 6, -1, -1, 6),
            (4, 3, 1, 0, -1, 1), (4, 3, 4, -1, 0, 4), (4, 3, 6, -1, -1, 6),
            (4, 3, 8, -1, 0, 6), (2, 4, 1, 0, -1, 1), (2, 4, 3, 1, 0, 3),
            (2, 4, 5, 1, -1, 5), (2, 4, 6, 0, -1, 5), (3, 4, 1, 0, -1, 1),
            (3, 4, 4, -1, 0, 4), (3, 4, 5, 0, -1, 6), (3, 4, 6, -1, -1, 6)]

    for i, cell in enumerate(index_array):
        for j in range(len(cell)):
            assert cell[j] == test[i][j]


def test_free_slip_equivalence():
    # check if Free slip BC does the same if the normal direction is specified or not

    stencil = LBStencil(Stencil.D2Q9)
    dh = create_data_handling(domain_size=(4, 4), periodicity=(False, False))
    src1 = dh.add_array('src1', values_per_cell=stencil.Q, alignment=True)
    src2 = dh.add_array('src2', values_per_cell=stencil.Q, alignment=True)
    dh.fill('src1', 0.0, ghost_layers=True)
    dh.fill('src2', 0.0, ghost_layers=True)

    shape = dh.gather_array('src1', ghost_layers=True).shape

    num = 0
    for x in range(shape[0]):
        for y in range(shape[1]):
            for direction in range(shape[2]):
                dh.cpu_arrays['src1'][x, y, direction] = num
                dh.cpu_arrays['src2'][x, y, direction] = num
                num += 1

    method = create_lb_method(lbm_config=LBMConfig(stencil=stencil, method=Method.SRT, relaxation_rate=1.8))

    bh1 = LatticeBoltzmannBoundaryHandling(method, dh, 'src1', name="bh1")
    free_slip1 = FreeSlip(stencil=stencil)
    bh1.set_boundary(free_slip1, slice_from_direction('N', dh.dim))

    bh2 = LatticeBoltzmannBoundaryHandling(method, dh, 'src2', name="bh2")
    free_slip2 = FreeSlip(stencil=stencil, normal_direction=(0, -1))
    bh2.set_boundary(free_slip2, slice_from_direction('N', dh.dim))

    bh1()
    bh2()

    assert np.array_equal(dh.gather_array('src1'), dh.gather_array('src2'))


def test_exotic_boundaries():
    step = LatticeBoltzmannStep((50, 50), relaxation_rate=1.8, compressible=False, zero_centered=True, periodicity=False)
    add_box_boundary(step.boundary_handling, NeumannByCopy())
    step.boundary_handling.set_boundary(StreamInConstant(0), make_slice[0, :])
    step.run(100)
    assert np.max(step.velocity[:, :, :]) < 1e-13


def test_boundary_utility_functions():
    stencil = LBStencil(Stencil.D2Q9)
    method = create_lb_method(lbm_config=LBMConfig(stencil=stencil))

    noslip = NoSlip("noslip")
    assert noslip == NoSlip("noslip")
    assert not noslip == NoSlip("test")
    assert not noslip == UBB((0, 0), name="ubb")

    assert noslip.name == "noslip"
    noslip.name = "test name setter"
    assert noslip.name == "test name setter"

    ubb = UBB((0, 0), name="ubb")
    assert ubb == UBB((0, 0), name="ubb")
    assert not noslip == UBB((0, 0), name="test")
    assert not ubb == NoSlip("noslip")

    simple_extrapolation = SimpleExtrapolationOutflow(normal_direction=stencil[4], stencil=stencil, name="simple")
    assert simple_extrapolation == SimpleExtrapolationOutflow(normal_direction=stencil[4],
                                                              stencil=stencil, name="simple")
    assert not simple_extrapolation == SimpleExtrapolationOutflow(normal_direction=stencil[4],
                                                                  stencil=stencil, name="test")
    assert not simple_extrapolation == NoSlip("noslip")

    outflow = ExtrapolationOutflow(normal_direction=stencil[4], lb_method=method, name="outflow")
    assert outflow == ExtrapolationOutflow(normal_direction=stencil[4], lb_method=method, name="outflow")
    assert not outflow == ExtrapolationOutflow(normal_direction=stencil[4], lb_method=method, name="test")
    assert not outflow == simple_extrapolation

    density = FixedDensity(density=1.0, name="fixedDensity")
    assert density == FixedDensity(density=1.0, name="fixedDensity")
    assert not density == FixedDensity(density=1.0, name="test")
    assert not density == UBB((0, 0), name="ubb")

    diffusion = DiffusionDirichlet(concentration=1.0, name="diffusion")
    assert diffusion == DiffusionDirichlet(concentration=1.0, name="diffusion")
    assert not diffusion == DiffusionDirichlet(concentration=1.0, name="test")
    assert not diffusion == density

    neumann = NeumannByCopy(name="Neumann")
    assert neumann == NeumannByCopy(name="Neumann")
    assert not neumann == NeumannByCopy(name="test")
    assert not neumann == diffusion

    stream = StreamInConstant(constant=1.0, name="stream")
    assert stream == StreamInConstant(constant=1.0, name="stream")
    assert not stream == StreamInConstant(constant=1.0, name="test")
    assert not stream == noslip
