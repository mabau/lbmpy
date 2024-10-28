import pystencils as ps

import numpy as np

from lbmpy.stencils import LBStencil
from pystencils.slicing import get_slice_before_ghost_layer, get_ghost_region_slice
from lbmpy.creationfunctions import create_lb_update_rule, LBMConfig, LBMOptimisation
from lbmpy.advanced_streaming.communication import (
    get_communication_slices,
    _fix_length_one_slices,
    LBMPeriodicityHandling,
    periodic_pdf_gpu_copy_kernel,
)
from lbmpy.advanced_streaming.utility import streaming_patterns, Timestep
from lbmpy.enums import Stencil

import pytest


@pytest.mark.parametrize(
    "stencil", [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27]
)
@pytest.mark.parametrize("streaming_pattern", streaming_patterns)
@pytest.mark.parametrize("timestep", [Timestep.EVEN, Timestep.ODD])
def test_slices_not_empty(stencil, streaming_pattern, timestep):
    stencil = LBStencil(stencil)
    arr = np.zeros((4,) * stencil.D + (stencil.Q,))
    slices = get_communication_slices(
        stencil,
        streaming_pattern=streaming_pattern,
        prev_timestep=timestep,
        ghost_layers=1,
    )
    for _, slices_list in slices.items():
        for src, dst in slices_list:
            assert all(s != 0 for s in arr[src].shape)
            assert all(s != 0 for s in arr[dst].shape)


@pytest.mark.parametrize(
    "stencil", [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27]
)
@pytest.mark.parametrize("streaming_pattern", streaming_patterns)
@pytest.mark.parametrize("timestep", [Timestep.EVEN, Timestep.ODD])
def test_src_dst_same_shape(stencil, streaming_pattern, timestep):
    stencil = LBStencil(stencil)
    arr = np.zeros((4,) * stencil.D + (stencil.Q,))
    slices = get_communication_slices(
        stencil,
        streaming_pattern=streaming_pattern,
        prev_timestep=timestep,
        ghost_layers=1,
    )
    for _, slices_list in slices.items():
        for src, dst in slices_list:
            src_shape = arr[src].shape
            dst_shape = arr[dst].shape
            assert src_shape == dst_shape


@pytest.mark.parametrize(
    "stencil", [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27]
)
def test_pull_communication_slices(stencil):
    stencil = LBStencil(stencil)

    slices = get_communication_slices(
        stencil, streaming_pattern="pull", prev_timestep=Timestep.BOTH, ghost_layers=1
    )
    for i, d in enumerate(stencil):
        if i == 0:
            continue

        for s in slices[d]:
            if s[0][-1] == i:
                src = s[0][:-1]
                dst = s[1][:-1]
                break

        inner_slice = _fix_length_one_slices(
            get_slice_before_ghost_layer(d, ghost_layers=1)
        )
        inv_dir = (-e for e in d)
        gl_slice = _fix_length_one_slices(
            get_ghost_region_slice(inv_dir, ghost_layers=1)
        )
        assert src == inner_slice
        assert dst == gl_slice


@pytest.mark.parametrize("direction", LBStencil(Stencil.D3Q27).stencil_entries)
@pytest.mark.parametrize("pull", [False, True])
def test_gpu_comm_kernels(direction: tuple, pull: bool):
    pytest.importorskip("cupy")

    stencil = LBStencil(Stencil.D3Q27)
    inv_dir = stencil[stencil.inverse_index(direction)]
    target = ps.Target.GPU

    domain_size = (4,) * stencil.D

    dh: ps.datahandling.SerialDataHandling = ps.create_data_handling(
        domain_size,
        periodicity=(True,) * stencil.D,
        parallel=False,
        default_target=target,
    )

    field = dh.add_array("field", values_per_cell=2)

    if pull:
        dst_slice = get_ghost_region_slice(inv_dir)
        src_slice = get_slice_before_ghost_layer(direction)
    else:
        dst_slice = get_slice_before_ghost_layer(direction)
        src_slice = get_ghost_region_slice(inv_dir)

    src_slice += (1,)
    dst_slice += (1,)

    kernel = periodic_pdf_gpu_copy_kernel(field, src_slice, dst_slice)

    dh.cpu_arrays[field.name][src_slice] = 42.0
    dh.all_to_gpu()

    dh.run_kernel(kernel)

    dh.all_to_cpu()
    np.testing.assert_equal(dh.cpu_arrays[field.name][dst_slice], 42.0)


@pytest.mark.parametrize("stencil", [Stencil.D2Q9, Stencil.D3Q19])
@pytest.mark.parametrize("streaming_pattern", streaming_patterns)
def test_direct_copy_and_kernels_equivalence(stencil: Stencil, streaming_pattern: str):
    pytest.importorskip("cupy")

    target = ps.Target.GPU
    stencil = LBStencil(stencil)
    domain_size = (4,) * stencil.D

    dh: ps.datahandling.SerialDataHandling = ps.create_data_handling(
        domain_size,
        periodicity=(True,) * stencil.D,
        parallel=False,
        default_target=target,
    )

    pdfs_a = dh.add_array("pdfs_a", values_per_cell=stencil.Q)
    pdfs_b = dh.add_array("pdfs_b", values_per_cell=stencil.Q)

    dh.fill(pdfs_a.name, 0.0, ghost_layers=True)
    dh.fill(pdfs_b.name, 0.0, ghost_layers=True)

    for q in range(stencil.Q):
        sl = ps.make_slice[:4, :4, q] if stencil.D == 2 else ps.make_slice[:4, :4, :4, q]
        dh.cpu_arrays[pdfs_a.name][sl] = q
        dh.cpu_arrays[pdfs_b.name][sl] = q

    dh.all_to_gpu()

    direct_copy = LBMPeriodicityHandling(stencil, dh, pdfs_a.name, streaming_pattern, cupy_direct_copy=True)
    kernels_copy = LBMPeriodicityHandling(stencil, dh, pdfs_b.name, streaming_pattern, cupy_direct_copy=False)

    direct_copy(Timestep.EVEN)
    kernels_copy(Timestep.EVEN)

    dh.all_to_cpu()

    np.testing.assert_equal(
        dh.cpu_arrays[pdfs_a.name],
        dh.cpu_arrays[pdfs_b.name]
    )


@pytest.mark.parametrize(
    "stencil_name", [Stencil.D2Q9, Stencil.D3Q15, Stencil.D3Q19, Stencil.D3Q27]
)
def test_optimised_and_full_communication_equivalence(stencil_name):
    target = ps.Target.CPU
    stencil = LBStencil(stencil_name)
    domain_size = (4,) * stencil.D

    dh = ps.create_data_handling(
        domain_size,
        periodicity=(True,) * stencil.D,
        parallel=False,
        default_target=target,
    )

    pdf = dh.add_array("pdf", values_per_cell=len(stencil), dtype=np.int64)
    dh.fill("pdf", 0, ghost_layers=True)
    pdf_tmp = dh.add_array("pdf_tmp", values_per_cell=len(stencil), dtype=np.int64)
    dh.fill("pdf_tmp", 0, ghost_layers=True)

    gl = dh.ghost_layers_of_field("pdf")

    num = 0
    for idx, x in np.ndenumerate(dh.cpu_arrays["pdf"]):
        dh.cpu_arrays["pdf"][idx] = num
        dh.cpu_arrays["pdf_tmp"][idx] = num
        num += 1

    lbm_config = LBMConfig(stencil=stencil, kernel_type="stream_pull_only")
    lbm_opt = LBMOptimisation(symbolic_field=pdf, symbolic_temporary_field=pdf_tmp)
    config = ps.CreateKernelConfig(target=dh.default_target, cpu_openmp=True)

    ac = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_opt)
    ast = ps.create_kernel(ac, config=config)
    stream = ast.compile()

    full_communication = dh.synchronization_function(
        pdf.name, target=dh.default_target, optimization={"openmp": True}
    )
    full_communication()

    dh.run_kernel(stream)
    dh.swap("pdf", "pdf_tmp")
    pdf_full_communication = np.copy(dh.cpu_arrays["pdf"])

    num = 0
    for idx, x in np.ndenumerate(dh.cpu_arrays["pdf"]):
        dh.cpu_arrays["pdf"][idx] = num
        dh.cpu_arrays["pdf_tmp"][idx] = num
        num += 1

    optimised_communication = LBMPeriodicityHandling(
        stencil=stencil,
        data_handling=dh,
        pdf_field_name=pdf.name,
        streaming_pattern="pull",
    )
    optimised_communication()
    dh.run_kernel(stream)
    dh.swap("pdf", "pdf_tmp")

    if stencil.D == 3:
        for i in range(gl, domain_size[0]):
            for j in range(gl, domain_size[1]):
                for k in range(gl, domain_size[2]):
                    for f in range(len(stencil)):
                        assert (
                            dh.cpu_arrays["pdf"][i, j, k, f]
                            == pdf_full_communication[i, j, k, f]
                        ), print(f)
    else:
        for i in range(gl, domain_size[0]):
            for j in range(gl, domain_size[1]):
                for f in range(len(stencil)):
                    assert (
                        dh.cpu_arrays["pdf"][i, j, f] == pdf_full_communication[i, j, f]
                    )
