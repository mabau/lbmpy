import os
from types import MappingProxyType

import numpy as np

from lbmpy.scenarios import create_lid_driven_cavity as run_ldc_lbmpy
from pystencils import make_slice


def create_force_driven_channel(force=1e-6, domain_size=None, dim=2, radius=None, length=None,
                                optimization=MappingProxyType({}), lbm_kernel=None, kernel_params=MappingProxyType({}),
                                **kwargs):
    from lbmpy.lbstep import LatticeBoltzmannStep
    from lbmpy.boundaries import NoSlip
    from pystencils.slicing import slice_from_direction
    wall_boundary = NoSlip()

    if domain_size is not None:
        dim = len(domain_size)
    else:
        if dim is None or radius is None or length is None:
            raise ValueError("Pass either 'domain_size' or 'dim', 'radius' and 'length'")

    assert dim in (2, 3)
    kwargs['force'] = tuple([force, 0, 0][:dim])

    round_channel = False
    if radius is not None:
        assert length is not None
        if dim == 3:
            domain_size = (length, 2 * radius + 1, 2 * radius + 1)
            round_channel = True
        else:
            if domain_size is None:
                domain_size = (length, 2 * radius)

    if 'force_model' not in kwargs:
        kwargs['force_model'] = 'guo'

    lb_step = LatticeBoltzmannStep(domain_size, optimization=optimization, lbm_kernel=lbm_kernel,
                                   kernel_params=kernel_params, periodicity=(True, False, False), **kwargs)

    boundary_handling = lb_step.boundary_handling
    if dim == 2:
        for direction in ('N', 'S'):
            boundary_handling.set_boundary(wall_boundary, slice_from_direction(direction, dim))
    elif dim == 3:
        if round_channel:
            def circle_mask_cb(_, y, z):
                y_mid = np.max(y) // 2
                z_mid = np.max(z) // 2
                return (y - y_mid) ** 2 + (z - z_mid) ** 2 > radius ** 2

            boundary_handling.set_boundary(wall_boundary, mask_callback=circle_mask_cb)
        else:
            for direction in ('N', 'S', 'T', 'B'):
                boundary_handling.set_boundary(wall_boundary, slice_from_direction(direction, dim))

    assert domain_size is not None
    if 'force_model' not in kwargs:
        kwargs['force_model'] = 'guo'

    return lb_step


def plot_velocity_fields(vel1, vel2):
    import lbmpy.plot as plt
    diff = np.average(np.abs(vel2 - vel1))
    has_diff = diff > 1e-12
    num_plots = 3 if has_diff else 2
    plt.subplot(1, num_plots, 1)
    plt.title("lbmpy")
    plt.vector_field(vel1)
    plt.subplot(1, num_plots, 2)
    plt.title("walberla ref")
    plt.vector_field(vel2)
    if has_diff:
        plt.title("Difference (%f)" % diff)
        plt.subplot(1, num_plots, 3)
        plt.vector_field(vel2 - vel1)
    plt.show()


def compare_scenario(lbmpy_scenario_creator, walberla_scenario_creator, optimization=MappingProxyType({}),
                     action='Testing', name='ss', plot="off", **kwargs):
    if 'time_steps' in kwargs:
        time_steps = kwargs['time_steps']
        del kwargs['time_steps']
    else:
        time_steps = 100

    ref_file_path = get_directory_reference_files()

    if action == 'Testing':
        reference = np.load(os.path.join(ref_file_path, name + ".npz"))

        lbmpy_version = lbmpy_scenario_creator(optimization=optimization, **kwargs)
        lbmpy_version.run(time_steps)

        rho_lbmpy = lbmpy_version.density_slice(make_slice[:, :] if lbmpy_version.dim == 2 else make_slice[:, :, :])
        vel_lbmpy = lbmpy_version.velocity_slice(make_slice[:, :] if lbmpy_version.dim == 2 else make_slice[:, :, :])

        if plot == "on":
            plot_velocity_fields(vel_lbmpy, reference['vel'])

        np.testing.assert_almost_equal(reference['rho'], rho_lbmpy, err_msg="Density fields are different")
        np.testing.assert_almost_equal(reference['vel'], vel_lbmpy, err_msg="Velocity fields are different")

    else:

        wlb_time_loop = walberla_scenario_creator(**kwargs)
        pdfs_wlb, rho_wlb, vel_wlb = wlb_time_loop(time_steps)

        if os.path.exists(ref_file_path + name + ".npz"):
            os.remove(ref_file_path + name + ".npz")
        np.savez_compressed(ref_file_path + name, pdfs=pdfs_wlb, rho=rho_wlb, vel=vel_wlb)


def get_directory_reference_files():
    script_file = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_file)
    return os.path.join(script_dir, "reference_files")


def compare_lid_driven_cavity(optimization=MappingProxyType({}), action='Testing', plot="off", **kwargs):
    if kwargs['method'] == 'MRT':
        name = "LidDrivenCavity_" + kwargs['method']
    else:
        name = "LidDrivenCavity_" + kwargs['method'] + "_" + kwargs['force_model']
    if kwargs['compressible']:
        name = name + "_compressible"
    else:
        name = name + "_incompressible"

    try:
        from lbmpy_tests.walberla_scenario_setup import create_lid_driven_cavity as run_lid_driven_cavity_walberla
    except ImportError:
        run_lid_driven_cavity_walberla = None

    return compare_scenario(run_ldc_lbmpy, run_lid_driven_cavity_walberla, optimization, action, name, plot, **kwargs)


def compare_force_driven_channel(optimization=MappingProxyType({}), action='Testing', plot="off", **kwargs):
    from functools import partial
    lbmpy_func = partial(create_force_driven_channel, dim=2)

    name = "ForceDrivenChannel_" + kwargs['force_model']
    if kwargs['compressible']:
        name = name + "_compressible"
    else:
        name = name + "_incompressible"

    try:
        from lbmpy_tests.walberla_scenario_setup import create_lid_driven_cavity as run_force_driven_channel_walberla
    except ImportError:
        run_force_driven_channel_walberla = None

    return compare_scenario(lbmpy_func, run_force_driven_channel_walberla, optimization, action, name, plot, **kwargs)


def test_channel_srt(action='Testing', plot="off"):
    params = {'force': 0.0001,
              'radius': 40,
              'length': 40,
              'method': 'SRT',
              'stencil': 'D2Q9',
              'time_steps': 500,
              'maxwellian_moments': False,
              'relaxation_rates': [1.8]}

    if action == 'Testing' or action == 'Regenerate':
        for force_model_name in ('simple', 'luo', 'guo'):
            for compressible in (False, True):
                print("%s Channel SRT, Force Model %s, compressible %d" % (action, force_model_name, compressible))
                compare_force_driven_channel(compressible=compressible, force_model=force_model_name,
                                             action=action, plot=plot, **params)
    else:
        print("Possible Actions: Regenerate or Testing")


def test_ldc_srt(action='Testing', plot="off"):
    force = (0.0001, -0.00002)
    if action == 'Testing' or action == 'Regenerate':
        for force_model_name in ('simple', 'luo', 'guo'):
            for compressible in (False, True):
                print("%s LidDrivenCavity SRT, Force Model %s, compressible %d" % (action, force_model_name,
                                                                                   compressible))
                compare_lid_driven_cavity(domain_size=(16, 19), lid_velocity=0.005, stencil='D2Q9',
                                          method='SRT', relaxation_rates=[1.8], compressible=compressible,
                                          maxwellian_moments=False,
                                          force=force, force_model=force_model_name, action=action, plot=plot)
    else:
        print("Possible Actions: Regenerate or Testing")


def test_ldc_trt(action='Testing', plot="off"):
    f = (0.0001, -0.00002, 0.0000124)
    if action == 'Testing' or action == 'Regenerate':
        for force_modelName in ('luo',):  # guo for multiple relaxation rates has to be implemented...
            for compressible in (True, False):
                print("%s LidDrivenCavity TRT, Force Model %s, compressible %d" % (action, force_modelName,
                                                                                   compressible))
                # print("testing", force_modelName, compressible)
                compare_lid_driven_cavity(domain_size=(16, 17, 18), lid_velocity=0.005, stencil='D3Q19',
                                          method='TRT', relaxation_rates=[1.8, 1.3], compressible=compressible,
                                          maxwellian_moments=False,
                                          force=f, force_model=force_modelName, action=action, plot=plot)
    else:
        print("Possible Actions: Regenerate or Testing")


def test_ldc_mrt(action='Testing', plot="off"):
    from lbmpy.methods import mrt_orthogonal_modes_literature
    from lbmpy.stencils import get_stencil
    if action == 'Testing' or action == 'Regenerate':
        print("%s LidDrivenCavity MRT, compressible 0" % action)
        moments = mrt_orthogonal_modes_literature(get_stencil("D3Q19"), True, False)
        compare_lid_driven_cavity(domain_size=(16, 17, 18), lid_velocity=0.005, stencil='D3Q19',
                                  method='MRT', nested_moments=moments, compressible=False, maxwellian_moments=False,
                                  relaxation_rates=[1, 1.3, 1.4, 1.5, 1.25, 1.36, 1.12], action=action, plot=plot)
    else:
        print("Possible Actions: Regenerate or Testing")
