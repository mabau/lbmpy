import waLBerla as wlb
import numba
from lbmpy.walberlaConnection import makeLbmpySweepFromWalberlaLatticeModel
import time

default_parameters = {
    'stencil': 'D3Q19',
    'equilibriumAccuracyOrder': 2,
    'compressible': False,
    'collisionModel': 'SRT',
    'variableLoopBounds': False,
    'layout': 'fzyx',
    'domain_size': (200, 200, 200),
    'threads': 1,
    'timesteps': 5,
    'setup_timesteps': 0,
}


def create_walberla_lattice_model(stencil, collisionModel, equilibriumAccuracyOrder, compressible):
    forceModel = wlb.lbm.forceModels.NoForce()
    if collisionModel == 'SRT':
        collisionModel = wlb.lbm.collisionModels.SRT(1.8)
    elif collisionModel == 'TRT':
        collisionModel = wlb.lbm.collisionModels.TRT.constructWithMagicNumber(1.8)
    else:
        raise ValueError("Unknown collision model " + collisionModel)

    return wlb.lbm.makeLatticeModel(stencil, collisionModel, forceModel,
                                    compressible, equilibriumAccuracyOrder)


def create_blockstorage(domain_size, walberla_lattice_model, layout):
    blocks = wlb.createUniformBlockGrid(cells=tuple(domain_size), periodic=(0, 0, 0))
    if layout == 'fzyx':
        layout_enum = wlb.field.fzyx
    elif layout == 'zyxf':
        layout_enum = wlb.field.zyxf
    else:
        raise ValueError("Unknown layout " + layout)

    wlb.lbm.addPdfFieldToStorage(blocks, "pdfs", walberla_lattice_model, initialDensity=1.0, layout=layout_enum)
    return blocks


def run_sweep(blocks, sweep, timesteps):
    for t in range(timesteps):
        for block in blocks:
            sweep(block)


def lbmpy_timing(blocks, walberla_lattice_model, variableLoopBounds, setup_timesteps, timesteps):
    sweep = makeLbmpySweepFromWalberlaLatticeModel(walberla_lattice_model, blocks, 'pdfs', variableLoopBounds)
    run_sweep(blocks, sweep, setup_timesteps)
    start_time = time.perf_counter()
    run_sweep(blocks, sweep, timesteps)
    time_elapsed = time.perf_counter() - start_time
    return time_elapsed / timesteps


def walberla_timing(blocks, setup_timesteps, timesteps):
    sweep = wlb.lbm.makeCellwiseSweep(blocks, "pdfs").streamCollide
    run_sweep(blocks, sweep, setup_timesteps)
    start_time = time.perf_counter()
    run_sweep(blocks, sweep, timesteps)
    time_elapsed = time.perf_counter() - start_time
    return time_elapsed / timesteps


def secondsPerTimestepToMLUPS(domain_size, timing):
    cells = 1
    for i in domain_size:
        cells *= i

    return cells / timing * 1e-6


def run_timing(**kwargs):
    for arg in kwargs:
        if arg not in default_parameters:
            raise ValueError("Unknown parameter " + str(arg))

    kwargs.update(default_parameters)

    #from omp import set_num_threads
    #set_num_threads(kwargs['threads'])

    result_lbmpy = None
    result_wlb = None
    for useWalberla in [False, True]:

        lattice_model = create_walberla_lattice_model(kwargs['stencil'], kwargs['collisionModel'],
                                                      kwargs['equilibriumAccuracyOrder'], kwargs['compressible'])
        blocks = create_blockstorage(kwargs['domain_size'], lattice_model, kwargs['layout'])

        if useWalberla:
            result_wlb = walberla_timing(blocks, kwargs['setup_timesteps'], kwargs['timesteps'])
        else:
            result_lbmpy = lbmpy_timing(blocks, lattice_model, kwargs['variableLoopBounds'],
                                        kwargs['setup_timesteps'], kwargs['timesteps'])

    domain_size = kwargs['domain_size']
    print("waLBerla: %f MLUPS   -  lbmpy: %f MLUPS" % (secondsPerTimestepToMLUPS(domain_size, result_wlb),
                                                       secondsPerTimestepToMLUPS(domain_size, result_lbmpy)))


if __name__ == "__main__":
    from waLBerla import build_info
    print(build_info.compiler_flags)
    run_timing(compressible=True, equilibriumAccuracyOrder=2)

