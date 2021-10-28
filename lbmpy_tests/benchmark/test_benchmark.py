from statistics import median

import numpy as np
import sympy as sp

from lbmpy.scenarios import create_lid_driven_cavity
from pystencils.cpu.cpujit import add_or_change_compiler_flags
from pystencils.runhelper import ParameterStudy


def parameter_filter(parameters):
    """Returns false for parameter combinations which are invalid or not implemented yet."""
    is_entropic_kbc = parameters['method'].startswith('trt-kbc-') and parameters.get("entropic", False)
    if is_entropic_kbc and parameters['stencil'] == 'D3Q19':
        return False
    if is_entropic_kbc and not parameters['compressible']:
        return False
    return True


def optimization_options_cpu(all_vectorization_options=False, all_cse_options=False, cores=(1,), with_split=False,
                             instruction_set='avx'):
    """Generator of different CPU optimization options.

    Args:
        all_vectorization_options: if true, explore all different vectorization possibilities (x6 more options)
        all_cse_options: if true, explore all cse options
        cores: sequence of core numbers
        with_split: if true, yield configurations with and without split, otherwise only without split
        instruction_set: 'sse', 'avx' or 'avx2'
    """

    if all_vectorization_options:
        vectorization_options = [
            {'instruction_set': instruction_set,
             'assume_aligned': assume_aligned,
             'nontemporal': nontemporal,
             'assume_inner_stride_one': True,
             'assume_sufficient_line_padding': lp}
            for assume_aligned in (False, True)
            for nontemporal in ((False, True) if assume_aligned else (False,))
            for lp in (False, True)
        ]
        vectorization_options.append(False)
    else:
        vectorization_options = [{'instruction_set': instruction_set, 'assume_aligned': True,
                                  'nontemporal': True, 'assume_inner_stride_one': True}]

    if all_cse_options:
        cse_options = [
            {'cse_pdfs': cse_pdfs, 'cse_global': cse_global}
            for cse_pdfs in (False, True) for cse_global in (False, True)
        ]
    else:
        cse_options = [{'cse_pdfs': False, 'cse_global': True}]

    for vectorization_option in vectorization_options:
        for cse_option in cse_options:
            for split in (False, True) if with_split else (False,):
                for openmp in cores:
                    for field_layout in ('fzyx', 'zyxf'):
                        if field_layout == 'zyxf' and vectorization_option is not False:
                            continue
                        opt_option = cse_option.copy()
                        opt_option['vectorization'] = vectorization_option
                        opt_option['split'] = split
                        opt_option['openmp'] = openmp
                        opt_option['field_layout'] = field_layout
                        yield opt_option


def method_options(dim=2, with_srt=False, with_mrt=False,
                   with_entropic=False, with_cumulant=False, with_smagorinsky=False, with_d3q27=True):
    """Generator for different lbmpy method parameters

    Args:
        dim: 2D or 3D
        with_srt: include single relaxation time models (by default only TRT are included)
        with_mrt: include multi-relaxation time models
        with_entropic: include entropic models
        with_cumulant: include cumulant models
        with_smagorinsky: include methods with Smagorinsky turbulence model
    """
    relaxation_rates = tuple(np.linspace(1.1, 1.9, 27))
    rr_free = "rr_free"
    methods = [{'method': 'trt'}]
    if with_mrt:
        methods += [{'method': 'mrt'}, {'method': 'mrt_raw'}]

    if with_srt:
        methods += [{'method': 'srt'}]

    if with_entropic:
        methods += [{'entropic': True, 'method': 'mrt', 'relaxation_rates': [1.5, 1.5, rr_free, rr_free,
                                                                             rr_free, rr_free]},
                    {'entropic': True, 'method': 'mrt', 'relaxation_rates': [1.5, rr_free, rr_free, rr_free,
                                                                             rr_free, rr_free]},
                    {'entropic': True, 'method': 'trt-kbc-n1'},
                    {'entropic': True, 'method': 'trt-kbc-n2'},
                    {'entropic': True, 'method': 'trt-kbc-n3'},
                    {'entropic': True, 'method': 'trt-kbc-n4'},
                    {'method': 'entropic-srt'}]

    if with_cumulant:
        methods += [{'cumulant': True, 'method': 'srt'},
                    {'cumulant': True, 'method': 'trt'},
                    {'cumulant': True, 'method': 'mrt3'},
                    {'cumulant': True, 'method': 'mrt_raw'}]

    if with_smagorinsky:
        methods += [{'smagorinsky': True, 'method': 'srt'},
                    {'smagorinsky': True, 'method': 'mrt3'}]

    stencils3d = ('D3Q19', 'D3Q27') if with_d3q27 else ("D3Q19",)
    for stencil in ("D2Q9",) if dim == 2 else stencils3d:
        for method in methods:
            options = {'compressible': True, 'stencil': stencil, 'relaxation_rates': relaxation_rates}
            options.update(method)
            if parameter_filter(options):
                yield options


def benchmark_scenarios(domain_size, method_option_params={}, optimization_option_params={},
                        fixed_loop_sizes=True, fixed_relaxation_rates=True):

    method_option_params['dim'] = len(domain_size)
    for method_option in method_options(**method_option_params):
        for optimization in optimization_options_cpu(**optimization_option_params):
            result = method_option.copy()
            result.update({
                'domain_size': domain_size,
                'optimization': optimization,
                'fixed_loop_sizes': fixed_loop_sizes,
                'fixed_relaxation_rates': fixed_relaxation_rates,
            })
            yield result


def run(domain_size, **kwargs):
    color = {'yellow': '\033[93m',
             'blue': '\033[94m',
             'green': '\033[92m',
             'bold': '\033[1m',
             'cend': '\033[0m',
             }
    study_name = kwargs.get('study_name', 'study')
    del kwargs['study_name']

    if 'relaxation_rates' in kwargs:
        kwargs['relaxation_rates'] = [sp.sympify(e) for e in kwargs['relaxation_rates']]

    if 'compiler_flags' in kwargs:
        add_or_change_compiler_flags(kwargs['compiler_flags'].split())
        del kwargs['compiler_flags']
    else:
        add_or_change_compiler_flags("-march=native")

    opt_str = str(kwargs['optimization']).replace("'", "").replace("False", "0").replace("True", "1")
    opt_str = opt_str.replace("assume_aligend", 'align').replace("instruction_set", "is").replace("nontemporal", "nt")
    param_str = "{bold}{domain_size}{cend} {blue}method: {method: <5}, stencil: {stencil}, {cend} " \
                "comp: {compressible:d}, const_loop: {fixed_loop_sizes:d}, const_rr: {fixed_relaxation_rates:d}, " \
                "{green}opt: {opt_str}{cend}"
    param_str = param_str.format(opt_str=opt_str, domain_size=domain_size, **kwargs, **color)
    sc = create_lid_driven_cavity(domain_size, **kwargs)

    mlups = [sc.benchmark(time_for_benchmark=2) for _ in range(5)]
    if not np.isfinite(sc.data_handling.max(sc.velocity_data_name)):
        print("-> ", param_str, " got unstable", flush=True)
        return {'mlups_max': None,
                'mlups_median': None,
                'all_measurements': [],
                'stable': False,
                'study_name': study_name}

    result_str = "  {yellow}{bold}{mlups:.0f}±{diff:.2f} MLUPS {cend}".format(mlups=median(mlups),
                                                                              diff=max(mlups) - min(mlups),
                                                                              **color)
    print("-> ", param_str, result_str, flush=True)
    return {'mlups_max': max(mlups),
            'mlups_median': median(mlups),
            'all_measurements': mlups,
            'stable': True}


def study_optimization_options(study, domain_sizes=((1024, 1024), (256, 256, 128)),
                               with_srt=False, with_mrt=False, with_entropic=False, with_cumulant=False,
                               with_smagorinsky=False,
                               all_vectorization_options=False, all_cse_options=False, cores=(1,), with_split=False,
                               with_symbols=False, overwrite_params={}):
    mp = {'with_mrt': with_mrt, 'with_entropic': with_entropic, 'with_cumulant': with_cumulant,
          'with_smagorinsky': with_smagorinsky, 'with_srt': with_srt}
    op = {'all_vectorization_options': all_vectorization_options, 'all_cse_options': all_cse_options,
          'cores': cores, 'with_split': with_split}

    for ds in domain_sizes:
        for const_ls in (True, False) if with_symbols else (True,):
            for const_rr in (True, False) if with_symbols else (True,):
                for params in benchmark_scenarios(ds, mp, op,
                                                  fixed_loop_sizes=const_ls, fixed_relaxation_rates=const_rr):
                    params.update(overwrite_params)
                    params['study_name'] = 'optimization_options'
                    study.add_run(params, weight=ds[0] // domain_sizes[0][0])
    return study


def study_block_sizes_trt(study):
    mp = {'with_mrt': False, 'with_entropic': False, 'with_smagorinsky': False, 'with_srt': False, 'with_d3q27': False}
    op = {'all_vectorization_options': True, 'all_cse_options': False,
          'cores': (1, 2, 3, 4), 'with_split': True}

    domain_sizes = [
        # 3D
        (8, 8, 8), (12, 12, 12), (16, 16, 16),
        (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256),
        (257, 257, 257), (258, 258, 258), (259, 259, 259), (260, 260, 260),
        (261, 261, 261), (262, 262, 262), (263, 263, 263), (264, 264, 264),
        # 2D
        (32, 32), (64, 64), (70, 70), (80, 80), (84, 84), (90, 90),  # L2 boundary
        (128, 128), (256, 256), (300, 300), (320, 320), (340, 340), (350, 350),  # L3 boundary
        (400, 400), (512, 512), (1024, 1024), (2048, 2048), (8192, 8192),
    ]

    for ds in domain_sizes:
        for fixed in (True, False):
            for params in benchmark_scenarios(ds, mp, op, fixed_loop_sizes=fixed, fixed_relaxation_rates=fixed):
                params['study_name'] = 'block_sizes_trt'
                study.add_run(params, weight=ds[0] // domain_sizes[0][0])


# @pytest.mark.longrun
# def test_run():
#     """Called by test suite - ensures that benchmark can be run"""
#     s = ParameterStudy(run)
#     study_optimization_options(s, domain_sizes=((17, 23), (19, 17, 18))).run()


def study_compiler_flags(study):
    mp = {'with_mrt': True, 'with_entropic': False, 'with_cumulant': False,
          'with_smagorinsky': False, 'with_srt': False}
    op = {'all_vectorization_options': False, 'all_cse_options': False,
          'cores': (1, 2, 3, 4), 'with_split': True}

    vector_configs = [
        ('-march=native -mavx512vl',          "avx512"),
        ('-march=native -mavx512vl -mno-fma', "avx512"),
        ('-mavx2',            "avx"),
        ('-mavx2 -mno-fma',   "avx"),
    ]

    for fixed in (False, True):
        for flags, instruction_set in vector_configs:
            for params in benchmark_scenarios((128, 128, 128), mp, op,
                                              fixed_loop_sizes=fixed, fixed_relaxation_rates=fixed):
                    params['study_name'] = 'vectorization_flags'
                    if 'vectorization' in params['optimization'] and isinstance(params['optimization']['vectorization'], dict):
                        params['optimization']['vectorization']['instruction_set'] = instruction_set
                        params['compiler_flags'] = flags
                    study.add_run(params)


def main():
    s = ParameterStudy(run)

    study_compiler_flags(s)

    #study_block_sizes_trt(s)
    #
    #study_optimization_options(s, domain_sizes=((128, 128, 128), (1024, 1024)),
    #                           with_mrt=True, all_vectorization_options=True, with_smagorinsky=True,
    #                           with_entropic=True, with_srt=True, with_cumulant=True,
    #                           all_cse_options=True, with_split=False, with_symbols=False,
    #                           cores=(1, 2, 3, 4))
    #
    #study_optimization_options(s, domain_sizes=((128, 128, 128), (1024, 1024)),
    #                           with_mrt=True, all_vectorization_options=True, with_smagorinsky=True, with_srt=True,
    #                           all_cse_options=True, with_split=True, with_symbols=True,
    #                           cores=(1, 2, 3, 4))

    s.run_from_command_line()


if __name__ == '__main__':
    main()
