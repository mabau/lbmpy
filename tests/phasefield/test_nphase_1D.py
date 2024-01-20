from functools import partial

import numpy as np

from lbmpy.phasefield.analytical import (
    analytic_interface_profile, n_phases_correction_function,
    n_phases_correction_function_sign_switch)
from lbmpy.phasefield.experiments1D import init_sharp_interface
from lbmpy.phasefield.scenarios import create_n_phase_model
from pystencils import create_data_handling, make_slice


def extract_profile(sc, width, phase_idx=1):
    width //= 2
    interface_x = sc.data_handling.shape[0] // 4
    extraction_slice = make_slice[interface_x - width:interface_x + width, 0, phase_idx]
    return sc.phi_slice(extraction_slice).copy()


def analytic_profile(width, alpha):
    x = np.arange(width) - (width // 2)
    return np.array([analytic_interface_profile(x_i - 0.5, alpha) for x_i in x], dtype=np.float64)


def error(profile1, profile2):
    return np.sum(np.abs(profile1 - profile2)) / profile1.shape[0]


def run_n_phase_1d(num_phases, interface_width=1, correction_factor=0.5,
                   correction_power=2, surface_tension=0.0025, domain_width=100,
                   initialization='step', interface_with='last', evaluation_steps=5000, time_steps=50000):

    assert num_phases > 3

    dh = create_data_handling((domain_width, 1), periodicity=True)
    if correction_power is not None and correction_factor is not None:
        if correction_power == 'sign_switch':
            f2 = partial(n_phases_correction_function_sign_switch, beta=correction_factor)
        else:
            f2 = partial(n_phases_correction_function, beta=correction_factor, power=correction_power)
    else:
        f2 = lambda c: c ** 2 * (1 - c) ** 2

    sc = create_n_phase_model(data_handling=dh, f2=f2, surface_tensions=surface_tension,
                              num_phases=num_phases, alpha=interface_width)

    if initialization == 'step':
        if interface_with == 'last':
            init_sharp_interface(sc, phase_idx=1, inverse=False)
        elif interface_with == 'two':
            init_sharp_interface(sc, phase_idx=1, inverse=False)
            init_sharp_interface(sc, phase_idx=0, inverse=True)
        else:
            raise ValueError("Parameter 'interface_with' has to be either 'last' or 'two'")
    else:
        raise ValueError("Unsupported value for 'initialization parameter")

    sc.set_pdf_fields_from_macroscopic_values()

    outer_steps = time_steps // evaluation_steps
    eval_results = []
    stable = True
    for os in range(outer_steps):
        eval_result = {}
        sc.run(evaluation_steps)
        phi_slice = sc.phi[:, 0, 1]
        eval_result['phi_min'], eval_result['phi_max'] = np.min(phi_slice), np.max(phi_slice)
        if np.isnan(eval_result['phi_max']):
            stable = False
            break

        simulated = extract_profile(sc, 50 * interface_width)
        analytic = analytic_profile(50 * interface_width, interface_width)
        eval_result['error'] = error(simulated, analytic)
        eval_result['other_min'], eval_result['other_max'] = np.min(sc.phi[:, 0, 2:]), np.max(sc.phi[:, 0, 2:])

        eval_results.append(eval_result)

    result = {'stable': stable, 'eval_results': eval_results}
    print("α={interface_width}, p={correction_power}, β={correction_factor}, ".format(**locals()) +
          "st={surface_tension}, init={initialization}".format(**locals()), end="")

    if stable:
        result.update(eval_results[-1])
        result['profile'] = list(extract_profile(sc, 50 * interface_width))
        print(" -> err={result['error']:.4f}, "
              "min/max={result['other_min']:.4f}/{result['other_max']:.4f}".format(**locals()))
    else:
        print("  -> unstable")
    return result


def study_1d(study):
    for num_phases in (4, ):
        for alpha in (1, 2, 8):
            for st in (0.005, 0.005 / 2, 0.005 / 4, 0.005 / 8, 0.005 / 16):
                for beta in (0.0001, 0.001, 0.01, 0.1, 1, 10, 100):
                    for correction_power in (2, 4, 'sign_switch'):
                        for if_type in ('last', 'two'):
                            params = {
                                'num_phases': num_phases,
                                'interface_width': alpha,
                                'correction_factor': beta,
                                'correction_power': correction_power,
                                'surface_tension': st,
                                'domain_width': alpha * 100,
                                'interface_with': if_type,
                            }
                            study.add_run(params)


if __name__ == '__main__':
    from pystencils.runhelper import ParameterStudy
    s = ParameterStudy(run_n_phase_1d)
    study_1d(s)
    s.run_from_command_line()
