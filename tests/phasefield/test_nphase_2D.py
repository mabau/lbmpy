from functools import partial
from time import time

import numpy as np

from lbmpy.phasefield.analytical import (
    n_phases_correction_function, n_phases_correction_function_sign_switch)
from lbmpy.phasefield.contact_angle_circle_fitting import liquid_lens_neumann_angles
from lbmpy.phasefield.post_processing import analytic_neumann_angles
from lbmpy.phasefield.scenarios import create_n_phase_model, create_three_phase_model
from pystencils import create_data_handling, make_slice
from pystencils.utils import boolean_array_bounding_box

color = {'yellow': '\033[93m',
         'blue': '\033[94m',
         'green': '\033[92m',
         'bold': '\033[1m',
         'cend': '\033[0m',
         }


def random_string(length):
    import random
    import string
    return ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(length))


def run_n_phase_2d(num_phases, interface_width=1, correction_factor=0.5,
                   correction_power=4, domain_width=100, domain_aspect_ratio=2,
                   kappas=(0.005, 0.005, 0.005),
                   initialization='step', interface_with='two', evaluation_steps=10000, time_steps=10000000,
                   angle_convergence_threshold=0.001, return_scenario=False,
                   f2=None, **kwargs):
    start_time = time()

    expected_angles = analytic_neumann_angles(kappas)
    angle_format = ", ".join(["{:.1f}".format(a) for a in expected_angles])
    print("{num_phases} phases at {domain_width} domain, {blue}corr: {correction_factor}**{correction_power},"
          "{green} angles: {expected_angles}:{cend}".format(num_phases=num_phases, domain_width=domain_width,
                                                            correction_factor=correction_factor,
                                                            correction_power=correction_power,
                                                            expected_angles=angle_format,
                                                            **color)
          )

    dh = create_data_handling((domain_width, domain_width // domain_aspect_ratio), periodicity=True)
    if f2 is None:
        if correction_power is not None and correction_factor is not None:
            if correction_power == 'sign_switch':
                f2 = partial(n_phases_correction_function_sign_switch, beta=correction_factor)
            else:
                f2 = partial(n_phases_correction_function, beta=correction_factor, power=correction_power)
        else:
            f2 = lambda c: c ** 2 * (1 - c) ** 2

    full_kappas = [sum(kappas) / 3] * num_phases
    if interface_with == 'last':
        full_kappas[0], full_kappas[1], full_kappas[-1] = kappas
    else:
        full_kappas[0], full_kappas[1], full_kappas[2] = kappas

    def surface_tensions(i, j):
        if i == j:
            return 0
        return (full_kappas[i] + full_kappas[j]) / 6 * interface_width

    if num_phases > 3:
        sc = create_n_phase_model(data_handling=dh, f2=f2, surface_tensions=surface_tensions,
                                  num_phases=num_phases, alpha=interface_width,
                                  **kwargs)
    elif num_phases == 3:
        sc = create_three_phase_model(data_handling=dh, kappa=full_kappas, alpha=interface_width,
                                      **kwargs)

    if initialization == 'step':
        if interface_with == 'last':
            drop_phase_idx = num_phases - 1
        elif interface_with == 'two':
            drop_phase_idx = 2
        else:
            raise ValueError("Parameter 'interface_with' has to be either 'last' or 'two'")
        sc.set_single_concentration(make_slice[:, 0.5:], 0, value=1)
        sc.set_single_concentration(make_slice[:, :0.5], 1, value=1)
        sc.set_single_concentration(make_slice[0.3:0.7, 0.3:0.7], drop_phase_idx, value=1)
    else:
        raise ValueError("Unsupported value for 'initialization parameter")

    sc.set_pdf_fields_from_macroscopic_values()
    print("   - {yellow}{time}s{cend} Compiled".format(time=int(time() - start_time), **color))

    if return_scenario:
        return sc

    outer_steps = time_steps // evaluation_steps
    eval_results = []
    last_angles = None
    stable = True
    converged = False
    abort_reason = ''
    for os in range(outer_steps):
        sc.run(evaluation_steps)
        max_phi = sc.data_handling.max(sc.phi_field_name)
        if np.isnan(max_phi):
            stable = False
            print("   {yellow}-unstable{cend}".format(**color))
            abort_reason = 'nan'
            break
        try:
            angles = liquid_lens_neumann_angles(sc.concentration[:, :, :], drop_phase_idx=drop_phase_idx)
        except (ValueError, AssertionError):
            stable = False
            print("   {yellow}-problem detecting angle{cend}".format(**color))
            abort_reason = 'angle detection failed'
            break

        drop_bb = boolean_array_bounding_box(sc.concentration[:, :, drop_phase_idx] > 0.5)
        domain_too_small = False
        min_distance = 2
        for bounds, shape in zip(drop_bb, sc.shape):
            if bounds[0] < min_distance or bounds[1] > shape - 1 - min_distance:
                domain_too_small = True
        if domain_too_small:
            print("   {yellow}-domain too small - drop touched boundary{cend}".format(**color))
            abort_reason = 'domain too small'
            break

        angle_format = ", ".join(["{:.1f}".format(a) for a in angles])
        print("   - {yellow}{time}s{cend}, {step}: {green}{angles}{cend}".format(time=int(time() - start_time),
                                                                                 step=sc.time_steps_run,
                                                                                 angles=angle_format,
                                                                                 **color))
        eval_results.append({'theta{}'.format(i): value for i, value in enumerate(angles)})
        if last_angles is not None:
            max_diff = max(abs(a - b) for a, b in zip(last_angles, angles))
            if max_diff < angle_convergence_threshold:
                converged = True
                print("   {green}-converged{cend}".format(**color))
                break
        last_angles = angles

    result = {'stable': stable, 'converged': converged, 'eval_results': eval_results, 'abort_reason': abort_reason}
    if stable:
        data_file_name = random_string(20)
        result['data_file_name'] = data_file_name
        print("   - writing result to {}".format(data_file_name))
        result.update(eval_results[-1])
        sc.data_handling.save_all(data_file_name)

    return result


def study_3phase(study, **kwargs):
    kappas = [(0.01, 0.02, k3) for k3 in (0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001)]
    for k in kappas:
        for d in ['standard', 'isotropic', 'isotropic_hd']:
            params = {'num_phases': 3,
                      'domain_width': 300,
                      'kappas': k,
                      'discretization': d}
            params.update(kwargs)
            study.add_run(params)


def study_2d(study, **kwargs):
    kb = 0.05 / 4
    kappas = [
        (kb, kb / 2, kb / 2),
        (kb, kb / 2, kb / 4),
        (kb, kb / 2, kb / 8),
        (kb, kb / 2, kb / 16),
        (kb, kb / 2, kb / 32),
        (kb, kb / 2, kb / 64),
        (kb, kb / 2, kb / 128),
        (kb, kb / 2, kb / 256),
    ]

    for domain_width in (100, 260, 500):
        for num_phases in (4, 5):
            for interface_with in ['two', 'last']:
                for d in ['standard', 'isotropic']:
                    for kappa in kappas:
                        for triple_point_factor in (np.average(kappa) * 0.8,
                                                    np.average(kappa) * 1,
                                                    np.average(kappa) * 1.2,
                                                    0.5 * np.min(kappa),
                                                    np.min(kappa),
                                                    np.max(kappa)):
                            aspect = 2
                            if kappa[0] / kappa[2] >= 64:
                                aspect = 3
                            if kappa[0] / kappa[2] >= 128:
                                aspect = 4

                            params = {
                                'num_phases': num_phases,
                                'correction_power': 'sign_switch',
                                'correction_factor': 1,
                                'kappas': kappa,
                                'domain_width': domain_width,
                                'discretization': d,
                                'angle_convergence_threshold': 0.01,
                                'interface_with': interface_with,
                                'triple_point_energy': triple_point_factor,
                                'domain_aspect_ratio': aspect,
                            }
                            params.update(kwargs)
                            study.add_run(params)


def main():
    from pystencils.runhelper import ParameterStudy
    s = ParameterStudy(run_n_phase_2d)
    # study_3phase(s)
    study_2d(s)
    s.run_from_command_line()


if __name__ == '__main__':
    main()
