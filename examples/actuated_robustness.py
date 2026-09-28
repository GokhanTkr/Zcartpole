"""Screen illustrative n=3 plans with motor lag, dead time and model errors.

Run after ``pip install -e '.[opt]'``. The motor values and uncertainty cases
are examples, not identified hardware data or reliability probabilities.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from zcartpole import (ActuatorTVLQRTracker, RobustScenario, ZNCartPole,
                       assess_plan, select_robust_swingup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('actuated_robustness_results.json'))
    parser.add_argument('--dt', type=float, default=0.002)
    args = parser.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0 or any(
        abs(round(delay/args.dt)*args.dt-delay) > 1e-9
        for delay in (.01, .02)
    ):
        parser.error('--dt must be positive and divide both 10 and 20 ms')

    base = dict(m=(.1,)*3, l=(.4,)*3, M=1.0, u_max=80.0,
                rod_mass=(.05,)*3, cart_damping=.15,
                joint_damping=(.005,)*3)
    nominal = ZNCartPole(**base)
    initial = tuple(np.pi+a for a in (.05, .03, .02))
    scenarios = (
        RobustScenario('nominal_motor_20_ms_delay_10_ms', nominal,
                       tau=.02, command_delay=.01),
        RobustScenario('motor_16_ms_delay_10_ms', nominal,
                       tau=.016, command_delay=.01),
        RobustScenario('motor_24_ms_delay_10_ms', nominal,
                       tau=.024, command_delay=.01),
        RobustScenario('rod_mass_plus_10_percent',
                       ZNCartPole(**(base | {'rod_mass':(.055,)*3})),
                       tau=.02, command_delay=.01),
        RobustScenario('initial_angles_plus_0.005_rad', nominal,
                       th0=tuple(a+.005 for a in initial),
                       tau=.02, command_delay=.01),
    )
    factory = lambda model, plan: ActuatorTVLQRTracker(model, plan, tau=.02)
    selection = select_robust_swingup(
        nominal, scenarios,
        ({'T':6, 'cart_margin':.01}, {'T':6, 'cart_margin':.02}),
        N=100, x_max=.5, th0=initial, dt=args.dt,
        controller_factory=factory, predict_delay=True,
        prediction_tau=.02, min_cart_clearance=.005)
    data = dict(note='Finite illustrative screening; no hardware guarantee',
                design_motor_tau_s=.02, design_command_delay_s=.01,
                min_cart_clearance_m=.005, success=selection.success,
                candidates=[])
    for candidate in selection.candidates:
        row = dict(settings=candidate.settings, plan_success=candidate.plan.success,
                   J=candidate.plan.J, seed_k=candidate.plan.seed_k,
                   windings=candidate.plan.windings, selected=(
                       selection.plan is candidate.plan),
                   passed_screen=candidate.success,
                   scenarios=[asdict(s) for s in candidate.scenarios])
        data['candidates'].append(row)
        print(f"candidate {row['settings']}: "
              f"{'passed' if row['passed_screen'] else 'failed'}, "
              f"J={row['J']}", flush=True)
        for s in row['scenarios']:
            print(f"  {s['name']}: {'caught' if s['success'] else 'failed'}, "
                  f"max |x|={s['max_cart']:.5f} m", flush=True)

    if selection.success:
        stress = (
            RobustScenario('motor_20_ms_delay_20_ms', nominal,
                           tau=.02, command_delay=.02),
            RobustScenario('all_masses_plus_20_percent',
                           ZNCartPole(**(base | dict(M=1.2, m=(.12,)*3,
                                                      rod_mass=(.06,)*3))),
                           tau=.02, command_delay=.01),
        )
        data['stress_cases_not_in_selection'] = [asdict(s) for s in assess_plan(
            nominal, selection.plan, stress, dt=args.dt,
            controller_factory=factory, predict_delay=True,
            prediction_tau=.02)]
        for s in data['stress_cases_not_in_selection']:
            print(f"stress {s['name']}: "
                  f"{'caught' if s['success'] else 'failed'}, "
                  f"first rail violation={s['first_rail_violation_s']}",
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
