"""Run a configured balance or swing-up case with one to three links.

Examples:
  python examples/run_links.py --links 2 --mode balance -o n2_balance
  python examples/run_links.py --links 3 --mode swingup -o n3_swingup

The parameters below are examples. Replace them with a measured system model.
"""
import argparse
from math import pi

from zcartpole import SystemConfig, run_system


def make_config(n, mode, animation=False):
    if n not in (1, 2, 3):
        raise ValueError('links must be 1, 2 or 3')
    if mode not in ('balance', 'swingup'):
        raise ValueError('mode must be balance or swingup')
    angles = ([0.03] * n if mode == 'balance' else
              [pi + angle for angle in (0.05, 0.03, 0.02)[:n]])
    return SystemConfig.from_dict({
        'schema_version': 1,
        'name': f'n{n}_{mode}',
        'rail': {'half_travel_m': 0.5},
        'cart': {'mass_kg': 1.0, 'viscous_friction_ns_m': 0.15},
        'links': [
            {'length_m': 0.4, 'tip_mass_kg': 0.1, 'rod_mass_kg': 0.05,
             'joint_damping_nms_rad': 0.005}
            for _ in range(n)
        ],
        'motor': {
            'type': 'transfer_function',
            'command_to_force': {
                'numerator': [1.0], 'denominator': [0.02, 1.0]},
            'command_unit': 'N', 'command_min': -80.0,
            'command_max': 80.0, 'force_limit_n': 80.0,
            'command_delay_s': 0.01,
        },
        'initial': {'angles_rad': angles},
        'simulation': {
            'mode': mode, 'dt_s': 0.002,
            'duration_s': 6.0,
            'plan_duration_s': 3.0 * max(n - 1, 1),
            'plan_nodes': 100, 'seeds': [0, 1],
            'cart_margin_m': 0.02 if n == 3 else 0.005,
            'catch_extra_s': 3.0,
            'animation': animation,
        },
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--mode', choices=('balance', 'swingup'), required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--animation', action='store_true')
    args = parser.parse_args()
    result = run_system(make_config(args.links, args.mode, args.animation),
                        args.output)
    print(f'{result.status}: {result.output_dir / "summary.json"}')
    return 0 if result.status == 'caught' else 1


if __name__ == '__main__':
    raise SystemExit(main())
