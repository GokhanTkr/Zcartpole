"""Plan, simulate and animate n=1,2,3 with point or illustrative rigid rods.

Install: pip install -e '.[opt,viz]'
Run: python examples/animate_link_counts.py
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from zcartpole import ZNCartPole, plan_swingup, save_animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links', nargs='+', type=int, choices=(1, 2, 3),
                        default=(1, 2, 3), help='Link counts to run')
    parser.add_argument('--output-dir', type=Path, default=Path('link_demos'))
    parser.add_argument('--N', type=int, default=100, help='Collocation intervals')
    parser.add_argument('--dt', type=float, default=0.002, help='Simulation step (s)')
    parser.add_argument('--preset', choices=('point', 'rigid'), default='point',
                        help='rigid: 50 g uniform rods and viscous damping')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for n in dict.fromkeys(args.links):
        physical = dict(rod_mass=(0.05,)*n, cart_damping=0.15,
                        joint_damping=(0.005,)*n) if args.preset == 'rigid' else {}
        model = ZNCartPole(m=(0.1,)*n, l=(0.4,)*n, u_max=80.0, **physical)
        th0 = tuple(np.pi+offset for offset in (0.05, 0.03, 0.02)[:n])
        T = 3*max(n-1, 1)
        start = time.perf_counter()
        margin = 0.01 if args.preset == 'rigid' else 0.005
        plan = plan_swingup(model, T=T, N=args.N, x_max=0.5, th0=th0,
                            cart_margin=margin)
        item = dict(n=n, preset=args.preset, T=T, N=args.N,
                    point_mass=model.m.tolist(), rod_mass=model.rod_mass.tolist(),
                    cart_damping=model.cart_damping,
                    joint_damping=model.joint_damping.tolist(),
                    x_max=0.5, cart_margin=plan.cart_margin,
                    solver_success=plan.success, J=plan.J,
                    windings=plan.windings, chosen_seed=plan.seed_k,
                    attempts=[asdict(attempt) for attempt in plan.attempts])
        if plan.success:
            sol = plan.simulate_catch(model, t_extra=3, dt=args.dt)
            report = plan.evaluate_catch(model, sol)
            gif = args.output_dir/f'n{n}_swingup.gif'
            save_animation(model, sol, gif, fps=15, speed=1.5,
                           x_limit=plan.x_max)
            item.update(catch=asdict(report), gif=str(gif))
        item['wall_seconds'] = time.perf_counter()-start
        summary.append(item)
        (args.output_dir/'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        caught = item.get('catch', {}).get('success', False)
        print(f'n={n}: solver={plan.success} catch={caught} '
              f'J={plan.J} windings={plan.windings} '
              f'elapsed={item["wall_seconds"]:.1f}s', flush=True)

    print(f'Results: {args.output_dir / "summary.json"}')
    if any(not item.get('catch', {}).get('success', False) for item in summary):
        raise SystemExit('At least one case did not pass the closed-loop check')


if __name__ == '__main__':
    main()
