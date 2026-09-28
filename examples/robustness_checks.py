"""Challenge one n=3 plan with model mismatch, initial error and command delay.

The rail has no collision model. Each rollout stops at its first rail violation.
These illustrative cases are diagnostics, not reliability probabilities.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from zcartpole import ZNCartPole, plan_swingup


def rollout(plant, tracker, plan, angles, *, delay=0.0, dt=0.002):
    steps = round((plan.t[-1]+3.0)/dt)
    lag = round(delay/dt)
    if abs(lag*dt-delay) > 1e-9:
        raise ValueError('delay must be a multiple of dt')
    state = plant.pack(0.0, 0.0, np.exp(1j*np.asarray(angles)),
                       np.zeros(plant.n))
    queue = [0.0]*lag
    times, states, forces = [], [], []
    for k in range(steps+1):
        t = k*dt
        command = tracker.control(t, state)
        queue.append(command)
        applied = float(np.clip(queue.pop(0), -plant.u_max, plant.u_max))
        times.append(t)
        states.append(state.copy())
        forces.append(applied)
        if abs(state[0]) > plan.x_max+1e-5:
            return dict(completed=False, first_rail_violation_s=t,
                        max_cart=float(np.max(np.abs(np.asarray(states)[:, 0]))),
                        max_force=float(np.max(np.abs(forces))),
                        catch=None)
        if k < steps:
            state = plant.step(state, applied, dt)
    sol = SimpleNamespace(t=np.asarray(times), y=np.asarray(states).T,
                          u=np.asarray(forces))
    report = plan.evaluate_catch(tracker.m, sol)
    return dict(completed=True, first_rail_violation_s=None,
                max_cart=report.max_cart, max_force=report.max_force,
                catch=asdict(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('robustness_results.json'))
    parser.add_argument('--dt', type=float, default=0.002)
    args = parser.parse_args()
    if not np.isfinite(args.dt) or args.dt <= 0:
        parser.error('--dt must be positive')

    base = dict(m=(.1,)*3, l=(.4,)*3, M=1.0, u_max=80,
                rod_mass=(.05,)*3, cart_damping=.15,
                joint_damping=(.005,)*3)
    nominal = ZNCartPole(**base)
    angles = tuple(np.pi+a for a in (.05, .03, .02))
    plan = plan_swingup(nominal, T=6, N=100, x_max=.5,
                        cart_margin=.01, th0=angles)
    if not plan.success:
        raise SystemExit(f'Nominal planning failed: {plan.attempts}')
    tracker = plan.tracker(nominal)

    cases = [
        ('nominal', {}, 0.0, 0.0),
        ('initial_angles_plus_0.05_rad', {}, .05, 0.0),
        ('rod_masses_plus_20_percent', {'rod_mass':(.06,)*3}, 0.0, 0.0),
        ('all_masses_plus_20_percent',
         {'M':1.2, 'm':(.12,)*3, 'rod_mass':(.06,)*3}, 0.0, 0.0),
        ('link_lengths_plus_10_percent', {'l':(.44,)*3}, 0.0, 0.0),
        ('viscous_damping_times_3',
         {'cart_damping':.45, 'joint_damping':(.015,)*3}, 0.0, 0.0),
        ('command_delay_20_ms', {}, 0.0, .02),
    ]
    data = dict(note='Illustrative fixed scenarios, not Monte Carlo or hardware data',
                plan_J=plan.J, nominal_bound=plan.x_max,
                scenarios=[])
    for name, changes, offset, delay in cases:
        plant = ZNCartPole(**(base | changes))
        result = rollout(plant, tracker, plan, tuple(a+offset for a in angles),
                         delay=delay, dt=args.dt)
        data['scenarios'].append(dict(name=name, **result))
        status = 'rail violation' if not result['completed'] else (
            'caught' if result['catch']['success'] else 'failed catch')
        print(f'{name}: {status}, max |x|={result["max_cart"]:.4f} m', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
