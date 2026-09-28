"""Pilot: periodic-terminal phase vs. z, both with free terminal winding.

Each formulation gets the same physical native seed and two IPOPT calls:
unpenalized then penalized. This is separate from E2's fixed-k comparison.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from zcartpole import ZNCartPole
from zcartpole.opt import optimize_swingup
from e2_fair_benchmark import (MODEL_PARAMS, N, T, TH0, W_PENALTY,
                               native_pair_seed, theta_optimize_swingup,
                               validate_z)

OUT = Path(__file__).resolve().parent / 'results' / 'e3_free_winding_phase.json'


def run(x_max=0.5, intervals=N):
    model = ZNCartPole(**MODEL_PARAMS)
    phase_seed = native_pair_seed(model, T, intervals, TH0, x_max)
    phase_stages = []
    for penalty in (0.0, W_PENALTY):
        ok, _, Xp, Up, J, iterations, seconds = theta_optimize_swingup(
            model, T, intervals, x_max, TH0, None, penalty, phase_seed)
        phase_stages.append(dict(penalty=penalty, ok=ok, J=J,
                                 iterations=iterations, seconds=seconds))
        phase_seed = (Xp, Up)
    phase_turns = (np.asarray([Xp[2+2*i, -1] for i in range(model.n)])
                   / (2*np.pi)).tolist() if phase_stages[-1]['ok'] else None

    z_seed = None
    z_stages = []
    for penalty in (0.0, W_PENALTY):
        info = {}
        start = time.perf_counter()
        ok, _, Xz, Uz, J = optimize_swingup(
            model, T=T, N=intervals, x_max=x_max, th0=TH0,
            w_penalty=penalty, warm=z_seed, info=info)
        seconds = time.perf_counter() - start
        assert len(info['calls']) == 1
        diagnostics = validate_z(model, Xz, T=T) if ok else None
        z_stages.append(dict(penalty=penalty, ok=ok, J=J,
                             iterations=info['calls'][0]['iter'], seconds=seconds,
                             validation=diagnostics))
        z_seed = (Xz, Uz)

    return dict(settings=dict(T=T, N=intervals, x_max=x_max, th0=list(TH0),
                              w_penalty=W_PENALTY, model=MODEL_PARAMS,
                              starting_path='native z and its identical physical phase lift',
                              calls_per_formulation=2),
                phase=dict(stages=phase_stages, terminal_turns=phase_turns,
                           valid=bool(phase_stages[-1]['ok'] and phase_turns is not None
                                      and all(abs(t-round(t)) < 1e-5
                                              for t in phase_turns))),
                z=dict(stages=z_stages,
                       valid=bool(z_stages[-1]['ok'] and
                                  z_stages[-1]['validation'] and
                                  z_stages[-1]['validation']['valid'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x-max', type=float, default=0.5)
    parser.add_argument('--N', type=int, default=N)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.x_max <= 0 or args.N < 2:
        parser.error('x_max must be positive and N >= 2')
    result = run(args.x_max, args.N)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name in ('phase', 'z'):
        stages = result[name]['stages']
        print(f"{name}: valid={result[name]['valid']} "
              f"J={stages[-1]['J']} "
              f"seconds={sum(s['seconds'] for s in stages):.1f} "
              f"iterations={sum(s['iterations'] or 0 for s in stages)}", flush=True)
    print(f'Wrote: {args.output}')


if __name__ == '__main__':
    main()
