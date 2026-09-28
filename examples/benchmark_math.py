"""Compare the optional compiled mechanics with the Python reference.

Run: python examples/benchmark_math.py
The timing excludes model construction and first-call JIT compilation.
"""
import argparse
import time

import numpy as np

from zcartpole import ZNCartPole


def run(model, initial, steps, dt):
    state = initial.copy()
    start = time.perf_counter()
    for _ in range(steps):
        state = model.step(state, 0.0, dt)
    return time.perf_counter() - start, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    common = dict(m=[.1] * 3, l=[.4] * 3, M=1.0, u_max=80.0,
                  rod_mass=[.05] * 3, cart_damping=.15,
                  joint_damping=[.005] * 3)
    python_model = ZNCartPole(**common, backend='python')
    try:
        fast_model = ZNCartPole(**common, backend='numba')
    except ImportError as exc:
        parser.error(str(exc))
    start = python_model.pack(0, 0, np.exp(1j * np.array([3.1, 3.15, 3.17])),
                              np.zeros(3))
    # Compile before timing, and use the same initial state for each path.
    fast_model.step(start, 0.0, .002)
    slow_s, slow_end = run(python_model, start, args.steps, .002)
    fast_s, fast_end = run(fast_model, start, args.steps, .002)
    error = float(np.max(np.abs(slow_end - fast_end)))
    print(f'{args.steps} steps: Python {slow_s:.4f} s, Numba {fast_s:.4f} s')
    print(f'ratio {slow_s / fast_s:.1f}x; final-state max error {error:.3g}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
