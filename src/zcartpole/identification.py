"""Fit a first-order command-to-cart-force response from a measured step."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def fit_first_order_step(time_s, command, force_n):
    """Return gain, time constant, delay and fit diagnostics for one held step.

    Samples must include a settled baseline, one command step and enough
    post-step data. The cart should be held at zero speed during measurement.
    """
    t, u, f = (np.asarray(v, dtype=float) for v in (time_s, command, force_n))
    if (t.ndim != 1 or u.shape != t.shape or f.shape != t.shape
        or len(t) < 12 or not all(np.isfinite(v).all() for v in (t, u, f))
        or np.any(np.diff(t) <= 0)):
        raise ValueError('Need at least 12 finite, strictly time-ordered samples')
    changes = np.abs(np.diff(u))
    index = int(np.argmax(changes))+1
    if index < 3 or len(t)-index < 8:
        raise ValueError('Record at least three baseline and eight post-step samples')
    u0, u1 = float(np.median(u[:index])), float(np.median(u[index:]))
    du = u1-u0
    if abs(du) < 1e-9 or np.max(np.abs(u[:index]-u0)) > .02*abs(du) or (
        np.max(np.abs(u[index:]-u1)) > .02*abs(du)):
        raise ValueError('Command must contain one clean, held step')
    t_step = t[index]
    span = t[-1]-t_step
    baseline = float(np.median(f[:index]))
    final = float(np.median(f[-max(3, len(f[index:])//5):]))
    if abs(final-baseline) < 1e-9:
        raise ValueError('Force response is too small to identify')
    dt = float(np.median(np.diff(t)))
    def response(p):
        gain, tau, delay, offset = p
        elapsed = np.maximum(t-t_step-delay, 0)
        return offset+gain*du*(-np.expm1(-elapsed/tau))
    initial = [(final-baseline)/du, max(span/5, dt/10), 0, baseline]
    result = least_squares(lambda p: response(p)-f, initial,
                           bounds=([-np.inf, dt/100, 0, -np.inf],
                                   [np.inf, span*20, span*.9, np.inf]),
                           max_nfev=3000)
    gain, tau, delay, offset = result.x
    predicted = response(result.x)
    rmse = float(np.sqrt(np.mean((f-predicted)**2)))
    return dict(gain_n_per_command=float(gain), time_constant_s=float(tau),
                command_delay_s=float(delay), baseline_force_n=float(offset),
                fit_rmse_n=rmse, step_time_s=float(t_step),
                command_before=u0, command_after=u1,
                measurement_duration_s=float(t[-1]-t[0]))


def fit_step_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    try:
        arrays = [[float(row[field]) for row in rows]
                  for field in ('time_s', 'command', 'force_n')]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('CSV requires numeric time_s, command, force_n columns') from exc
    return fit_first_order_step(*arrays)


def main():
    parser = argparse.ArgumentParser(description='Fit a complete actuator chain from a held-force step')
    parser.add_argument('csv', help='Measured time_s,command,force_n CSV')
    parser.add_argument('--output', '-o', required=True, help='New JSON output path')
    args = parser.parse_args()
    try:
        result = fit_step_csv(args.csv)
        result['source_sha256'] = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
        output = Path(args.output)
        with output.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'gain={result["gain_n_per_command"]:.6g}, '
          f'tau={result["time_constant_s"]:.6g} s, '
          f'delay={result["command_delay_s"]:.6g} s, '
          f'RMSE={result["fit_rmse_n"]:.6g} N')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
