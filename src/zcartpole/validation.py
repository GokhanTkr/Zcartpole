"""Independent actuator model validation from a separate measured input trace."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import SystemConfig
from .linear_actuator import LinearActuator


def _finite_number(value, name, *, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(number) or (positive and number <= 0):
        raise ValueError(f'{name} must be finite'+(' and positive' if positive else ''))
    return number


def _load_model(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('Model JSON must be an object')
    if 'schema_version' in data:
        actuator = SystemConfig.from_dict(data).build_motor()
        if not isinstance(actuator, LinearActuator):
            raise ValueError('Validation requires a general linear actuator model')
        return actuator, 'system_config', None, None
    gain = _finite_number(data.get('gain_n_per_command'), 'gain_n_per_command')
    tau = _finite_number(data.get('time_constant_s'), 'time_constant_s', positive=True)
    delay = _finite_number(data.get('command_delay_s', 0), 'command_delay_s')
    if delay < 0:
        raise ValueError('command_delay_s must be nonnegative')
    actuator = LinearActuator.from_transfer_functions(
        [gain], [tau, 1], command_min=-1e100, command_max=1e100,
        force_limit_n=1e100, delay_s=delay)
    training_range = (data.get('command_before'), data.get('command_after'))
    if all(x is not None for x in training_range):
        training_range = tuple(_finite_number(x, 'training command')
                               for x in training_range)
    else:
        training_range = None
    return actuator, 'fitted_first_order', data.get('source_sha256'), training_range


def _load_measurements(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {'time_s', 'command', 'force_n'} <= set(reader.fieldnames):
            raise ValueError('CSV requires time_s, command, force_n columns')
        has_speed = 'velocity_m_s' in reader.fieldnames
        rows = list(reader)
    try:
        t = np.array([float(row['time_s']) for row in rows])
        u = np.array([float(row['command']) for row in rows])
        f = np.array([float(row['force_n']) for row in rows])
        v = np.array([float(row['velocity_m_s']) for row in rows]) if has_speed else np.zeros(len(t))
    except (ValueError, TypeError) as exc:
        raise ValueError('CSV contains missing or nonnumeric samples') from exc
    if (len(t) < 12 or any(not np.isfinite(z).all() for z in (t, u, f, v))
        or np.any(np.diff(t) <= 0)):
        raise ValueError('Need at least 12 finite, strictly time-ordered samples')
    return t, u, f, v, has_speed


def predict_force_trace(actuator, time_s, command, measured_force_n,
                        velocity_m_s=None):
    """Predict from measured inputs; measured output sets only pre-step offset.

    Input samples are zero-order held. Delay can be fractional relative to the
    sampling period. The initial actuator state is assumed at equilibrium.
    """
    t, u, f = (np.asarray(a, dtype=float) for a in
               (time_s, command, measured_force_n))
    v = np.zeros(len(t)) if velocity_m_s is None else np.asarray(velocity_m_s, dtype=float)
    if (t.ndim != 1 or len(t) < 12 or u.shape != t.shape or f.shape != t.shape
        or v.shape != t.shape or any(not np.isfinite(z).all() for z in (t, u, f, v))
        or np.any(np.diff(t) <= 0)):
        raise ValueError('Invalid measured time, command, force or velocity arrays')
    du = np.abs(np.diff(u))
    dv = np.abs(np.diff(v))
    changes = np.flatnonzero(du > max(.02*float(np.ptp(u)), 1e-9))
    if len(changes) == 0:
        changes = np.flatnonzero(dv > max(.05*float(np.ptp(v)), 1e-4))
    if len(changes) == 0 or changes[0]+1 < 3 or len(t)-changes[0]-1 < 8:
        raise ValueError('Record a settled baseline (>=3 samples) and >=8 later samples')
    baseline_count = int(changes[0]+1)
    if np.max(np.abs(f[:baseline_count]-np.median(f[:baseline_count]))) > (
        .1*max(np.ptp(f), 1e-9)):
        raise ValueError('Pre-step force baseline is not settled')
    if (np.any(u < actuator.command_min-1e-9)
        or np.any(u > actuator.command_max+1e-9)):
        raise ValueError('Measured command exceeds configured actuator bounds')
    n = actuator.A.shape[0]
    if n:
        try:
            state = -np.linalg.solve(actuator.A,
                      actuator.B_command*u[0]+actuator.B_velocity*v[0])
        except np.linalg.LinAlgError as exc:
            raise ValueError('Initial actuator equilibrium is undefined for this state-space model') from exc
    else:
        state = np.zeros(0)
    raw_baseline = actuator.output(state, u[0], v[0])
    offset = float(np.median(f[:baseline_count])-raw_baseline)
    delay = actuator.delay_s
    shifted = t[1:]+delay
    events = np.unique(np.r_[t, shifted[(shifted > t[0]) & (shifted < t[-1])]])
    observed_indices = {float(time):k for k, time in enumerate(t)}
    predicted = np.empty(len(t))
    applied = np.empty(len(t))
    for k, current in enumerate(events):
        index = observed_indices.get(float(current))
        if index is not None:
            ui = min(max(np.searchsorted(t, current-delay, side='right')-1, 0), len(t)-1)
            applied[index] = u[ui]
            raw = actuator.C_force@state+actuator.D_command*u[ui]+actuator.D_velocity*v[index]
            predicted[index] = np.clip(raw+offset, -actuator.force_limit_n,
                                       actuator.force_limit_n)
        if k == len(events)-1:
            break
        following = events[k+1]
        midpoint = (current+following)/2
        ui = min(max(np.searchsorted(t, midpoint-delay, side='right')-1, 0), len(t)-1)
        vi = min(max(np.searchsorted(t, midpoint, side='right')-1, 0), len(t)-1)
        Ad, Bu, Bv, _, _, _ = actuator.discrete(float(following-current))
        state = Ad@state+Bu*u[ui]+Bv*v[vi]
    return predicted, applied, offset, baseline_count


def validate_actuator(model_json, measurements_csv, output_dir, *,
                      max_rmse_n=None, plot=False):
    """Write residual CSV, JSON metrics and optional PNG; never overwrite a run."""
    model_path, measured_path = Path(model_json), Path(measurements_csv)
    actuator, kind, train_hash, training_range = _load_model(model_path)
    measured_hash = hashlib.sha256(measured_path.read_bytes()).hexdigest()
    if train_hash == measured_hash:
        raise ValueError('Validation CSV is identical to the fitting CSV; use a separate measurement')
    t, u, f, v, has_speed = _load_measurements(measured_path)
    if not has_speed and (np.any(actuator.B_velocity) or actuator.D_velocity):
        raise ValueError('Model uses cart speed; CSV requires velocity_m_s column')
    predicted, applied, offset, baseline_count = predict_force_trace(actuator, t, u, f, v)
    residual = f-predicted
    dynamic_range = float(np.ptp(f))
    rmse = float(np.sqrt(np.mean(residual**2)))
    if max_rmse_n is not None:
        max_rmse_n = _finite_number(max_rmse_n, 'max_rmse_n', positive=True)
    warnings = []
    if training_range is not None and (min(u) < min(training_range)-1e-9
                                       or max(u) > max(training_range)+1e-9):
        warnings.append('Validation command extends beyond fitting command range; this is extrapolation')
    if kind == 'fitted_first_order' and np.max(np.abs(v)) > 1e-9:
        warnings.append('Fitted step model did not identify cart-speed dependence')
    if train_hash is None:
        warnings.append('Model has no training-data fingerprint; dataset independence cannot be verified automatically')
    summary = dict(status=('evaluated' if max_rmse_n is None else
                            'threshold_passed' if rmse <= max_rmse_n else 'threshold_failed'),
                   model_kind=kind, model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                   measurement_sha256=measured_hash,
                   independence=('different_content_fingerprint' if train_hash else 'unverified'),
                   samples=len(t), baseline_samples=baseline_count,
                   duration_s=float(t[-1]-t[0]),
                   initial_state_assumption='steady_state_at_initial_input',
                   baseline_offset_n=offset, command_min=float(min(u)),
                   command_max=float(max(u)), velocity_min_m_s=float(min(v)),
                   velocity_max_m_s=float(max(v)), measured_force_range_n=dynamic_range,
                   rmse_n=rmse, mae_n=float(np.mean(np.abs(residual))),
                   max_abs_error_n=float(np.max(np.abs(residual))),
                   mean_error_n=float(np.mean(residual)),
                   predicted_force_saturation_fraction=float(np.mean(
                       np.abs(predicted) >= actuator.force_limit_n-1e-9)),
                   nrmse_percent_of_range=(float(100*rmse/dynamic_range)
                                             if dynamic_range > 0 else None),
                   max_rmse_n=max_rmse_n, warnings=warnings)
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ValueError('Install zcartpole[viz] for --plot') from exc
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    with (path/'validation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'command', 'applied_command', 'velocity_m_s',
                         'measured_force_n', 'predicted_force_n', 'residual_n'])
        writer.writerows(zip(t, u, applied, v, f, predicted, residual))
    (path/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    (path/'model.json').write_bytes(model_path.read_bytes())
    if plot:
        fig, (ax, rx) = plt.subplots(2, 1, sharex=True, figsize=(9, 5),
                                     gridspec_kw={'height_ratios':[3,1]})
        ax.plot(t, f, label='Measured force')
        ax.plot(t, predicted, label='Model prediction')
        ax.set_ylabel('Force (N)'); ax.legend(); ax.grid(True)
        rx.plot(t, residual)
        rx.set(xlabel='Time (s)', ylabel='Error (N)'); rx.grid(True)
        fig.tight_layout(); fig.savefig(path/'validation.png', dpi=140)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Validate an actuator model on a separate measured trace')
    parser.add_argument('model', help='Fitted JSON or complete system configuration JSON')
    parser.add_argument('csv', help='Independent CSV with time_s,command,force_n; optional velocity_m_s')
    parser.add_argument('--output', '-o', required=True, help='New report directory')
    parser.add_argument('--max-rmse-n', type=float, help='Optional engineering acceptance limit in newtons')
    parser.add_argument('--plot', action='store_true', help='Save force and residual PNG (viz dependency)')
    args = parser.parse_args()
    try:
        result = validate_actuator(args.model, args.csv, args.output,
                                   max_rmse_n=args.max_rmse_n, plot=args.plot)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'{result["status"]}: RMSE={result["rmse_n"]:.6g} N, '
          f'peak error={result["max_abs_error_n"]:.6g} N; '
          f'{Path(args.output)/"summary.json"}')
    return 1 if result['status'] == 'threshold_failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
