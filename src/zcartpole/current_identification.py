"""Identify a SINAMICS linear motor current loop from an exported PRBS trace.

This identifies current setpoint -> current feedback, never cart force.
The contiguous holdout is another segment of the same acquisition.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import cont2discrete, lfilter


HEADERS = ('X (ms)', 'C0:Lineardirektantrieb.r77',
           'C1:Lineardirektantrieb.r78[0]')


def load_siemens_current_csv(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter=';')
        if tuple(reader.fieldnames or ()) != HEADERS:
            raise ValueError('Expected original Siemens X (ms), r77, r78[0] columns')
        try:
            data = np.asarray([[float(row[key].replace(',', '.')) for key in HEADERS]
                               for row in reader], dtype=float)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError('Invalid numeric sample in Siemens CSV') from exc
    if data.ndim != 2 or data.shape[0] < 100 or not np.isfinite(data).all():
        raise ValueError('At least 100 finite samples are required')
    t, u, y = data.T
    dt = np.diff(t)
    if (np.any(dt <= 0) or np.max(np.abs(dt - np.median(dt))) >
            1e-5*np.median(dt) or np.ptp(u) < 1e-6):
        raise ValueError('Require uniform, increasing time and varying setpoint')
    return (t-t[0])*1e-3, u, y


def _candidate(u, y, dt, ntrain, order, delay_samples, tau):
    n = len(u)
    q = np.zeros(n)
    q[delay_samples:] = u[:n-delay_samples]
    if order == 1:
        decay = np.exp(-dt/tau)
        z = lfilter([1-decay], [1, -decay], np.r_[0, q[:-1]])
        transient = [decay**np.arange(n)]
    else:
        b, a, _ = cont2discrete(([1], [tau*tau, 2*tau, 1]), dt,
                                 method='zoh')
        z = lfilter(b.ravel(), a.ravel(), q)
        decay = np.exp(-dt/tau)
        transient = [decay**np.arange(n),
                     np.arange(n)*decay**np.arange(n)]
    X = np.column_stack([np.ones(n), z, *transient])
    coeff = np.linalg.lstsq(X[:ntrain], y[:ntrain], rcond=None)[0]
    pred = X@coeff
    sse = float(np.sum((pred[:ntrain]-y[:ntrain])**2))
    return sse, coeff, pred


def identify_current_loop(t, u, y, *, train_fraction=.65, max_delay_samples=8):
    t, u, y = (np.asarray(a, dtype=float) for a in (t, u, y))
    if (t.ndim != 1 or u.shape != t.shape or y.shape != t.shape or
            len(t) < 100 or not all(np.isfinite(a).all() for a in (t,u,y)) or
            np.any(np.diff(t) <= 0)):
        raise ValueError('Invalid current trace')
    dt = float(np.median(np.diff(t)))
    if np.max(np.abs(np.diff(t)-dt)) > 1e-5*dt:
        raise ValueError('Sampling interval must be uniform')
    if not .5 <= train_fraction <= .85:
        raise ValueError('train_fraction must be between .5 and .85')
    ntrain = int(len(t)*train_fraction)
    models = []
    for order in (1, 2):
        for delay in range(max_delay_samples+1):
            objective = lambda logtau: _candidate(
                u, y, dt, ntrain, order, delay, np.exp(logtau))[0]
            opt = minimize_scalar(objective, bounds=(np.log(.2*dt),
                                   np.log(50*dt)), method='bounded',
                                   options={'xatol':1e-9})
            tau = float(np.exp(opt.x))
            sse, coeff, pred = _candidate(u, y, dt, ntrain, order, delay, tau)
            parameter_count = order+4  # gain, offset, tau, initial modes, delay
            aic = ntrain*np.log(max(sse/ntrain, 1e-24))+2*parameter_count
            models.append(dict(order=order, delay_samples=delay, tau_s=tau,
                               gain=float(coeff[1]), offset_a=float(coeff[0]),
                               transient_coefficients_a=coeff[2:].tolist(),
                               aic=float(aic), train_rmse_a=float(np.sqrt(sse/ntrain)),
                               prediction=pred))
    chosen = min(models, key=lambda m:m['aic'])
    pred = chosen.pop('prediction')
    residual = y-pred
    validation = residual[ntrain:]
    full_range = float(np.ptp(y[ntrain:]))
    summary = dict(signal='current_setpoint_to_actual_current',
                   signal_unit='A_rms_per_Siemens_parameter_manual',
                   force_calibrated=False,
                   validation_kind='contiguous_holdout_same_experiment',
                   samples=len(t), train_samples=ntrain,
                   holdout_samples=len(t)-ntrain,
                   sampling_interval_s=dt, duration_s=float(t[-1]-t[0]),
                   command_range_a=[float(np.min(u)), float(np.max(u))],
                   measured_range_a=[float(np.min(y)), float(np.max(y))],
                   model_order=chosen['order'],
                   current_gain_a_per_a=chosen['gain'],
                   time_constant_s=chosen['tau_s'],
                   command_delay_s=chosen['delay_samples']*dt,
                   offset_a=chosen['offset_a'],
                   initial_transient_coefficients_a=chosen['transient_coefficients_a'],
                   selection='minimum_training_AIC_over_orders_1_2_and_discrete_delays',
                   train_rmse_a=chosen['train_rmse_a'],
                   holdout_rmse_a=float(np.sqrt(np.mean(validation**2))),
                   holdout_mae_a=float(np.mean(np.abs(validation))),
                   holdout_nrmse_percent_of_range=(100*float(np.sqrt(np.mean(validation**2)))/full_range
                                                    if full_range else None),
                   holdout_baseline_rmse_a=float(np.sqrt(np.mean((y[ntrain:]-np.mean(y[:ntrain]))**2))),
                   holdout_residual_lag1_correlation=float(np.corrcoef(validation[1:],validation[:-1])[0,1]),
                   candidates=[{k:v for k,v in m.items() if k != 'prediction'} for m in models],
                   limitations=['Holdout is from the same PRBS run, not an independent experiment.',
                                'r77 and r78[0] are force-generating currents; no force sensor calibrates N.',
                                'Time delay is resolved only to one sampling interval.',
                                'The transfer model covers the recorded current range and fixed conditions.'])
    if chosen['order'] == 1:
        den = [chosen['tau_s'], 1.]
    else:
        tau = chosen['tau_s']
        den = [tau*tau, 2*tau, 1.]
    summary['current_transfer_function'] = dict(
        numerator=[chosen['gain']], denominator=den,
        command_delay_s=chosen['delay_samples']*dt,
        input_unit='A_rms_setpoint', output_unit='A_rms_feedback')
    return summary, pred


def identify_siemens_csv(csv_path, output_dir, *, plot=False):
    source = Path(csv_path)
    t,u,y = load_siemens_current_csv(source)
    result,pred = identify_current_loop(t,u,y)
    result['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    if plot:
        import matplotlib.pyplot as plt
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out/'current_loop_fit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with (out/'current_loop_prediction.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('time_s','setpoint_a','actual_a','predicted_a','residual_a','segment'))
        writer.writerows((float(t[i]),float(u[i]),float(y[i]),float(pred[i]),
                          float(y[i]-pred[i]), 'train' if i<result['train_samples'] else 'holdout')
                         for i in range(len(t)))
    if plot:
        n=result['train_samples']
        fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True)
        axs[0].plot(t*1e3,u,color='#5574aa',lw=.7)
        axs[0].set_ylabel('Setpoint (A RMS)')
        axs[1].plot(t*1e3,y,label='Measured drive current',lw=.8)
        axs[1].plot(t*1e3,pred,label='Model',lw=.8,alpha=.8)
        axs[1].legend(loc='upper right'); axs[1].set_ylabel('Current (A RMS)')
        axs[2].plot(t*1e3,y-pred,lw=.7)
        axs[2].set_ylabel('Error (A RMS)'); axs[2].set_xlabel('Time (ms)')
        for ax in axs:
            ax.axvline(t[n]*1e3,color='#bf5b4b',ls='--',lw=1)
            ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(out/'current_loop.png',dpi=150)
        plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description='Identify Siemens PRBS current loop (not cart force)')
    parser.add_argument('csv')
    parser.add_argument('--output','-o',required=True)
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    try:
        result=identify_siemens_csv(args.csv,args.output,plot=args.plot)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print(f'current loop: order={result["model_order"]}, '
          f'holdout RMSE={result["holdout_rmse_a"]:.4g} A RMS; '
          f'{Path(args.output)/"current_loop_fit.json"}')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
