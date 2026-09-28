"""E2: matched-start and equal-budget theta/z swing-up benchmarks.

E2-A uses one penalized IPOPT call per formulation from the same physical path.
E2-B gives each formulation two calls per seed (unpenalized then penalized),
with one seed per tested k plus a paired native seed. z has no winding
constraint: k labels initial paths only.
Failures and every solver call remain in the JSON; only valid solutions are ranked.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from zcartpole import ZNCartPole
from zcartpole.opt import build_warm_start, optimize_swingup, unit_circle_error
from zcartpole.phase_opt import theta_optimize_swingup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "e2_fair_benchmark.json"
N = 100
T = 6.0
TH0 = (np.pi + 0.05, np.pi + 0.03, np.pi + 0.02)
W_PENALTY = 3e-3
UNIT_TOL = 1e-3  # maximum nodewise |c²+s²-1| for unconstrained HS discretization
ALIAS_TOL = 0.25  # revolutions; integral vs. unwrapped node angles
X_MAX_GRID = [0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12]
K_GRID = [0, 1, 2, 3]
MODEL_PARAMS = dict(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4), u_max=80.0)


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
def z_solution_windings(X, n, T=None, return_diagnostics=False):
    """Measure net revolutions from angular velocity, with unwrap cross-check.

    The old unwrap-only method aliases whenever adjacent nodes turn by >= pi.
    Simpson quadrature uses node angular velocities and an interpolated
    midpoint velocity. Disagreement flags a mesh that needs refinement.
    """
    X = np.asarray(X, dtype=float)
    out, diagnostics = [], []
    for i in range(n):
        c, s = X[2+3*i], X[3+3*i]
        w = X[4+3*i]
        angles = np.unwrap(np.arctan2(s, c))
        unwrapped = float((angles[-1]-angles[0])/(2*np.pi))
        if T is None:
            # Backwards-compatible API; callers requiring reliable winding
            # classification must supply the trajectory duration.
            value = unwrapped
            discrepancy = None
        else:
            h = T/(X.shape[1]-1)
            # Trapezoidal angular-velocity integral: independent of atan2 branch.
            value = float(h*(0.5*w[0]+np.sum(w[1:-1])+0.5*w[-1])/(2*np.pi))
            discrepancy = abs(value-unwrapped)
        out.append(value)
        diagnostics.append(dict(unwrap=unwrapped, integral=value,
                                discrepancy=discrepancy,
                                max_node_angle_step=float(np.max(np.abs(np.diff(angles))))) )
    return (out, diagnostics) if return_diagnostics else out


def validate_z(model, X, T=T, unit_tol=UNIT_TOL, alias_tol=ALIAS_TOL):
    """Only numerical/physical validity; NO prescribed winding or k."""
    unit_error = unit_circle_error(model, X)
    turns, diag = z_solution_windings(X, model.n, T=T,
                                     return_diagnostics=True)
    unit_ok = np.isfinite(unit_error) and unit_error < unit_tol
    alias_ok = all(np.isfinite(d["discrepancy"]) and
                   d["discrepancy"] < alias_tol for d in diag)
    return dict(valid=bool(unit_ok and alias_ok), unit_ok=bool(unit_ok),
                alias_ok=bool(alias_ok), unit_circle_error=unit_error,
                unit_tolerance=unit_tol, alias_tolerance=alias_tol,
                achieved_windings=turns, winding_diagnostics=diag)


def seed_waypoints(th0, k):
    """An endpoint-consistent physical angular path; high k uses a pump."""
    if k <= 1:
        return [[(1.0, 2 * np.pi * k)] for _ in th0]
    return [[(0.35, angle + 0.9 * np.pi),
             (0.55, angle + 0.5 * np.pi),
             (1.0, 2 * np.pi * k)] for angle in th0]


def paired_seed(model, T, N, th0, k, x_max):
    """Build the same x, v, angle, angular velocity and force in both coordinates."""
    wp = seed_waypoints(th0, k)
    zX, U = build_warm_start(model, T, N, th0, wp,
                            x_amp=min(0.3, 0.9 * x_max))
    thX = np.zeros((2 + 2 * model.n, N + 1))
    thX[:2] = zX[:2]
    tt = np.linspace(0, 1, N + 1)
    for i in range(model.n):
        pts = [(0.0, th0[i])] + wp[i]
        angle = np.interp(tt, [p[0] for p in pts], [p[1] for p in pts])
        thX[2 + 2*i] = angle
        thX[3 + 2*i] = zX[4 + 3*i]
    return (thX, U.copy()), (zX, U.copy())


def native_pair_seed(model, T, N, th0, x_max):
    """Theta's physical counterpart of z's built-in no-warm initial values.

    z is passed warm=None so its own default initialization is exercised.
    The native angular path ends near 2*pi and is paired with theta k=1.
    """
    tt = np.linspace(0, 1, N + 1)
    thX = np.zeros((2 + 2*model.n, N + 1))
    thX[0] = min(0.3, 0.9*x_max) * np.sin(2*np.pi*tt)
    for i in range(model.n):
        thX[2+2*i] = th0[i] + (np.pi - 0.05)*tt
        thX[3+2*i] = np.pi/T
    return thX, np.zeros(N+1)


def _finite_trajectory(X, U):
    return bool(np.all(np.isfinite(X)) and np.all(np.isfinite(U)))


def solve_theta(model, xm, k, warm, penalty, T=T, N=N, th0=TH0):
    ok, _, X, U, J, iters, elapsed = theta_optimize_swingup(
        model, T, N, xm, th0, (k,) * model.n, penalty, warm)
    valid = bool(ok and J is not None and np.isfinite(J)
                 and _finite_trajectory(X, U))
    return dict(ok=bool(ok), valid=valid, J=float(J) if valid else None,
                iter=iters, solve_s=elapsed), (X, U)


def solve_z(model, xm, warm, penalty, T=T, N=N, th0=TH0):
    info = {}
    started = time.perf_counter()
    ok, _, X, U, J = optimize_swingup(model, T=T, N=N, x_max=xm,
                                       th0=th0, w_penalty=penalty,
                                       warm=warm, info=info)
    elapsed = time.perf_counter() - started
    validation = validate_z(model, X, T=T) if ok and _finite_trajectory(X, U) else None
    valid = bool(ok and J is not None and np.isfinite(J)
                 and validation and validation['valid'])
    calls = info.get('calls', [])
    if len(calls) != 1:
        raise RuntimeError(f'Expected exactly one z IPOPT call, got {len(calls)}')
    return dict(ok=bool(ok), valid=valid, J=float(J) if valid else None,
                iter=calls[0]['iter'], solve_s=elapsed, validation=validation,
                achieved_windings=(validation['achieved_windings']
                                   if validation else None)), (X, U)


def single_pair(model, xm, k, T=T, N=N, th0=TH0):
    th_seed, z_seed = paired_seed(model, T, N, th0, k, xm)
    theta, _ = solve_theta(model, xm, k, th_seed, W_PENALTY, T, N, th0)
    z, _ = solve_z(model, xm, z_seed, W_PENALTY, T, N, th0)
    return dict(k=k, x_max=xm, theta=theta, z=z)


def two_stage(model, xm, k, seed, family, T=T, N=N, th0=TH0,
              seed_name=None):
    """Exactly two IPOPT calls, even if the unpenalized call fails.

    A failed preliminary iterate is still used for the penalized call. Both
    stages and the failure remain visible; no success is attributed to stage 0.
    """
    solve = solve_theta if family == 'theta' else solve_z
    first, intermediate = (solve(model, xm, k, seed, 0.0, T, N, th0)
                           if family == 'theta' else
                           solve(model, xm, seed, 0.0, T, N, th0))
    second, _ = (solve(model, xm, k, intermediate, W_PENALTY, T, N, th0)
                 if family == 'theta' else
                 solve(model, xm, intermediate, W_PENALTY, T, N, th0))
    return dict(k=k, seed=seed_name or f'k={k}',
                stages=[dict(penalty=0.0, **first),
                             dict(penalty=W_PENALTY, **second)],
                ok=second['ok'], valid=second['valid'], J=second['J'],
                achieved_windings=second.get('achieved_windings'),
                calls=2, iterations=sum(c['iter'] or 0 for c in (first, second)),
                solve_s=first['solve_s'] + second['solve_s'])


def equal_budget(model, xm, ks, T=T, N=N, th0=TH0):
    theta, z = [], []
    for k in ks:
        th_seed, z_seed = paired_seed(model, T, N, th0, k, xm)
        theta.append(two_stage(model, xm, k, th_seed, 'theta', T, N, th0))
        z.append(two_stage(model, xm, k, z_seed, 'z', T, N, th0))
    theta.append(two_stage(model, xm, 1, native_pair_seed(model, T, N, th0, xm),
                           'theta', T, N, th0, seed_name='native'))
    z.append(two_stage(model, xm, 1, None, 'z', T, N, th0,
                       seed_name='native'))
    def best(rows):
        valid = [r for r in rows if r['valid']]
        return min(valid, key=lambda r: r['J']) if valid else None
    bt, bz = best(theta), best(z)
    return dict(x_max=xm, theta=theta, z=z,
                theta_best_seed=bt['seed'] if bt else None,
                z_best_seed=bz['seed'] if bz else None,
                theta_best_J=bt['J'] if bt else None,
                z_best_J=bz['J'] if bz else None,
                z_best_windings=bz['achieved_windings'] if bz else None,
                theta_calls=sum(r['calls'] for r in theta),
                z_calls=sum(r['calls'] for r in z),
                theta_iterations=sum(r['iterations'] for r in theta),
                z_iterations=sum(r['iterations'] for r in z),
                theta_s=sum(r['solve_s'] for r in theta),
                z_s=sum(r['solve_s'] for r in z))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fast', action='store_true', help='Only x_max=0.5 and k=0,1 plus native')
    ap.add_argument('--part', choices=['a', 'b', 'both'], default='both')
    ap.add_argument('--x-max', type=float, nargs='+', help='Override cart limits')
    ap.add_argument('--k', type=int, nargs='+', help='Override winding seed labels')
    ap.add_argument('--N', type=int, default=N, help='Collocation intervals (default 100)')
    ap.add_argument('--output', type=Path, default=OUT)
    args = ap.parse_args(argv)
    ks = list(dict.fromkeys(args.k if args.k is not None else ([0, 1] if args.fast else K_GRID)))
    xgrid = args.x_max if args.x_max is not None else ([0.5] if args.fast else X_MAX_GRID)
    if not ks or any(k < 0 for k in ks) or not xgrid or any(x <= 0 for x in xgrid) or args.N < 2:
        ap.error('k must be nonnegative, x_max positive, and N >= 2')
    model = ZNCartPole(**MODEL_PARAMS)
    result = dict(settings=dict(T=T, N=args.N, th0=list(TH0), x_max_grid=xgrid,
                                k_seed_grid=ks, w_penalty=W_PENALTY,
                                unit_tolerance=UNIT_TOL, alias_tolerance=ALIAS_TOL,
                                model=MODEL_PARAMS),
                  interpretation='A: one penalized call per paired physical seed. '
                  'B: two IPOPT calls per seed and formulation, including a paired native seed; '
                  'z terminal winding is free. '
                  'Unsuccessful calls are recorded; best means best valid observed, not global optimum.',
                  E2_A=[], E2_B=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for xm in xgrid:
        for k in ks:
            if args.part in ('a', 'both'):
                row = single_pair(model, xm, k, T, args.N, TH0)
                result['E2_A'].append(row)
                print(f"[E2-A] k={k} x_max={xm} theta J={row['theta']['J']} "
                      f"ok={row['theta']['ok']} | z J={row['z']['J']} "
                      f"valid={row['z']['valid']} winding={row['z']['achieved_windings']}", flush=True)
                args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        if args.part in ('b', 'both'):
            row = equal_budget(model, xm, ks, T, args.N, TH0)
            result['E2_B'].append(row)
            print(f"[E2-B] x_max={xm} theta={row['theta_calls']} calls "
                  f"z={row['z_calls']} calls; best seed: "
                  f"theta={row['theta_best_seed']} z={row['z_best_seed']}", flush=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Wrote: {args.output}')
    return result


if __name__ == '__main__':
    main()
