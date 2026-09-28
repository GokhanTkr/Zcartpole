"""One-call, winding-free swing-up planning for a cart with an n-link pendulum.

The phase is unwrapped inside the optimizer. A periodic terminal equality
accepts every integer winding; seed labels only choose initial trajectories.
The selected path is converted to the existing z state layout for tracking.
"""
from dataclasses import dataclass
import time
from typing import Optional, Tuple

import numpy as np

from .phase_opt import build_phase_seed, theta_optimize_swingup


@dataclass(frozen=True)
class SwingupAttempt:
    seed_k: int
    solver_ok: bool
    valid: bool
    J: Optional[float]
    windings: Optional[Tuple[int, ...]]
    iterations: Optional[int]
    seconds: float
    reason: str
    terminal_error: Optional[float] = None
    max_cart: Optional[float] = None
    max_force: Optional[float] = None


@dataclass(frozen=True)
class CatchReport:
    """Measured closed-loop outcome, with explicit terminal tolerances."""
    success: bool
    caught: bool
    within_limits: bool
    final_cart: float
    final_cart_speed: float
    final_angle_error: float
    final_angular_speed: float
    max_cart: float
    max_force: float
    peak_tracking_angle_error: float


@dataclass
class SwingupResult:
    success: bool
    message: str
    attempts: Tuple[SwingupAttempt, ...]
    th0: Tuple[float, ...]
    x_max: float
    cart_margin: float = 0.0
    t: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None       # existing z layout: x,v,c,s,w,...
    U: Optional[np.ndarray] = None
    phase_X: Optional[np.ndarray] = None # unwrapped x,v,theta,w,...
    J: Optional[float] = None
    windings: Optional[Tuple[int, ...]] = None
    seed_k: Optional[int] = None
    x0: float = 0.0
    v0: float = 0.0
    w0: Optional[Tuple[float, ...]] = None

    def tracker(self, model):
        """Build the existing TVLQR tracker from a successful plan."""
        if not self.success:
            raise RuntimeError('No valid swing-up trajectory is available')
        from .core import TVLQRTracker
        return TVLQRTracker(model, self.t, self.X, self.U)

    def simulate_catch(self, model, t_extra=3.0, dt=1e-3):
        """Simulate TVLQR tracking; return arrays ``t``, ``y`` and ``u``."""
        if (not np.isfinite(t_extra) or t_extra < 0 or
                not np.isfinite(dt) or dt <= 0):
            raise ValueError('t_extra must be nonnegative and dt positive')
        from .core import simulate
        tracker = self.tracker(model)
        return simulate(model, tracker.control, th0=self.th0,
                        x0=self.x0, v0=self.v0, w0=self.w0,
                        t_max=float(self.t[-1]) + t_extra, dt=dt)

    def evaluate_catch(self, model, sol=None, *, cart_tol=0.05,
                       speed_tol=0.1, angle_tol=0.1, omega_tol=0.2,
                       limit_tol=1e-5):
        """Measure the closed-loop path and its terminal upright condition.

        Pass a previously simulated ``sol`` to avoid running simulation twice.
        ``success`` requires terminal capture and trajectory-wide cart/force
        limits. These thresholds are user choices, not stability guarantees.
        """
        if not self.success:
            raise RuntimeError('No valid swing-up trajectory is available')
        tolerances = (cart_tol, speed_tol, angle_tol, omega_tol, limit_tol)
        if any(not np.isfinite(a) or a < 0 for a in tolerances):
            raise ValueError('Tolerances must be finite and nonnegative')
        if sol is None:
            sol = self.simulate_catch(model)
        t = np.asarray(sol.t, dtype=float)
        Y = np.asarray(sol.y, dtype=float)
        U = np.asarray(sol.u, dtype=float)
        if (t.ndim != 1 or t.size < 2 or Y.shape != (2+3*model.n, t.size)
                or U.shape != (t.size,) or not np.isfinite(t).all()
                or not np.isfinite(Y).all() or not np.isfinite(U).all()
                or np.any(np.diff(t) <= 0) or t[-1] < self.t[-1]-1e-8):
            raise ValueError('Invalid simulation trajectory')

        angles = np.arctan2(Y[3::3], Y[2::3])
        last_angle = float(np.max(np.abs(angles[:, -1])))
        last_omega = float(np.max(np.abs(Y[4::3, -1])))
        last_cart = float(abs(Y[0, -1]))
        last_speed = float(abs(Y[1, -1]))
        max_cart = float(np.max(np.abs(Y[0])))
        max_force = float(np.max(np.abs(U)))
        mask = t <= self.t[-1]
        nominal_angles = np.array([
            np.interp(t[mask], self.t, self.phase_X[2+2*i])
            for i in range(model.n)])
        delta = angles[:, mask] - nominal_angles
        peak_tracking = float(np.max(np.abs(np.arctan2(np.sin(delta), np.cos(delta)))))
        caught = (last_cart <= cart_tol and last_speed <= speed_tol
                  and last_angle <= angle_tol and last_omega <= omega_tol)
        within = (max_cart <= self.x_max + limit_tol
                  and max_force <= model.u_max + limit_tol)
        return CatchReport(bool(caught and within), bool(caught), bool(within),
                           last_cart, last_speed, last_angle, last_omega,
                           max_cart, max_force, peak_tracking)


def _phase_to_z(X, n):
    X = np.asarray(X, dtype=float)
    z = np.empty((2 + 3*n, X.shape[1]))
    z[:2] = X[:2]
    for i in range(n):
        z[2+3*i] = np.cos(X[2+2*i])
        z[3+3*i] = np.sin(X[2+2*i])
        z[4+3*i] = X[3+2*i]
    return z


def _check_solution(model, X, U, J, x_max):
    """Check the returned node values before exposing a trajectory for tracking."""
    if J is None or not np.isfinite(J):
        return False, 'nonfinite_objective', None, None, None, None
    X, U = np.asarray(X, dtype=float), np.asarray(U, dtype=float).ravel()
    if X.shape[0] != 2+2*model.n or X.shape[1] != U.size or not (
        np.isfinite(X).all() and np.isfinite(U).all()
    ):
        return False, 'nonfinite_trajectory', None, None, None, None
    turns = X[2:2+2*model.n:2, -1]/(2*np.pi)
    winding = tuple(int(round(t)) for t in turns)
    terminal_error = float(max(np.max(np.abs(turns-np.round(turns))),
                               np.max(np.abs(X[[0, 1] + [3+2*i for i in range(model.n)], -1])),
                               abs(U[-1])))
    max_cart = float(np.max(np.abs(X[0])))
    max_force = float(np.max(np.abs(U)))
    valid = (terminal_error < 1e-4 and max_cart <= x_max+1e-5
             and max_force <= model.u_max+1e-5)
    return (bool(valid), 'ok' if valid else 'constraint_violation', winding,
            terminal_error, max_cart, max_force)


def plan_swingup(model, *, T=None, N=100, x_max=0.5, th0=None,
                 seeds=(0, 1), w_penalty=None, max_iter=1000,
                 cart_margin=0.005, x0=0.0, v0=0.0, w0=None):
    """Find the cheapest valid local swing-up among winding-shaped phase seeds.

    No winding is fixed in the NLP. The returned ``X`` uses the model's z
    layout and works with ``TVLQRTracker``; ``phase_X`` preserves full turns.
    Failed seeds remain in ``attempts``. The nominal cart path is bounded by
    ``x_max-cart_margin`` to leave tracking room; closed-loop feasibility is
    measured separately with ``evaluate_catch``. CasADi is needed only here.
    """
    if T is None:
        T = 3.0*max(model.n-1, 1)
    if th0 is None:
        th0 = tuple(np.pi+0.05-0.015*i for i in range(model.n))
    th0 = tuple(float(a) for a in th0)
    w0 = tuple(0.0 for _ in range(model.n)) if w0 is None else tuple(w0)
    if w_penalty is None:
        w_penalty = 0.0 if model.n <= 2 else 3e-3
    seed_ks = tuple(dict.fromkeys(seeds))
    if (not np.isfinite(T) or T <= 0 or not isinstance(N, int) or N < 2
        or not np.isfinite(x_max) or x_max <= 0
        or not np.isfinite(cart_margin) or cart_margin < 0
        or cart_margin >= x_max or len(th0) != model.n
        or not np.isfinite(th0).all() or len(w0) != model.n
        or not np.isfinite(w0).all() or not np.isfinite(x0)
        or abs(x0) >= x_max-cart_margin or not np.isfinite(v0)
        or not np.isfinite(w_penalty)
        or w_penalty < 0 or not isinstance(max_iter, int) or max_iter < 1
        or not seed_ks or any(type(k) is not int or k < 0 for k in seed_ks)):
        raise ValueError('Invalid T, N, x_max, cart_margin, th0, w_penalty, max_iter or seeds')

    nominal_bound = x_max-cart_margin
    attempts = []
    best = None
    for k in seed_ks:
        initial = build_phase_seed(model, T, N, th0, k, nominal_bound,
                                   x0=x0, v0=v0, w0=w0)
        start = time.perf_counter()
        ok, t, X, U, J, iterations, _ = theta_optimize_swingup(
            model, T, N, nominal_bound, th0, None, w_penalty, initial, max_iter,
            x0=x0, v0=v0, w0=w0)
        seconds = time.perf_counter()-start
        if ok:
            valid, reason, turns, error, cart, force = _check_solution(
                model, X, U, J, nominal_bound)
        else:
            valid, reason, turns, error, cart, force = (
                False, 'ipopt_no_convergence', None, None, None, None)
        attempt = SwingupAttempt(k, bool(ok), valid,
                                 float(J) if valid else None, turns, iterations,
                                 seconds, reason, error, cart, force)
        attempts.append(attempt)
        if valid and (best is None or attempt.J < best[0].J):
            best = attempt, np.asarray(t), np.asarray(X), np.asarray(U).ravel()

    if best is None:
        return SwingupResult(False, 'No valid trajectory; inspect attempts',
                             tuple(attempts), th0, x_max,
                             cart_margin=cart_margin, x0=x0, v0=v0, w0=w0)
    chosen, t, phase_X, U = best
    return SwingupResult(True, 'ok', tuple(attempts), th0, x_max,
                         cart_margin=cart_margin,
                         t=t, X=_phase_to_z(phase_X, model.n), U=U,
                         phase_X=phase_X, J=chosen.J,
                         windings=chosen.windings, seed_k=chosen.seed_k,
                         x0=x0, v0=v0, w0=w0)
