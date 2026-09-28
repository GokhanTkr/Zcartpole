"""Legacy complex-coordinate Hermite-Simpson swing-up optimizer.

CasADi is imported only when optimization is requested. The recommended
entry point for new runs is plan_swingup."""
import warnings

import numpy as np

from .symbolic import symbolic_accel


def unit_circle_error(model, X):
    """Largest |c²+s²-1| at returned collocation nodes."""
    X = np.asarray(X, dtype=float)
    return float(max(np.max(np.abs(X[2+3*i]**2 + X[3+3*i]**2 - 1))
                     for i in range(model.n)))


def _unit_nodes(model, X, tolerance=1e-5):
    """Reject a materially off-manifold solution; do not silently alter its path.

    Hermite-Simpson does not preserve quadratic invariants exactly. Imposing
    exact unit-circle equality at every node can overconstrain the discrete NLP.
    A tiny drift is acceptable, but post-solve normalization changes the
    optimized trajectory and can invalidate its collocation residuals.
    """
    X = np.asarray(X, dtype=float)
    error = unit_circle_error(model, X)
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(f"Unit-circle drift {error:.3g} exceeds {tolerance:.3g}")
    return X


def _angle_quadrature(w0, wm, w1, h):
    """Quadratic angular-velocity quadrature to midpoint and end of a step.

    Works with both numerical values and CasADi expressions. The midpoint
    velocity is the Hermite-Simpson midpoint estimate.
    """
    return h * (5*w0 + 8*wm - w1) / 24, h * (w0 + 4*wm + w1) / 6


def build_warm_start(model, T, N, th0, waypoints, x_amp=0.3):
    """Build a manual initial path for the legacy optimizer.
    
    Give one list of (time_fraction, absolute_angle_rad) waypoints per link.
    The first point is added at the actual initial angle. The final angle
    should be an absolute multiple of 2*pi, not initial_angle + 2*pi*k.
    Return (state_guess, force_guess) for optimize_swingup(warm=...)."""
    n = model.n
    tt = np.linspace(0, 1, N + 1)
    X0 = np.zeros((2 + 3 * n, N + 1))
    X0[0, :] = x_amp * np.sin(2 * np.pi * tt)
    X0[1, :] = x_amp * (2 * np.pi / T) * np.cos(2 * np.pi * tt)
    for i in range(n):
        pts = [(0.0, th0[i])] + list(waypoints[i])
        ts = np.array([p[0] for p in pts], dtype=float)
        angs = np.array([p[1] for p in pts], dtype=float)
        order = np.argsort(ts)
        ts, angs = ts[order], angs[order]
        ang = np.interp(tt, ts, angs)
        w = np.gradient(ang, tt) / T
        X0[2 + 3 * i, :] = np.cos(ang)
        X0[3 + 3 * i, :] = np.sin(ang)
        X0[4 + 3 * i, :] = w
    U0 = np.zeros(N + 1)
    return X0, U0


def optimize_swingup(model, T=None, N=100, x_max=0.5, th0=None, w_penalty=None, warm=None,
                     info=None, integrator='hs'):
    """Find a locally optimized swing-up path with sampled rail and force bounds.
    
    T is the horizon, N is the number of intervals, th0 holds initial angles,
    and warm may supply (state_guess, force_guess). The optional info dict
    records solver calls. integrator selects Hermite-Simpson or the experimental
    Lie update. Return (ok, times, states, forces, cost)."""
    import casadi as ca

    n = model.n
    if T is None:
        T = 3.0 * max(n - 1, 1)
    if th0 is None:
        th0 = tuple(np.pi + 0.05 - 0.015 * i for i in range(n))
    if w_penalty is None:
        w_penalty = 0.0 if n <= 2 else 3e-3
    if integrator not in ('hs', 'lie'):
        raise ValueError("integrator must be 'hs' or 'lie'")

    def f(X, u):
        cs = [X[2 + 3 * i] for i in range(n)]
        ss = [X[3 + 3 * i] for i in range(n)]
        ws = [X[4 + 3 * i] for i in range(n)]

        a = symbolic_accel(model, cs, ss, ws, u, X[1])
        deriv = [X[1], a[0]]
        for i in range(n):
            deriv += [-ws[i] * ss[i], ws[i] * cs[i], a[i + 1]]
        return ca.vertcat(*deriv)

    dim = 2 + 3 * n

    def build_and_solve(w_pen_weight, warm=None):
        opti = ca.Opti()
        X, U = opti.variable(dim, N + 1), opti.variable(1, N + 1)
        h = T / N
        for k in range(N):
            fk, fk1 = f(X[:, k], U[:, k]), f(X[:, k + 1], U[:, k + 1])
            Xm = 0.5 * (X[:, k] + X[:, k + 1]) + h / 8 * (fk - fk1)
            if integrator == 'lie':
                parts = [Xm[j] for j in range(dim)]
                increments = []
                for i in range(n):
                    ci, si, wi = 2 + 3*i, 3 + 3*i, 4 + 3*i
                    half, full = _angle_quadrature(X[wi, k], Xm[wi], X[wi, k+1], h)
                    c0, s0 = X[ci, k], X[si, k]
                    parts[ci] = c0*ca.cos(half) - s0*ca.sin(half)
                    parts[si] = s0*ca.cos(half) + c0*ca.sin(half)
                    increments.append((ci, si, full))
                Xm = ca.vertcat(*parts)
            fm = f(Xm, 0.5 * (U[:, k] + U[:, k + 1]))
            if integrator == 'hs':
                opti.subject_to(X[:, k + 1] - X[:, k]
                                == h / 6 * (fk + 4*fm + fk1))
            else:
                rows = [0, 1] + [4 + 3*i for i in range(n)]
                opti.subject_to(X[rows, k + 1] - X[rows, k]
                                == h / 6 * (fk[rows] + 4*fm[rows] + fk1[rows]))
                for ci, si, full in increments:
                    c0, s0 = X[ci, k], X[si, k]
                    opti.subject_to(X[ci, k+1] == c0*ca.cos(full) - s0*ca.sin(full))
                    opti.subject_to(X[si, k+1] == s0*ca.cos(full) + c0*ca.sin(full))
        x0_state = [0, 0]
        xN_state = [0, 0]
        for i in range(n):
            x0_state += [np.cos(th0[i]), np.sin(th0[i]), 0]
            xN_state += [1, 0, 0]
        opti.subject_to(X[:, 0] == x0_state)
        opti.subject_to(X[:, N] == xN_state)
        opti.subject_to(opti.bounded(-x_max, X[0, :], x_max))
        opti.subject_to(opti.bounded(-model.u_max, U, model.u_max))
        opti.subject_to(U[:, N] == 0)                       
        w_pen = sum(ca.sumsqr(X[4 + 3 * i, :]) for i in range(n))
        opti.minimize(h * ca.sumsqr(U) / model.u_max ** 2 + 1e-3 * h * ca.sumsqr(X[1, :])
                     + w_pen_weight * h * w_pen)

        if warm is None:
            tt = np.linspace(0, 1, N + 1)
            for i in range(n):
                
                
                
                ang = th0[i] + (np.pi - 0.05) * tt
                opti.set_initial(X[2 + 3 * i, :], np.cos(ang))
                opti.set_initial(X[3 + 3 * i, :], np.sin(ang))
                opti.set_initial(X[4 + 3 * i, :], np.pi / T)
            
            
            
            
            
            
            
            
            x_amp0 = min(0.3, 0.9 * x_max)
            opti.set_initial(X[0, :], x_amp0 * np.sin(2 * np.pi * tt))
        else:
            opti.set_initial(X, warm[0])
            opti.set_initial(U, warm[1])
        
        
        
        
        
        
        
        opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                              'ipopt.max_iter': 3000, 'ipopt.tol': 1e-6,
                              'ipopt.hessian_approximation': 'limited-memory'})
        try:
            sol = opti.solve()
            n_iter = int(sol.stats().get('iter_count', -1))
            return True, sol.value(X), sol.value(U), float(sol.value(opti.f)), n_iter
        except RuntimeError:
            stats = opti.debug.stats()
            n_iter = int(stats['iter_count']) if 'iter_count' in stats else None
            return False, opti.debug.value(X), opti.debug.value(U), None, n_iter

    # Keep the raw collocation solution and report manifold drift explicitly.
    # Do not add exact unit constraints at every node: the discretization does
    # not preserve this invariant exactly, so that can make the NLP inconsistent.
    if info is not None:
        info.clear()
        info['calls'] = []

    def record(stage, ok, n_iter):
        if info is not None:
            info['calls'].append({'stage': stage, 'ok': bool(ok), 'iter': n_iter})

    tn = np.linspace(0, T, N + 1)
    if warm is not None:
        
        
        
        ok, X, U, J, n_iter = build_and_solve(w_penalty, warm=warm)
        record('warm_single', ok, n_iter)
        return ok, tn, X, U, J
    if w_penalty <= 0.0:
        ok, X, U, J, n_iter = build_and_solve(0.0)
        record('single_no_penalty', ok, n_iter)
        return ok, tn, X, U, J

    ok0, X0, U0, _, n_iter0 = build_and_solve(0.0)          
    record('stage0_prewarm', ok0, n_iter0)
    if not ok0:
        
        
        
        
        
        
        
        warnings.warn(
            "First unpenalized pass did not converge; the next pass "
            "starts from its final iterate. Even a successful result "
            "may depend on this recovery.",
            RuntimeWarning, stacklevel=2)
    ok, Xr, Ur, J, n_iter1 = build_and_solve(w_penalty, warm=(X0, U0))  
    record('stage1_refine', ok, n_iter1)
    return ok, tn, Xr, Ur, J
