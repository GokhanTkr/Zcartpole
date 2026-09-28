"""Angle-coordinate swing-up NLP with fixed or periodic terminal winding.

CasADi is imported only when solving. This module is shared by the public
planner and the research benchmarks so both use the same transcription.
"""
import time
import numpy as np

from .symbolic import symbolic_accel


def build_phase_seed(model, T, N, th0, k, x_max, x0=0.0, v0=0.0, w0=None):
    """Make an unwrapped phase guess; k labels a seed, not a terminal constraint."""
    tt = np.linspace(0, 1, N + 1)
    X = np.zeros((2 + 2*model.n, N + 1))
    amp = min(0.3, 0.9*x_max)
    X[0] = amp*np.sin(2*np.pi*tt)
    X[1] = amp*(2*np.pi/T)*np.cos(2*np.pi*tt)
    for i in range(model.n):
        if k <= 1:
            points = [(0.0, th0[i]), (1.0, 2*np.pi*k)]
        else:
            points = [(0.0, th0[i]), (0.35, th0[i]+0.9*np.pi),
                      (0.55, th0[i]+0.5*np.pi), (1.0, 2*np.pi*k)]
        angle = np.interp(tt, [p[0] for p in points], [p[1] for p in points])
        X[2+2*i] = angle
        X[3+2*i] = np.gradient(angle, tt)/T
    X[0, 0], X[1, 0] = x0, v0
    if w0 is not None:
        X[3::2, 0] = w0
    return X, np.zeros(N + 1)


def theta_optimize_swingup(model, T, N, x_max, th0, windings,
                           w_penalty=0.0, warm=None, max_iter=3000,
                           x0=0.0, v0=0.0, w0=None):
    """Solve a continuous-angle Hermite-Simpson swing-up problem.
    
    windings fixes final whole turns per link; None leaves them free.
    Return (ok, times, states, forces, cost, iterations, solve_seconds)."""
    import casadi as ca

    n = model.n
    dim = 2 + 2 * n

    def f(X, u):
        th = [X[2 + 2 * i] for i in range(n)]
        ws = [X[3 + 2 * i] for i in range(n)]
        cs = [ca.cos(t) for t in th]
        ss = [ca.sin(t) for t in th]

        a = symbolic_accel(model, cs, ss, ws, u, X[1])
        deriv = [X[1], a[0]]
        for i in range(n):
            deriv += [ws[i], a[i + 1]]
        return ca.vertcat(*deriv)

    opti = ca.Opti()
    X, U = opti.variable(dim, N + 1), opti.variable(1, N + 1)
    h = T / N
    for k in range(N):
        fk, fk1 = f(X[:, k], U[:, k]), f(X[:, k + 1], U[:, k + 1])
        Xm = 0.5 * (X[:, k] + X[:, k + 1]) + h / 8 * (fk - fk1)
        opti.subject_to(X[:, k + 1] - X[:, k]
                        == h / 6 * (fk + 4 * f(Xm, 0.5 * (U[:, k] + U[:, k + 1])) + fk1))

    w0 = np.zeros(n) if w0 is None else np.asarray(w0, dtype=float)
    x0_state = [x0, v0]
    xN_state = [0, 0]
    for i in range(n):
        x0_state += [th0[i], w0[i]]
        if windings is not None:
            xN_state += [2 * np.pi * windings[i], 0]
    opti.subject_to(X[:, 0] == x0_state)
    if windings is None:
        opti.subject_to(X[0:2, N] == [0, 0])
        for i in range(n):
            opti.subject_to(X[3 + 2*i, N] == 0)
            opti.subject_to(ca.sin(X[2 + 2*i, N]/2) == 0)
    else:
        opti.subject_to(X[:, N] == xN_state)
    opti.subject_to(opti.bounded(-x_max, X[0, :], x_max))
    opti.subject_to(opti.bounded(-model.u_max, U, model.u_max))
    opti.subject_to(U[:, N] == 0)
    w_pen = sum(ca.sumsqr(X[3 + 2 * i, :]) for i in range(n))
    opti.minimize(h * ca.sumsqr(U) / model.u_max ** 2 + 1e-3 * h * ca.sumsqr(X[1, :])
                 + w_penalty * h * w_pen)

    tt = np.linspace(0, 1, N + 1)
    if warm is not None:
        X0, U0 = warm
        opti.set_initial(X, X0)
        opti.set_initial(U, U0)
    else:
        for i in range(n):
            target = 2*np.pi*(windings[i] if windings is not None else 1)
            opti.set_initial(X[2 + 2 * i, :], th0[i] + (target - th0[i]) * tt)
            opti.set_initial(X[3 + 2 * i, :], (target - th0[i]) / T)
        opti.set_initial(X[0, :], min(0.3, 0.9 * x_max) * np.sin(2 * np.pi * tt))
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                          'ipopt.max_iter': max_iter, 'ipopt.tol': 1e-6,
                          'ipopt.hessian_approximation': 'limited-memory'})

    t0 = time.perf_counter()
    try:
        sol = opti.solve()
        solve_s = time.perf_counter() - t0
        return True, np.linspace(0, T, N + 1), sol.value(X), sol.value(U), \
            float(sol.value(opti.f)), int(sol.stats()['iter_count']), solve_s
    except RuntimeError:
        solve_s = time.perf_counter() - t0
        stats = opti.debug.stats()
        n_iter = int(stats['iter_count']) if 'iter_count' in stats else None
        return False, np.linspace(0, T, N + 1), opti.debug.value(X), opti.debug.value(U), \
            None, n_iter, solve_s
