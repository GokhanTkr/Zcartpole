"""Legacy two-link reference implementation for regression checks."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_continuous_are


class ZDoubleCartPole:
    def __init__(self, M=1.0, m1=0.1, m2=0.1, l1=0.5, l2=0.5, g=9.81, u_max=50.0):
        self.M, self.m1, self.m2 = M, m1, m2
        self.l1, self.l2, self.g, self.u_max = l1, l2, g, u_max
        self.E_ref = m1 * g * l1 + m2 * g * (l1 + l2)
        self.Q = np.diag([10.0, 200.0, 200.0, 1.0, 10.0, 10.0])
        self.R = np.array([[0.1]])
        self.K, self.P = self.design_lqr()

    # Convert between packed state and complex link states.
    @staticmethod
    def unpack(s):
        return s[0], s[1], s[2] + 1j * s[3], s[4], s[5] + 1j * s[6], s[7]

    @staticmethod
    def pack(x, v, z1, w1, z2, w2):
        return np.array([x, v, z1.real, z1.imag, w1, z2.real, z2.imag, w2])

    # Mechanical equations.
    def mass_matrix(self, z1, z2):
        r = z1 * np.conj(z2)
        m1, m2, l1, l2 = self.m1, self.m2, self.l1, self.l2
        return np.array([
            [self.M + m1 + m2, (m1 + m2) * l1 * z1.real, m2 * l2 * z2.real],
            [(m1 + m2) * l1 * z1.real, (m1 + m2) * l1 ** 2, m2 * l1 * l2 * r.real],
            [m2 * l2 * z2.real, m2 * l1 * l2 * r.real, m2 * l2 ** 2]])

    def rhs_forces(self, z1, w1, z2, w2, u):
        r = z1 * np.conj(z2)
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g
        return np.array([
            u + (m1 + m2) * l1 * w1 ** 2 * z1.imag + m2 * l2 * w2 ** 2 * z2.imag,
            -m2 * l1 * l2 * w2 ** 2 * r.imag + (m1 + m2) * g * l1 * z1.imag,
            m2 * l1 * l2 * w1 ** 2 * r.imag + m2 * g * l2 * z2.imag])

    def accel(self, z1, w1, z2, w2, u):
        """[x_ddot, w1_dot, w2_dot]"""
        return np.linalg.solve(self.mass_matrix(z1, z2), self.rhs_forces(z1, w1, z2, w2, u))

    def total_energy(self, s):
        x, v, z1, w1, z2, w2 = self.unpack(s)
        q = np.array([v, w1, w2])
        T = 0.5 * q @ self.mass_matrix(z1, z2) @ q
        V = (self.m1 * self.g * self.l1 * z1.real
             + self.m2 * self.g * (self.l1 * z1.real + self.l2 * z2.real))
        return T + V

    def momentum(self, s):
        """Yatay toplam momentum (u=0 iken korunur)."""
        x, v, z1, w1, z2, w2 = self.unpack(s)
        return self.mass_matrix(z1, z2)[0] @ np.array([v, w1, w2])

    def step(self, s, u, h):
        """Legacy reference for step."""
        x, v, z1, w1, z2, w2 = self.unpack(s)

        def f(y):                                   # y = [x, v, phi1, w1, phi2, w2]
            a = self.accel(z1 * np.exp(1j * y[2]), y[3], z2 * np.exp(1j * y[4]), y[5], u)
            return np.array([y[1], a[0], y[3], a[1], y[5], a[2]])

        y0 = np.array([x, v, 0.0, w1, 0.0, w2])
        k1 = f(y0)
        k2 = f(y0 + 0.5 * h * k1)
        k3 = f(y0 + 0.5 * h * k2)
        k4 = f(y0 + h * k3)
        y = y0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.pack(y[0], y[1], z1 * np.exp(1j * y[2]), y[3],
                         z2 * np.exp(1j * y[4]), y[5])

    
    def linearize(self, s_nom, u_nom):
        x, v, z1, w1, z2, w2 = self.unpack(s_nom)
        a_nom = self.accel(z1, w1, z2, w2, u_nom)

        def g(e, du):
            a = self.accel(z1 * np.exp(1j * e[1]), w1 + e[4],
                           z2 * np.exp(1j * e[2]), w2 + e[5], u_nom + du)
            return np.array([e[3], e[4], e[5], *(a - a_nom)])

        h, n = 1e-6, 6
        A = np.zeros((n, n))
        for i in range(n):
            d = np.zeros(n)
            d[i] = h
            A[:, i] = (g(d, 0) - g(-d, 0)) / (2 * h)
        B = ((g(np.zeros(n), h) - g(np.zeros(n), -h)) / (2 * h)).reshape(n, 1)
        return A, B

    def design_lqr(self):
        top = self.pack(0, 0, 1 + 0j, 0, 1 + 0j, 0)
        A, B = self.linearize(top, 0.0)
        P = solve_continuous_are(A, B, self.Q, self.R)
        return np.linalg.solve(self.R, B.T @ P).flatten(), P

    @staticmethod
    def error(s, s_ref):
        x, v, z1, w1, z2, w2 = ZDoubleCartPole.unpack(s)
        xr, vr, z1r, w1r, z2r, w2r = ZDoubleCartPole.unpack(s_ref)
        return np.array([x - xr, np.angle(z1 * np.conj(z1r)), np.angle(z2 * np.conj(z2r)),
                         v - vr, w1 - w1r, w2 - w2r])

    def lqr_control(self, s):
        top = self.pack(0, 0, 1 + 0j, 0, 1 + 0j, 0)
        return float(np.clip(-self.K @ self.error(s, top), -self.u_max, self.u_max))


# ======================================================================

# ======================================================================
def _unit_nodes(X):
    """Legacy reference for  unit nodes."""
    X = np.array(X, dtype=float)
    for k in (2, 5):
        n = np.hypot(X[k], X[k + 1])
        X[k] /= n
        X[k + 1] /= n
    return X


def optimize_swingup(model, T=3.0, N=100, x_max=0.5, th0=(np.pi + 0.05, np.pi + 0.02)):
    """Legacy reference for optimize swingup."""
    import casadi as ca
    M, m1, m2, l1, l2, g = model.M, model.m1, model.m2, model.l1, model.l2, model.g

    def f(X, u):
        x, v, c1, s1, w1, c2, s2, w2 = [X[i] for i in range(8)]
        rc, rs = c1 * c2 + s1 * s2, s1 * c2 - c1 * s2     # Re/Im (z1 conj z2)
        Mm = ca.vertcat(
            ca.horzcat(M + m1 + m2, (m1 + m2) * l1 * c1, m2 * l2 * c2),
            ca.horzcat((m1 + m2) * l1 * c1, (m1 + m2) * l1 ** 2, m2 * l1 * l2 * rc),
            ca.horzcat(m2 * l2 * c2, m2 * l1 * l2 * rc, m2 * l2 ** 2))
        F = ca.vertcat(u + (m1 + m2) * l1 * w1 ** 2 * s1 + m2 * l2 * w2 ** 2 * s2,
                       -m2 * l1 * l2 * w2 ** 2 * rs + (m1 + m2) * g * l1 * s1,
                       m2 * l1 * l2 * w1 ** 2 * rs + m2 * g * l2 * s2)
        a = ca.solve(Mm, F)
        return ca.vertcat(v, a[0], -w1 * s1, w1 * c1, a[1], -w2 * s2, w2 * c2, a[2])

    opti = ca.Opti()
    X, U = opti.variable(8, N + 1), opti.variable(1, N + 1)
    h = T / N
    for k in range(N):
        fk, fk1 = f(X[:, k], U[:, k]), f(X[:, k + 1], U[:, k + 1])
        Xm = 0.5 * (X[:, k] + X[:, k + 1]) + h / 8 * (fk - fk1)
        opti.subject_to(X[:, k + 1] - X[:, k]
                        == h / 6 * (fk + 4 * f(Xm, 0.5 * (U[:, k] + U[:, k + 1])) + fk1))
    opti.subject_to(X[:, 0] == [0, 0, np.cos(th0[0]), np.sin(th0[0]), 0,
                                np.cos(th0[1]), np.sin(th0[1]), 0])
    opti.subject_to(X[:, N] == [0, 0, 1, 0, 0, 1, 0, 0])
    opti.subject_to(opti.bounded(-x_max, X[0, :], x_max))
    opti.subject_to(opti.bounded(-model.u_max, U, model.u_max))
    opti.subject_to(U[:, N] == 0)                           
    opti.minimize(h * ca.sumsqr(U) / model.u_max ** 2 + 1e-3 * h * ca.sumsqr(X[1, :]))

    tt = np.linspace(0, 1, N + 1)
    for i, k in enumerate((2, 5)):
        
        
        
        ang = th0[i] + (np.pi - 0.05) * tt
        opti.set_initial(X[k, :], np.cos(ang))
        opti.set_initial(X[k + 1, :], np.sin(ang))
    opti.set_initial(X[4, :], np.pi / T)
    opti.set_initial(X[7, :], np.pi / T)
    opti.set_initial(X[0, :], 0.3 * np.sin(2 * np.pi * tt))
    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0,
                          'ipopt.max_iter': 1500, 'ipopt.tol': 1e-6})
    try:
        sol = opti.solve()
        return (True, np.linspace(0, T, N + 1), _unit_nodes(sol.value(X)), sol.value(U),
                float(sol.value(opti.f)))
    except RuntimeError:
        return (False, np.linspace(0, T, N + 1), _unit_nodes(opti.debug.value(X)),
                opti.debug.value(U), None)


# ======================================================================

# ======================================================================
class TVLQRTracker:
    def __init__(self, model, tn, X, U):
        self.m, self.tn, self.U = model, tn, U
        self.T = tn[-1]
        self.Xs = CubicSpline(tn, X, axis=1)
        N = len(tn) - 1
        AB = [model.linearize(X[:, k], U[k]) for k in range(N + 1)]
        Ri = np.linalg.inv(model.R)
        Q = model.Q
        dP = lambda P, A, B: -(A.T @ P + P @ A - P @ B @ Ri @ B.T @ P + Q)
        P = [None] * (N + 1)
        P[N] = model.P
        sub = 10
        for k in range(N, 0, -1):
            Pk, h = P[k], -(tn[k] - tn[k - 1]) / sub
            Af = lambda a: (1 - a) * AB[k][0] + a * AB[k - 1][0]
            Bf = lambda a: (1 - a) * AB[k][1] + a * AB[k - 1][1]
            for j in range(sub):
                a0, a1, a2 = j / sub, (j + .5) / sub, (j + 1) / sub
                k1 = dP(Pk, Af(a0), Bf(a0))
                k2 = dP(Pk + h / 2 * k1, Af(a1), Bf(a1))
                k3 = dP(Pk + h / 2 * k2, Af(a1), Bf(a1))
                k4 = dP(Pk + h * k3, Af(a2), Bf(a2))
                Pk = Pk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            P[k - 1] = Pk
        self.Ks = np.array([(Ri @ AB[k][1].T @ P[k]).flatten() for k in range(N + 1)])

    def control(self, t, s):
        if t >= self.T:
            return self.m.lqr_control(s)
        K = np.array([np.interp(t, self.tn, self.Ks[:, i]) for i in range(6)])
        e = self.m.error(s, self.Xs(t))
        u = np.interp(t, self.tn, self.U) - K @ e
        return float(np.clip(u, -self.m.u_max, self.m.u_max))


def simulate(model, controller, th0=(np.pi + 0.05, np.pi + 0.02), x0=0.0, t_max=8.0,
             dt=1e-3, w0=(0.0, 0.0)):
    """Legacy reference for simulate."""
    n = int(round(t_max / dt))
    s = model.pack(x0, 0.0, np.exp(1j * th0[0]), w0[0], np.exp(1j * th0[1]), w0[1])
    Y, Uc = np.empty((8, n + 1)), np.empty(n + 1)
    for k in range(n + 1):
        u = float(np.clip(controller(k * dt, s), -model.u_max, model.u_max))
        Y[:, k], Uc[k] = s, u
        if k < n:
            s = model.step(s, u, dt)
    return SimpleNamespace(t=np.arange(n + 1) * dt, y=Y, u=Uc)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = ZDoubleCartPole()
    ok, tn, X, U, J = optimize_swingup(model)
    print("path found:", ok, "| max|x| =", round(np.abs(X[0]).max(), 3),
          "| max|u| =", round(np.abs(U).max(), 1), "N")
    tracker = TVLQRTracker(model, tn, X, U)
    sol = simulate(model, tracker.control)                  
    z1, z2 = sol.y[2] + 1j * sol.y[3], sol.y[5] + 1j * sol.y[6]
    print(f"max|x| = {np.abs(sol.y[0]).max():.3f} m | final z1 = {z1[-1]:.4f}, z2 = {z2[-1]:.4f}")

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(sol.t, sol.y[0]); ax[0].axhline(.5, c='r', ls='--'); ax[0].axhline(-.5, c='r', ls='--')
    ax[0].set_ylabel("x [m]")
    ax[1].plot(sol.t, np.unwrap(np.angle(z1)), label="θ1 = arg z1")
    ax[1].plot(sol.t, np.unwrap(np.angle(z2)), label="θ2 = arg z2")
    ax[1].set_ylabel("angle [rad]"); ax[1].legend()
    ax[2].plot(sol.t, sol.u); ax[2].set_ylabel("u [N]"); ax[2].set_xlabel("t [s]")
    ax[0].set_title("Swing-up optimization, TVLQR tracking, and upright LQR")
    for a in ax: a.axvline(tn[-1], c='gray', ls=':'); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("sonuc.png", dpi=130)

    from zanimate import save_gif
    save_gif(model, sol, "swingup.gif", t_end=tn[-1] + 2.0, t_switch=tn[-1])
    print("Wrote: sonuc.png, swingup.gif")
