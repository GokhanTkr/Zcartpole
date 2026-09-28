"""Mechanical model for a cart and any number of serial pendulum links.

Link angles use unit complex states. The stepper updates them on the circle."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_continuous_are


class ZNCartPole:
    def __init__(self, m, l, M=1.0, g=9.81, u_max=50.0, Q=None, R=None,
                 rod_mass=None, rod_com=0.5, rod_inertia=None,
                 cart_damping=0.0, joint_damping=0.0,
                 cart_coulomb=0.0, coulomb_velocity=0.01, backend='auto'):
        """Serial planar cart-pole with endpoint masses and optional rigid rods.

        ``rod_com`` is a fraction of link length from its proximal hinge.
        ``rod_inertia`` is about each rod's center of mass; by default a
        uniform rod has I = rod_mass*l**2/12. Joint damping acts on relative
        angular velocity, while cart damping acts on horizontal velocity.
        Zero rod mass and damping recover the original point-mass model.
        Cart Coulomb friction is smoothed as F*tanh(v/v_s); it does not model
        static sticking or breakaway.
        ``backend`` is 'auto', 'python', or 'numba'. Auto uses Numba if installed.
        """
        m, l = np.asarray(m, dtype=float), np.asarray(l, dtype=float)
        if m.shape != l.shape or m.ndim != 1 or len(m) < 1:
            raise ValueError("m and l must be 1D arrays of equal positive length")
        self.n = len(m)

        def per_link(value, name):
            result = np.asarray(value, dtype=float)
            if result.ndim == 0:
                result = np.full(self.n, float(result))
            if result.shape != (self.n,) or not np.isfinite(result).all():
                raise ValueError(f'{name} must be a finite scalar or an n-vector')
            return result

        rod_mass = per_link(0.0 if rod_mass is None else rod_mass, 'rod_mass')
        rod_com = per_link(rod_com, 'rod_com')
        rod_inertia = per_link(rod_mass*l*l/12 if rod_inertia is None
                               else rod_inertia, 'rod_inertia')
        joint_damping = per_link(joint_damping, 'joint_damping')
        if (not np.isfinite(m).all() or not np.isfinite(l).all()
            or np.any(m < 0) or np.any(l <= 0) or np.any(m+rod_mass <= 0)
            or np.any(rod_mass < 0) or np.any(rod_inertia < 0)
            or np.any((rod_com < 0) | (rod_com > 1))
            or np.any(joint_damping < 0) or not np.isfinite(cart_damping)
            or cart_damping < 0 or not np.isfinite(cart_coulomb)
            or cart_coulomb < 0 or not np.isfinite(coulomb_velocity)
            or coulomb_velocity <= 0 or not np.isfinite(M) or M <= 0
            or not np.isfinite(g) or g <= 0 or not np.isfinite(u_max)
            or u_max <= 0
            or np.any(m + rod_mass*rod_com**2 + rod_inertia/l**2 <= 0)):
            raise ValueError('Masses, lengths, inertia and damping must be physical')
        self.M, self.m, self.l, self.g, self.u_max = M, m, l, g, u_max
        self.rod_mass, self.rod_com = rod_mass, rod_com
        self.rod_inertia = rod_inertia
        self.cart_damping, self.joint_damping = float(cart_damping), joint_damping
        self.cart_coulomb = float(cart_coulomb)
        self.coulomb_velocity = float(coulomb_velocity)
        self.S = np.cumsum(m[::-1])[::-1]          # old endpoint mass convention
        self.S_total = np.cumsum((m+rod_mass)[::-1])[::-1]
        downstream = np.r_[self.S_total[1:], 0.0]
        self.A = downstream + m + rod_mass*rod_com
        self.D = downstream + m + rod_mass*rod_com**2
        self.E_ref = float(self.g * np.sum(self.A*l))
        if backend not in ('auto', 'python', 'numba'):
            raise ValueError("backend must be 'auto', 'python', or 'numba'")
        self._fast_core = None
        if backend != 'python':
            try:
                from . import _fast_core
            except ImportError as exc:
                if backend == 'numba':
                    raise ImportError(
                        "Numba backend requires pip install 'zcartpole[speed]'"
                    ) from exc
            else:
                self._fast_core = _fast_core
        self.backend = 'numba' if self._fast_core else 'python'
        self._fast_scalars = np.array([
            self.M, self.S_total[0], self.g, self.cart_damping,
            self.cart_coulomb, self.coulomb_velocity], dtype=float)
        if Q is None:
            Q = np.diag([10.0] + [100.0 * self.n] * self.n + [1.0] + [10.0] * self.n)
        self.Q = Q
        self.R = np.array([[0.1]]) if R is None else np.asarray(R)
        self.K, self.P = self.design_lqr()

    # Convert between packed state and complex link states.
    def unpack(self, s):
        n = self.n
        rest = s[2:]
        zs = rest[0:3 * n:3] + 1j * rest[1:3 * n:3]
        ws = rest[2:3 * n:3]
        return s[0], s[1], np.asarray(zs, dtype=complex), np.asarray(ws, dtype=float)

    def pack(self, x, v, zs, ws):
        zs, ws = np.asarray(zs, dtype=complex), np.asarray(ws, dtype=float)
        out = np.empty(2 + 3 * self.n)
        out[0], out[1] = x, v
        out[2:2 + 3 * self.n:3] = zs.real
        out[3:2 + 3 * self.n:3] = zs.imag
        out[4:2 + 3 * self.n:3] = ws
        return out

    # Mechanical mass matrix and forces.
    def mass_matrix(self, zs):
        n, A, D, l = self.n, self.A, self.D, self.l
        Mm = np.zeros((n + 1, n + 1))
        Mm[0, 0] = self.M + self.S_total[0]
        for i in range(1, n + 1):
            Mm[0, i] = Mm[i, 0] = A[i - 1] * l[i - 1] * zs[i - 1].real
            Mm[i, i] = D[i - 1] * l[i - 1] ** 2 + self.rod_inertia[i - 1]
            for j in range(i + 1, n + 1):
                r = zs[i - 1] * np.conj(zs[j - 1])
                Mm[i, j] = Mm[j, i] = A[j - 1] * l[i - 1] * l[j - 1] * r.real
        return Mm

    def rhs_forces(self, zs, ws, u, v=0.0):
        n, A, l, g = self.n, self.A, self.l, self.g
        F = np.zeros(n + 1)
        F[0] = (u - self.cart_damping*v
                - self.cart_coulomb*np.tanh(v/self.coulomb_velocity)
                + float(np.sum(A * l * ws ** 2 * zs.imag)))
        for i in range(1, n + 1):
            Fi = A[i - 1] * g * l[i - 1] * zs[i - 1].imag
            for j in range(i + 1, n + 1):
                r = zs[i - 1] * np.conj(zs[j - 1])
                Fi += -A[j - 1] * l[i - 1] * l[j - 1] * ws[j - 1] ** 2 * r.imag
            for j in range(1, i):
                r = zs[j - 1] * np.conj(zs[i - 1])
                Fi += A[i - 1] * l[j - 1] * l[i - 1] * ws[j - 1] ** 2 * r.imag
            Fi -= self.joint_damping[i-1]*(ws[i-1] - (ws[i-2] if i > 1 else 0))
            if i < n:
                Fi += self.joint_damping[i]*(ws[i] - ws[i-1])
            F[i] = Fi
        return F

    def accel(self, zs, ws, u, v=0.0):
        """[x_ddot, w1_dot, ..., wn_dot]"""
        if self._fast_core is not None:
            return self._fast_core.accel(
                np.asarray(zs, dtype=np.complex128),
                np.asarray(ws, dtype=float), float(u), float(v),
                self._fast_scalars, self.A, self.D, self.l,
                self.rod_inertia, self.joint_damping)
        return np.linalg.solve(self.mass_matrix(zs), self.rhs_forces(zs, ws, u, v))

    def total_energy(self, s):
        x, v, zs, ws = self.unpack(s)
        q = np.concatenate(([v], ws))
        T = 0.5 * q @ self.mass_matrix(zs) @ q
        V = float(self.g * np.sum(self.A * self.l * zs.real))
        return T + V

    def momentum(self, s):
        """Return total horizontal momentum."""
        x, v, zs, ws = self.unpack(s)
        return self.mass_matrix(zs)[0] @ np.concatenate(([v], ws))

    def step(self, s, u, h):
        """Advance one zero-order-held force step with exponential-map RK4.

        Each stage rotates the initial unit complex link states by an angle increment."""
        if self._fast_core is not None:
            return self._fast_core.step(
                np.asarray(s, dtype=float), float(u), float(h),
                self._fast_scalars, self.A, self.D, self.l,
                self.rod_inertia, self.joint_damping)
        n = self.n
        x, v, zs, ws = self.unpack(s)

        def f(y):                                    # y = [x, v, phi1, w1, ..., phin, wn]
            phis, wcur = y[2:2 + 2 * n:2], y[3:2 + 2 * n:2]
            a = self.accel(zs * np.exp(1j * phis), wcur, u, y[1])
            dy = np.empty(2 + 2 * n)
            dy[0], dy[1] = y[1], a[0]
            dy[2:2 + 2 * n:2] = wcur
            dy[3:2 + 2 * n:2] = a[1:]
            return dy

        y0 = np.zeros(2 + 2 * n)
        y0[0], y0[1] = x, v
        y0[3:2 + 2 * n:2] = ws
        k1 = f(y0)
        k2 = f(y0 + 0.5 * h * k1)
        k3 = f(y0 + 0.5 * h * k2)
        k4 = f(y0 + h * k3)
        y = y0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.pack(y[0], y[1], zs * np.exp(1j * y[2:2 + 2 * n:2]), y[3:2 + 2 * n:2])

    
    def linearize(self, s_nom, u_nom):
        n = self.n
        x, v, zs, ws = self.unpack(s_nom)
        a_nom = self.accel(zs, ws, u_nom, v)

        def g(e, du):
            phis_e, v_e, ws_e = e[1:1 + n], e[1 + n], e[2 + n:2 + 2 * n]
            a = self.accel(zs * np.exp(1j * phis_e), ws + ws_e,
                           u_nom + du, v + v_e)
            out = np.empty(2 + 2 * n)
            out[0] = v_e
            out[1:1 + n] = ws_e
            out[1 + n:2 + 2 * n] = a - a_nom
            return out

        h, dim = 1e-6, 2 + 2 * n
        A = np.zeros((dim, dim))
        for i in range(dim):
            d = np.zeros(dim)
            d[i] = h
            A[:, i] = (g(d, 0) - g(-d, 0)) / (2 * h)
        B = ((g(np.zeros(dim), h) - g(np.zeros(dim), -h)) / (2 * h)).reshape(dim, 1)
        return A, B

    def top_state(self):
        return self.pack(0, 0, np.ones(self.n), np.zeros(self.n))

    def design_lqr(self):
        A, B = self.linearize(self.top_state(), 0.0)
        P = solve_continuous_are(A, B, self.Q, self.R)
        return np.linalg.solve(self.R, B.T @ P).flatten(), P

    def error(self, s, s_ref):
        x, v, zs, ws = self.unpack(s)
        xr, vr, zsr, wsr = self.unpack(s_ref)
        return np.concatenate(([x - xr], np.angle(zs * np.conj(zsr)), [v - vr], ws - wsr))

    def lqr_control(self, s):
        return float(np.clip(-self.K @ self.error(s, self.top_state()), -self.u_max, self.u_max))

# Time-varying LQR around a nominal path.
class TVLQRTracker:
    def __init__(self, model, tn, X, U):
        self.m, self.tn, self.U = model, tn, U
        self.T = tn[-1]
        self.Xs = CubicSpline(tn, X, axis=1)
        N, dim = len(tn) - 1, 2 + 2 * model.n
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
        self._dim = dim

    def control(self, t, s):
        if t >= self.T:
            return self.m.lqr_control(s)
        K = np.array([np.interp(t, self.tn, self.Ks[:, i]) for i in range(self._dim)])
        e = self.m.error(s, self.Xs(t))
        u = np.interp(t, self.tn, self.U) - K @ e
        return float(np.clip(u, -self.m.u_max, self.m.u_max))


def simulate(model, controller, th0, x0=0.0, t_max=8.0, dt=1e-3,
             w0=None, v0=0.0):
    """Simulate a controller with fixed-step, zero-order-held force.
    
    Return time, mechanical states, and applied forces."""
    n = model.n
    w0 = np.zeros(n) if w0 is None else np.asarray(w0, dtype=float)
    th0 = np.asarray(th0, dtype=float)
    steps = int(round(t_max / dt))
    s = model.pack(x0, v0, np.exp(1j * th0), w0)
    Y, Uc = np.empty((2 + 3 * n, steps + 1)), np.empty(steps + 1)
    for k in range(steps + 1):
        u = float(np.clip(controller(k * dt, s), -model.u_max, model.u_max))
        Y[:, k], Uc[k] = s, u
        if k < steps:
            s = model.step(s, u, dt)
    return SimpleNamespace(t=np.arange(steps + 1) * dt, y=Y, u=Uc)
