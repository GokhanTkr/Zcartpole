"""Optional Numba implementation of the mechanical acceleration and RK4 step.

The equations match core.ZNCartPole. Keep the pure Python path as a reference.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def accel(zs, ws, u, v, scalars, A, D, lengths, inertia, damping):
    """Solve M(q) q_ddot = rhs(q, q_dot, u)."""
    n = len(lengths)
    cart_mass, total_link_mass, gravity, viscous, coulomb, smooth = scalars
    matrix = np.zeros((n + 1, n + 1))
    force = np.zeros(n + 1)
    matrix[0, 0] = cart_mass + total_link_mass
    force[0] = u - viscous * v - coulomb * np.tanh(v / smooth)

    for i in range(n):
        zi = zs[i]
        li = lengths[i]
        matrix[0, i + 1] = A[i] * li * zi.real
        matrix[i + 1, 0] = matrix[0, i + 1]
        matrix[i + 1, i + 1] = D[i] * li * li + inertia[i]
        force[0] += A[i] * li * ws[i] * ws[i] * zi.imag
        fi = A[i] * gravity * li * zi.imag
        for j in range(i + 1, n):
            relative = zi * np.conj(zs[j])
            matrix[i + 1, j + 1] = A[j] * li * lengths[j] * relative.real
            matrix[j + 1, i + 1] = matrix[i + 1, j + 1]
            fi -= A[j] * li * lengths[j] * ws[j] * ws[j] * relative.imag
        for j in range(i):
            relative = zs[j] * np.conj(zi)
            fi += A[i] * lengths[j] * li * ws[j] * ws[j] * relative.imag
        fi -= damping[i] * (ws[i] - (ws[i - 1] if i else 0.0))
        if i + 1 < n:
            fi += damping[i + 1] * (ws[i + 1] - ws[i])
        force[i + 1] = fi
    return np.linalg.solve(matrix, force)


@njit(cache=True)
def _stage(y, initial_z, u, scalars, A, D, lengths, inertia, damping):
    n = len(lengths)
    z = np.empty(n, dtype=np.complex128)
    w = np.empty(n)
    for i in range(n):
        phi = y[2 + 2 * i]
        z[i] = initial_z[i] * (np.cos(phi) + 1j * np.sin(phi))
        w[i] = y[3 + 2 * i]
    a = accel(z, w, u, y[1], scalars, A, D, lengths, inertia, damping)
    dy = np.empty(2 + 2 * n)
    dy[0], dy[1] = y[1], a[0]
    for i in range(n):
        dy[2 + 2 * i] = w[i]
        dy[3 + 2 * i] = a[i + 1]
    return dy


@njit(cache=True)
def step(state, u, h, scalars, A, D, lengths, inertia, damping):
    """Advance one constant-force step while keeping every link on the circle."""
    n = len(lengths)
    initial_z = np.empty(n, dtype=np.complex128)
    y0 = np.zeros(2 + 2 * n)
    y0[0], y0[1] = state[0], state[1]
    for i in range(n):
        initial_z[i] = state[2 + 3 * i] + 1j * state[3 + 3 * i]
        y0[3 + 2 * i] = state[4 + 3 * i]
    k1 = _stage(y0, initial_z, u, scalars, A, D, lengths, inertia, damping)
    k2 = _stage(y0 + 0.5 * h * k1, initial_z, u,
                scalars, A, D, lengths, inertia, damping)
    k3 = _stage(y0 + 0.5 * h * k2, initial_z, u,
                scalars, A, D, lengths, inertia, damping)
    k4 = _stage(y0 + h * k3, initial_z, u,
                scalars, A, D, lengths, inertia, damping)
    y = y0 + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result = np.empty(2 + 3 * n)
    result[0], result[1] = y[0], y[1]
    for i in range(n):
        phi = y[2 + 2 * i]
        z = initial_z[i] * (np.cos(phi) + 1j * np.sin(phi))
        result[2 + 3 * i] = z.real
        result[3 + 3 * i] = z.imag
        result[4 + 3 * i] = y[3 + 2 * i]
    return result
