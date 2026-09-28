"""Check mechanical dynamics, integration, and local LQR behavior."""
import os
import sys

import numpy as np
import pytest

from zcartpole.core import ZNCartPole, TVLQRTracker, simulate                    # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "legacy"))
from legacy_double import ZDoubleCartPole                                        # noqa: E402
import legacy_double as _legacy2                                                 # noqa: E402
from legacy_triple import ZTripleCartPole                                        # noqa: E402
import legacy_triple as _legacy3                                                 # noqa: E402


# ======================================================================

# ======================================================================
def make_model(n):
    m = 0.1 * np.ones(n)
    l = 0.4 * np.ones(n)
    u_max = {2: 50.0, 3: 80.0}.get(n, 60.0 * n)   
    return ZNCartPole(m, l, u_max=u_max)


def free_run(model, th0, w0, t=2.0, dt=1e-3):
    return simulate(model, lambda t, s: 0.0, th0=th0, w0=w0, t_max=t, dt=dt)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_energy_conserved_without_input(n):
    m = make_model(n)
    rng = np.random.default_rng(n)
    th0, w0 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    sol = free_run(m, th0, w0)
    E = np.array([m.total_energy(sol.y[:, i]) for i in range(sol.y.shape[1])])
    assert np.ptp(E) < 1e-5


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_horizontal_momentum_conserved_without_input(n):
    m = make_model(n)
    rng = np.random.default_rng(n)
    th0, w0 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    sol = free_run(m, th0, w0)
    p = np.array([m.momentum(sol.y[:, i]) for i in range(sol.y.shape[1])])
    assert np.ptp(p) < 1e-5


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_unit_modulus_is_structural(n):
    """Check test unit modulus is structural."""
    m = make_model(n)
    rng = np.random.default_rng(n)
    th0, w0 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    sol = free_run(m, th0, w0, t=5.0)
    for i in range(n):
        mag = np.hypot(sol.y[2 + 3 * i], sol.y[3 + 3 * i])
        assert np.max(np.abs(mag - 1)) < 1e-12


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_integrator_is_fourth_order(n):
    """Check test integrator is fourth order."""
    m = make_model(n)
    rng = np.random.default_rng(n)
    th0, w0 = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)

    def drift(dt):
        sol = free_run(m, th0, w0, t=1.0, dt=dt)
        E = np.array([m.total_energy(sol.y[:, i]) for i in range(sol.y.shape[1])])
        return np.ptp(E)

    assert drift(4e-3) / drift(2e-3) > 10


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_lqr_closed_loop_is_stable(n):
    m = make_model(n)
    A, B = m.linearize(m.top_state(), 0.0)
    dim = 2 + 2 * n
    assert np.all(np.linalg.eigvals(A - B @ m.K.reshape(1, dim)).real < 0)


@pytest.mark.parametrize("n", [2, 3])
def test_lqr_catches_small_perturbation(n):
    """Check test lqr catches small perturbation."""
    m = make_model(n)
    th0 = {2: (0.05, -0.03), 3: (0.05, -0.03, 0.02)}[n]      
    sol = simulate(m, lambda t, s: m.lqr_control(s), th0=th0, t_max=6.0)
    for i in range(n):
        z_final = sol.y[2 + 3 * i, -1] + 1j * sol.y[3 + 3 * i, -1]
        assert abs(np.angle(z_final)) < 1e-2


# ======================================================================
# Regression against the older two- and three-link models.
# ======================================================================
@pytest.fixture
def pair2():
    gen = ZNCartPole(m=(0.1, 0.1), l=(0.5, 0.5), M=1.0, g=9.81, u_max=50.0)
    old = ZDoubleCartPole()
    return gen, old


@pytest.fixture
def pair3():
    gen = ZNCartPole(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4), M=1.0, g=9.81, u_max=80.0)
    old = ZTripleCartPole()
    return gen, old


def _random_state2(rng):
    th = rng.uniform(-np.pi, np.pi, 2)
    w = rng.uniform(-2, 2, 2)
    return np.array([rng.uniform(-0.3, 0.3), rng.uniform(-1, 1),
                     np.cos(th[0]), np.sin(th[0]), w[0],
                     np.cos(th[1]), np.sin(th[1]), w[1]])


def _random_state3(rng):
    th = rng.uniform(-np.pi, np.pi, 3)
    w = rng.uniform(-2, 2, 3)
    return np.array([rng.uniform(-0.3, 0.3), rng.uniform(-1, 1),
                     np.cos(th[0]), np.sin(th[0]), w[0],
                     np.cos(th[1]), np.sin(th[1]), w[1],
                     np.cos(th[2]), np.sin(th[2]), w[2]])


def test_mass_matrix_matches_double(pair2):
    gen, old = pair2
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = _random_state2(rng)
        _, _, zs, _ = gen.unpack(s)
        z1, z2 = zs
        np.testing.assert_allclose(gen.mass_matrix(zs), old.mass_matrix(z1, z2), atol=1e-12)


def test_mass_matrix_matches_triple(pair3):
    gen, old = pair3
    rng = np.random.default_rng(2)
    for _ in range(20):
        s = _random_state3(rng)
        _, _, zs, _ = gen.unpack(s)
        z1, z2, z3 = zs
        np.testing.assert_allclose(gen.mass_matrix(zs), old.mass_matrix(z1, z2, z3), atol=1e-12)


def test_rhs_forces_and_accel_match_double(pair2):
    gen, old = pair2
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = _random_state2(rng)
        _, _, zs, ws = gen.unpack(s)
        z1, z2 = zs
        w1, w2 = ws
        u = rng.uniform(-30, 30)
        np.testing.assert_allclose(gen.rhs_forces(zs, ws, u),
                                   old.rhs_forces(z1, w1, z2, w2, u), atol=1e-10)
        np.testing.assert_allclose(gen.accel(zs, ws, u),
                                   old.accel(z1, w1, z2, w2, u), atol=1e-10)


def test_rhs_forces_and_accel_match_triple(pair3):
    gen, old = pair3
    rng = np.random.default_rng(4)
    for _ in range(20):
        s = _random_state3(rng)
        _, _, zs, ws = gen.unpack(s)
        z1, z2, z3 = zs
        w1, w2, w3 = ws
        u = rng.uniform(-30, 30)
        np.testing.assert_allclose(gen.rhs_forces(zs, ws, u),
                                   old.rhs_forces(z1, w1, z2, w2, z3, w3, u), atol=1e-10)
        np.testing.assert_allclose(gen.accel(zs, ws, u),
                                   old.accel(z1, w1, z2, w2, z3, w3, u), atol=1e-10)


def test_energy_and_momentum_match_double(pair2):
    gen, old = pair2
    rng = np.random.default_rng(5)
    for _ in range(20):
        s = _random_state2(rng)
        assert gen.total_energy(s) == pytest.approx(old.total_energy(s), abs=1e-10)
        np.testing.assert_allclose(gen.momentum(s), old.momentum(s), atol=1e-10)


def test_energy_and_momentum_match_triple(pair3):
    gen, old = pair3
    rng = np.random.default_rng(6)
    for _ in range(20):
        s = _random_state3(rng)
        assert gen.total_energy(s) == pytest.approx(old.total_energy(s), abs=1e-10)
        np.testing.assert_allclose(gen.momentum(s), old.momentum(s), atol=1e-10)


def test_single_rk4_step_matches_double(pair2):
    gen, old = pair2
    rng = np.random.default_rng(7)
    for _ in range(20):
        s = _random_state2(rng)
        u, h = rng.uniform(-30, 30), 1e-3
        np.testing.assert_allclose(gen.step(s, u, h), old.step(s, u, h), atol=1e-10)


def test_single_rk4_step_matches_triple(pair3):
    gen, old = pair3
    rng = np.random.default_rng(8)
    for _ in range(20):
        s = _random_state3(rng)
        u, h = rng.uniform(-30, 30), 1e-3
        np.testing.assert_allclose(gen.step(s, u, h), old.step(s, u, h), atol=1e-10)


def test_linearize_matches_double(pair2):
    gen, old = pair2
    rng = np.random.default_rng(9)
    for _ in range(10):
        s = _random_state2(rng)
        u = rng.uniform(-30, 30)
        A1, B1 = gen.linearize(s, u)
        A2, B2 = old.linearize(s, u)
        np.testing.assert_allclose(A1, A2, atol=1e-6)
        np.testing.assert_allclose(B1, B2, atol=1e-6)


def test_linearize_matches_triple(pair3):
    gen, old = pair3
    rng = np.random.default_rng(10)
    for _ in range(10):
        s = _random_state3(rng)
        u = rng.uniform(-30, 30)
        A1, B1 = gen.linearize(s, u)
        A2, B2 = old.linearize(s, u)
        np.testing.assert_allclose(A1, A2, atol=1e-6)
        np.testing.assert_allclose(B1, B2, atol=1e-6)


def test_lqr_gain_matches_double(pair2):
    gen, old = pair2
    np.testing.assert_allclose(gen.K, old.K, atol=1e-6)


def test_lqr_gain_matches_triple(pair3):
    gen, old = pair3
    np.testing.assert_allclose(gen.K, old.K, atol=1e-6)


def test_lqr_closed_loop_trajectory_matches_double(pair2):
    gen, old = pair2
    sol_gen = simulate(gen, lambda t, s: gen.lqr_control(s), th0=(0.05, -0.03), t_max=3.0)
    sol_old = _legacy2.simulate(old, lambda t, s: old.lqr_control(s), th0=(0.05, -0.03), t_max=3.0)
    np.testing.assert_allclose(sol_gen.y, sol_old.y, atol=1e-6)
    np.testing.assert_allclose(sol_gen.u, sol_old.u, atol=1e-6)


def test_lqr_closed_loop_trajectory_matches_triple(pair3):
    gen, old = pair3
    th0 = (0.05, -0.03, 0.02)
    sol_gen = simulate(gen, lambda t, s: gen.lqr_control(s), th0=th0, t_max=3.0)
    sol_old = _legacy3.simulate(old, lambda t, s: old.lqr_control(s), th0=th0, t_max=3.0)
    np.testing.assert_allclose(sol_gen.y, sol_old.y, atol=1e-6)
    np.testing.assert_allclose(sol_gen.u, sol_old.u, atol=1e-6)
