"""Check swing-up constraints and legacy numerical regression."""
import os
import sys

import numpy as np
import pytest

ca = pytest.importorskip("casadi", reason="CasADi is not installed (pip install zcartpole[opt])")

from zcartpole.core import ZNCartPole, TVLQRTracker, simulate                    # noqa: E402
from zcartpole.opt import build_warm_start, optimize_swingup                     # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "legacy"))
from legacy_double import ZDoubleCartPole                                        # noqa: E402
import legacy_double as _legacy2                                                 # noqa: E402
from legacy_triple import ZTripleCartPole                                        # noqa: E402
import legacy_triple as _legacy3                                                 # noqa: E402


# Selected legacy optimizer checks.
@pytest.fixture(scope="module")
def traj2():
    m = ZNCartPole(m=(0.1, 0.1), l=(0.5, 0.5), u_max=50.0)
    ok, tn, X, U, _ = optimize_swingup(m, th0=(np.pi + 0.05, np.pi + 0.02))
    assert ok
    return m, tn, X, U


@pytest.fixture(scope="module")
def traj3():
    m = ZNCartPole(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4), u_max=80.0)
    ok, tn, X, U, _ = optimize_swingup(m, th0=(np.pi + 0.05, np.pi + 0.03, np.pi + 0.02))
    assert ok
    return m, tn, X, U


@pytest.mark.parametrize("traj_fixture", [
    "traj2", pytest.param("traj3", marks=pytest.mark.legacy_triple),
])
def test_trajectory_respects_limits(traj_fixture, request):
    m, tn, X, U = request.getfixturevalue(traj_fixture)
    assert np.abs(X[0]).max() <= 0.5 + 1e-6
    assert np.abs(U).max() <= m.u_max + 1e-6


@pytest.mark.parametrize("traj_fixture", [
    "traj2", pytest.param("traj3", marks=pytest.mark.legacy_triple),
])
def test_trajectory_ends_at_equilibrium_with_zero_input(traj_fixture, request):
    m, tn, X, U = request.getfixturevalue(traj_fixture)
    assert abs(U[-1]) < 1e-6
    for i in range(m.n):
        k = 2 + 3 * i
        radius = np.hypot(X[k], X[k + 1])
        # Raw Hermite-Simpson nodes have small manifold drift.
        assert np.max(np.abs(radius - 1.0)) < 1e-4
        assert abs(radius[-1] - 1.0) < 1e-6


def test_swingup_and_catch_double(traj2):
    m, tn, X, U = traj2
    tracker = TVLQRTracker(m, tn, X, U)
    sol = simulate(m, tracker.control, th0=(np.pi + 0.05, np.pi + 0.02), t_max=8.0)
    assert np.abs(sol.y[0]).max() < 0.55
    for i in range(2):
        z_final = sol.y[2 + 3 * i, -1] + 1j * sol.y[3 + 3 * i, -1]
        assert abs(np.angle(z_final)) < 1e-2


@pytest.mark.legacy_triple
def test_swingup_and_catch_triple(traj3):
    m, tn, X, U = traj3
    tracker = TVLQRTracker(m, tn, X, U)
    sol = simulate(m, tracker.control, th0=(np.pi + 0.05, np.pi + 0.03, np.pi + 0.02), t_max=9.0)
    assert np.abs(sol.y[0]).max() < 0.65
    for i in range(3):
        z_final = sol.y[2 + 3 * i, -1] + 1j * sol.y[3 + 3 * i, -1]
        assert abs(np.angle(z_final)) < 1e-2


# Both formulations should return feasible paths. Local minima may differ.
def _assert_feasible(X, U, x_max, u_max):
    assert np.abs(X[0]).max() <= x_max + 1e-6
    assert np.abs(U).max() <= u_max + 1e-6


def test_swingup_matches_legacy_double():
    gen = ZNCartPole(m=(0.1, 0.1), l=(0.5, 0.5), u_max=50.0)
    old = ZDoubleCartPole()
    th0 = (np.pi + 0.05, np.pi + 0.02)
    ok_g, tn_g, X_g, U_g, J_g = optimize_swingup(gen, T=3.0, N=100, x_max=0.5, th0=th0)
    ok_o, tn_o, X_o, U_o, J_o = _legacy2.optimize_swingup(old, T=3.0, N=100, x_max=0.5, th0=th0)
    assert ok_g and ok_o
    np.testing.assert_allclose(tn_g, tn_o, atol=1e-12)
    assert np.isfinite(J_g) and J_g >= 0
    assert np.isfinite(J_o) and J_o >= 0
    _assert_feasible(X_g, U_g, 0.5, gen.u_max)
    _assert_feasible(X_o, U_o, 0.5, old.u_max)


# Manual warm-start interface.
def test_build_warm_start_shape_and_endpoint():
    m = ZNCartPole(m=(0.1, 0.1), l=(0.5, 0.5), u_max=50.0)
    th0 = (np.pi + 0.05, np.pi + 0.02)
    X0, U0 = build_warm_start(
        m, T=3.0, N=100, th0=th0,
        
        waypoints=[[(1.0, 2 * np.pi * 2)],
                   [(0.5, th0[1] + 0.3 * np.pi), (1.0, 2 * np.pi * 1)]],
    )
    assert X0.shape == (2 + 3 * m.n, 101)
    assert U0.shape == (101,)
    
    for i in range(m.n):
        z0 = X0[2 + 3 * i, 0] + 1j * X0[3 + 3 * i, 0]
        assert abs(z0 - np.exp(1j * th0[i])) < 1e-9
    
    for i in range(m.n):
        r = np.hypot(X0[2 + 3 * i, :], X0[3 + 3 * i, :])
        np.testing.assert_allclose(r, 1.0, atol=1e-9)
    
    for i in range(m.n):
        assert X0[2 + 3 * i, -1] == pytest.approx(1.0, abs=1e-9)
        assert X0[3 + 3 * i, -1] == pytest.approx(0.0, abs=1e-9)


def test_optimize_swingup_with_custom_warm_start_reports_attempt():
    m = ZNCartPole(m=(0.1, 0.1), l=(0.5, 0.5), u_max=50.0)
    th0 = (np.pi + 0.05, np.pi + 0.02)
    warm = build_warm_start(
        m, T=3.0, N=100, th0=th0,
        waypoints=[[(1.0, 2 * np.pi * 2)] for i in range(2)],
    )
    info = {}
    ok, tn, X, U, J = optimize_swingup(
        m, T=3.0, N=100, x_max=0.5, th0=th0, warm=warm, info=info)
    assert len(tn) == 101 and X.shape == warm[0].shape and U.shape == warm[1].shape
    assert info['calls'][0]['stage'] == 'warm_single'
    assert info['calls'][0]['ok'] == ok
    if ok:
        _assert_feasible(X, U, 0.5, m.u_max)
        assert abs(U[-1]) < 1e-6
        for i in range(2):
            c, s = X[2 + 3 * i, :], X[3 + 3 * i, :]
            th = np.unwrap(np.arctan2(s, c))
            winding = (th[-1] - th[0]) / (2 * np.pi)
            assert abs(winding - 1.49) < 0.15
    else:
        assert J is None


@pytest.mark.legacy_triple
def test_swingup_matches_legacy_triple():
    gen = ZNCartPole(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4), u_max=80.0)
    old = ZTripleCartPole()
    th0 = (np.pi + 0.05, np.pi + 0.03, np.pi + 0.02)
    ok_g, tn_g, X_g, U_g, J_g = optimize_swingup(gen, T=6.0, N=100, x_max=0.5, th0=th0,
                                                 w_penalty=3e-3)
    ok_o, tn_o, X_o, U_o, J_o = _legacy3.optimize_swingup(old, T=6.0, N=100, x_max=0.5, th0=th0,
                                                          w_penalty=3e-3)
    assert ok_g and ok_o
    np.testing.assert_allclose(tn_g, tn_o, atol=1e-12)
    assert np.isfinite(J_g) and J_g >= 0
    assert np.isfinite(J_o) and J_o >= 0
    _assert_feasible(X_g, U_g, 0.5, gen.u_max)
    _assert_feasible(X_o, U_o, 0.5, old.u_max)
