"""The complex experiment must not reject a valid solution for its winding."""
import numpy as np
from experiments.e2_fair_benchmark import validate_z, paired_seed, native_pair_seed
from zcartpole.core import ZNCartPole


def test_validation_accepts_different_windings():
    model = ZNCartPole(m=(0.1,), l=(0.4,))
    for turns in (-1, 0, 2):
        t = np.linspace(0, 1, 101)
        X = np.zeros((5, len(t)))
        angle = 2*np.pi*turns*t
        X[2], X[3], X[4] = np.cos(angle), np.sin(angle), 2*np.pi*turns
        result = validate_z(model, X, T=1)
        assert result['valid']
        assert abs(result['achieved_windings'][0] - turns) < 1e-9


def test_invalid_unit_circle_rejected():
    model = ZNCartPole(m=(0.1,), l=(0.4,))
    X = np.zeros((5, 10))
    X[2] = 1.1
    assert not validate_z(model, X, T=1)['valid']


def test_small_discretization_drift_is_reported_and_accepted():
    model = ZNCartPole(m=(0.1,), l=(0.4,))
    X = np.zeros((5, 10))
    X[2] = np.sqrt(1 + 9.3e-5)
    result = validate_z(model, X, T=1)
    assert result['valid']
    np.testing.assert_allclose(result['unit_circle_error'], 9.3e-5, rtol=1e-8)


def test_paired_seed_represents_identical_physical_path():
    model = ZNCartPole(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4))
    for k in (0, 1, 2, 3):
        (th, ut), (z, uz) = paired_seed(model, 6, 100, (3.19, 3.17, 3.16), k, 0.5)
        np.testing.assert_allclose(th[:2], z[:2])
        np.testing.assert_allclose(ut, uz)
        for i in range(model.n):
            np.testing.assert_allclose(np.cos(th[2+2*i]), z[2+3*i])
            np.testing.assert_allclose(np.sin(th[2+2*i]), z[3+3*i])
            np.testing.assert_allclose(th[3+2*i], z[4+3*i])
            assert th[2+2*i, -1] == np.pi*2*k


def test_native_theta_seed_matches_z_default_values():
    model = ZNCartPole(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4))
    th0 = (np.pi + .05, np.pi + .03, np.pi + .02)
    X, U = native_pair_seed(model, 6, 100, th0, .5)
    tt = np.linspace(0, 1, 101)
    np.testing.assert_allclose(X[0], .3*np.sin(2*np.pi*tt))
    assert not U.any()
    for i in range(model.n):
        np.testing.assert_allclose(X[2+2*i], th0[i]+(np.pi-.05)*tt)
        np.testing.assert_allclose(X[3+2*i], np.pi/6)
