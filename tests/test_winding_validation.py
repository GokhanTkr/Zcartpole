"""Regression tests for trajectory validation and winding aliasing."""
import numpy as np
import pytest
from zcartpole.core import ZNCartPole
from zcartpole.opt import _unit_nodes, unit_circle_error


def test_off_manifold_trajectory_is_not_silently_normalized():
    m = ZNCartPole(m=(0.1, 0.1), l=(0.5, 0.5))
    X = np.zeros((2 + 3*m.n, 4))
    for i in range(m.n):
        X[2+3*i] = 1
    X[2] = 1.01
    assert unit_circle_error(m, X) > 0.01
    with pytest.raises(ValueError):
        _unit_nodes(m, X)
    X[2] = 1 + 1e-8
    np.testing.assert_array_equal(_unit_nodes(m, X), X)


def test_winding_integral_detects_unwrap_aliasing():
    from experiments.e2_fair_benchmark import z_solution_windings
    T = 1.0
    # Two full turns, but sampling aliases to a constant angle.
    t = np.linspace(0, T, 5)
    X = np.zeros((5, len(t)))
    X[2] = np.cos(4*np.pi*t)
    X[3] = np.sin(4*np.pi*t)
    X[4] = 4*np.pi
    winding, diagnostic = z_solution_windings(X, 1, T=T, return_diagnostics=True)
    assert winding[0] == pytest.approx(2)
    assert diagnostic[0]['discrepancy'] >= 0.9
