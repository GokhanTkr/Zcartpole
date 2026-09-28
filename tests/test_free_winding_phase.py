"""A periodic phase terminal constraint can reach distinct winding classes."""
import numpy as np
import pytest

from zcartpole import ZNCartPole
from experiments.e2_fair_benchmark import paired_seed, theta_optimize_swingup


def test_free_terminal_phase_accepts_both_winding_seeds():
    pytest.importorskip('casadi')
    model = ZNCartPole(m=(0.1,), l=(0.4,), u_max=80)
    th0 = (np.pi + .05,)
    for k in (0, 1):
        warm, _ = paired_seed(model, 3, 12, th0, k, 0.5)
        ok, _, X, _, _, _, _ = theta_optimize_swingup(
            model, 3, 12, 0.5, th0, None, 0.0, warm)
        assert ok
        assert X[2, -1]/(2*np.pi) == pytest.approx(k, abs=1e-5)
