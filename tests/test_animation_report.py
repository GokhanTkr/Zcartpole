"""Closed-loop report and GIF rendering contracts without solving an NLP."""
from types import SimpleNamespace

import numpy as np
import pytest

from zcartpole import ZNCartPole, SwingupResult, save_animation


def _sample():
    model = ZNCartPole(m=(0.1,), l=(0.4,), u_max=80)
    t = np.array([0.0, 0.5, 1.0])
    phase = np.zeros((4, 3))
    plan = SwingupResult(True, 'ok', (), (0.0,), 0.5, t=t,
                         phase_X=phase)
    y = np.zeros((5, 3))
    y[2] = 1
    sol = SimpleNamespace(t=t, y=y, u=np.zeros(3))
    return model, plan, sol


def test_report_catches_terminal_speed_and_path_limits():
    model, plan, sol = _sample()
    assert plan.evaluate_catch(model, sol).success
    sol.y[4, -1] = 0.3
    report = plan.evaluate_catch(model, sol)
    assert not report.caught and report.final_angular_speed == pytest.approx(0.3)
    sol.y[4, -1] = 0
    sol.y[0, 1] = 0.6
    report = plan.evaluate_catch(model, sol)
    assert report.caught and not report.within_limits and not report.success


def test_gif_contains_more_than_one_frame(tmp_path):
    pytest.importorskip('matplotlib')
    pillow = pytest.importorskip('PIL.Image')
    model, _, sol = _sample()
    sol.y[3, 0] = 1
    sol.y[2, 0] = 0
    output = save_animation(model, sol, tmp_path/'sample.gif', fps=5)
    with pillow.open(output) as im:
        assert im.n_frames >= 2
