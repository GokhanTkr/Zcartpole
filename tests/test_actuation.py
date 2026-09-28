"""Actuator response and compatibility with direct-force simulation."""
import numpy as np
import pytest

from zcartpole import (ZNCartPole, RobustScenario, plan_swingup,
                       select_robust_swingup, simulate, simulate_actuated)


def test_zero_lag_matches_direct_force_simulation():
    model = ZNCartPole(m=(.1,), l=(.4,), u_max=80)
    direct = simulate(model, lambda t, s: 12.0, th0=(.03,),
                      t_max=.2, dt=.002)
    actuated = simulate_actuated(model, lambda t, s, f: 12.0,
                                 th0=(.03,), t_max=.2, dt=.002)
    np.testing.assert_allclose(actuated.y, direct.y, atol=1e-12)
    np.testing.assert_allclose(actuated.u, direct.u, atol=1e-12)


def test_first_order_response_and_dead_time():
    model = ZNCartPole(m=(.1,), l=(.4,), u_max=80)
    result = simulate_actuated(model, lambda t, s, f: 12.0,
                               th0=(.03,), t_max=.2, dt=.002,
                               tau=.02, command_delay=.01)
    expected = 12*(1-np.exp(-np.maximum(result.t-.01, 0)/.02))
    np.testing.assert_allclose(result.u, expected, atol=1e-12)
    assert result.command[0] == pytest.approx(12.0)


def test_stop_at_rail_before_unmodeled_collision():
    model = ZNCartPole(m=(.1,), l=(.4,), u_max=80)
    result = simulate_actuated(model, lambda t, s, f: 80.0,
                               th0=(.03,), t_max=2, dt=.002,
                               rail_limit=.1)
    assert result.rail_violation_time is not None
    assert result.t[-1] == pytest.approx(result.rail_violation_time)
    assert result.t[-1] < 2.0


def test_predictor_passes_future_time_after_queued_commands():
    model = ZNCartPole(m=(.1,), l=(.4,), u_max=80)
    observed = []

    def control(t, state, force):
        observed.append((t, state.copy(), force))
        return 12.0

    simulate_actuated(model, control, th0=(.03,), t_max=.02, dt=.002,
                      tau=.02, command_delay=.01, predict_delay=True,
                      prediction_tau=.02)
    assert observed[0][0] == pytest.approx(.01)
    assert observed[0][2] == pytest.approx(0)
    assert observed[1][0] == pytest.approx(.012)
    assert observed[1][2] == pytest.approx(12*(1-np.exp(-.002/.02)))


def test_candidate_screen_reports_no_passing_candidate():
    pytest.importorskip('casadi')
    model = ZNCartPole(m=(.1,), l=(.4,), u_max=80)
    chosen = select_robust_swingup(
        model, [RobustScenario('nominal', model)],
        [{'T':3, 'cart_margin':.005, 'seeds':(0,)}],
        N=12, th0=(np.pi+.05,), dt=.01,
        min_cart_clearance=.49)
    assert chosen.candidates[0].plan.success
    assert not chosen.success
    assert chosen.plan is None
    assert len(chosen.candidates[0].scenarios) == 1
