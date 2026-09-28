"""Public end-to-end planner and failure-report contract."""
import numpy as np
import pytest

from zcartpole import ZNCartPole, plan_swingup


def test_phase_plan_exposes_trackable_z_trajectory():
    pytest.importorskip('casadi')
    model = ZNCartPole(m=(0.1,), l=(0.4,), u_max=80.0)
    plan = plan_swingup(model, T=3, N=12, th0=(np.pi + .05,))
    assert plan.success
    assert len(plan.attempts) == 2
    assert all(a.valid for a in plan.attempts)
    assert plan.J == min(a.J for a in plan.attempts)
    assert plan.X.shape == (5, 13)
    assert plan.phase_X.shape == (4, 13)
    assert plan.U.shape == plan.t.shape == (13,)
    assert np.max(np.abs(plan.X[2]**2 + plan.X[3]**2 - 1)) < 1e-12
    assert np.max(np.abs(plan.X[0])) <= plan.x_max - plan.cart_margin + 1e-5
    assert np.max(np.abs(plan.U)) <= model.u_max + 1e-5
    assert np.isfinite(plan.tracker(model).control(0.0, model.top_state()))


def test_solver_failure_records_attempts():
    pytest.importorskip('casadi')
    model = ZNCartPole(m=(0.1,), l=(0.4,), u_max=80.0)
    plan = plan_swingup(model, T=3, N=12, max_iter=1, seeds=(0, 1))
    assert not plan.success
    assert all(a.reason == 'ipopt_no_convergence' for a in plan.attempts)
    with pytest.raises(RuntimeError):
        plan.tracker(model)


def test_bad_parameters_rejected_before_solver():
    model = ZNCartPole(m=(0.1,), l=(0.4,), u_max=80.0)
    for kwargs in ({'th0': (1, 2)}, {'seeds': ()}, {'x_max': -1},
                   {'max_iter': 0}, {'N': 1}, {'cart_margin': 0.5}):
        with pytest.raises(ValueError):
            plan_swingup(model, **kwargs)
