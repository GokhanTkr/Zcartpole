"""Finite-scenario closed-loop checks and conservative plan selection.

These are deterministic tests of specified plants and actuator responses;
passing them does not imply chance constraints or hardware guarantees.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .actuation import simulate_actuated
from .planning import CatchReport, SwingupResult, plan_swingup


@dataclass(frozen=True)
class RobustScenario:
    name: str
    plant: object
    th0: Optional[Tuple[float, ...]] = None
    tau: float = 0.0
    command_delay: float = 0.0


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    success: bool
    first_rail_violation_s: Optional[float]
    max_cart: float
    max_force: float
    catch: Optional[CatchReport]


@dataclass(frozen=True)
class CandidateResult:
    settings: dict
    plan: SwingupResult
    scenarios: Tuple[ScenarioResult, ...]
    success: bool


@dataclass(frozen=True)
class RobustPlanResult:
    success: bool
    plan: Optional[SwingupResult]
    candidates: Tuple[CandidateResult, ...]
    message: str


def assess_plan(model, plan, scenarios, *, dt=0.002, t_extra=3.0,
                controller_factory=None, predict_delay=False,
                prediction_tau=None):
    """Run one plan through the supplied plants; stop at the first rail breach."""
    if not plan.success:
        raise ValueError('A valid nominal plan is required')
    if not np.isfinite(t_extra) or t_extra < 0:
        raise ValueError('t_extra must be finite and nonnegative')
    scenarios = tuple(scenarios)
    if not scenarios or len({c.name for c in scenarios}) != len(scenarios):
        raise ValueError('Scenarios need distinct names and at least one entry')
    tracker = plan.tracker(model) if controller_factory is None else (
        controller_factory(model, plan))
    control = (lambda t, s, f: tracker.control(t, s)) if (
        controller_factory is None) else tracker.control
    results = []
    for case in scenarios:
        if case.plant.n != model.n:
            raise ValueError('Each plant must have the same link count')
        angles = plan.th0 if case.th0 is None else case.th0
        sol = simulate_actuated(
            case.plant, control, angles,
            x0=plan.x0, v0=plan.v0, w0=plan.w0,
            t_max=float(plan.t[-1])+t_extra, dt=dt, tau=case.tau,
            command_delay=case.command_delay, rail_limit=plan.x_max,
            predict_delay=predict_delay, prediction_model=model,
            prediction_tau=prediction_tau)
        peak_cart = float(np.max(np.abs(sol.y[0])))
        peak_force = float(np.max(np.abs(sol.u)))
        report = None if sol.rail_violation_time is not None else (
            plan.evaluate_catch(model, sol))
        results.append(ScenarioResult(case.name, bool(report and report.success),
                                      sol.rail_violation_time, peak_cart,
                                      peak_force, report))
    return tuple(results)


def select_robust_swingup(model, scenarios, candidates, *, N=100, x_max=0.5,
                          th0=None, x0=0.0, v0=0.0, w0=None,
                          dt=0.002, t_extra=3.0,
                          controller_factory=None, predict_delay=False,
                          prediction_tau=None, min_cart_clearance=0.0):
    """Try nominal planner settings, then select the cheapest passing candidate.

    ``candidates`` is a sequence of dictionaries of ``plan_swingup`` options
    such as ``{'T': 8, 'cart_margin': 0.02}``. Each candidate is optimized only
    for the nominal model. Feasibility is checked on every supplied scenario;
    if none passes, ``success=False`` and ``plan=None``. This is finite-case
    screening, not a min-max NLP or a statistical robustness guarantee.
    """
    scenarios, candidates = tuple(scenarios), tuple(candidates)
    if not scenarios or not candidates:
        raise ValueError('Scenarios and candidate settings cannot be empty')
    if (not np.isfinite(min_cart_clearance) or min_cart_clearance < 0
            or min_cart_clearance >= x_max):
        raise ValueError('min_cart_clearance must be in [0, x_max)')
    tried = []
    best = None
    for settings in candidates:
        if not isinstance(settings, dict) or any(
            k not in {'T', 'cart_margin', 'seeds', 'w_penalty', 'max_iter'}
            for k in settings):
            raise ValueError('Invalid candidate settings')
        plan = plan_swingup(model, N=N, x_max=x_max, th0=th0,
                           x0=x0, v0=v0, w0=w0, **settings)
        checks = assess_plan(model, plan, scenarios, dt=dt, t_extra=t_extra,
                             controller_factory=controller_factory,
                             predict_delay=predict_delay,
                             prediction_tau=prediction_tau) if (
                                 plan.success) else ()
        passed = bool(checks) and all(
            c.success and c.max_cart <= x_max-min_cart_clearance+1e-9
            for c in checks)
        tried.append(CandidateResult(dict(settings), plan, checks, passed))
        if passed and (best is None or plan.J < best.J):
            best = plan
    return RobustPlanResult(best is not None, best, tuple(tried),
                            'ok' if best is not None else
                            'No candidate passed all supplied scenarios')
