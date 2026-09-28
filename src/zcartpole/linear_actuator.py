"""General continuous-time actuator with command and cart-speed inputs.

The output is horizontal cart force. A transfer function is converted to this
state-space representation; it may include the complete drive and gearbox.
"""
from dataclasses import dataclass
from functools import cached_property
from types import SimpleNamespace
import re

import numpy as np
from scipy.linalg import block_diag, expm
from scipy.signal import tf2ss


def _tf_realization(numerator, denominator):
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if (num.ndim != 1 or den.ndim != 1 or not len(num) or not len(den)
        or not np.isfinite(num).all() or not np.isfinite(den).all()
        or den[0] == 0 or len(num) > len(den)):
        raise ValueError('Transfer function must be finite, proper and have a nonzero leading denominator')
    if len(den) == 1:
        return (np.zeros((0, 0)), np.zeros((0, 1)),
                np.zeros((1, 0)), float(num[-1]/den[0]))
    A, B, C, D = tf2ss(num, den)
    return A, B, C, float(D[0, 0])


@dataclass(frozen=True)
class LinearActuator:
    A: np.ndarray
    B_command: np.ndarray
    B_velocity: np.ndarray
    C_force: np.ndarray
    D_command: float
    D_velocity: float
    command_min: float
    command_max: float
    force_limit_n: float
    delay_s: float = 0.0
    tracking_horizon_s: float = 0.002
    command_unit: str = 'command'

    def __post_init__(self):
        A = np.asarray(self.A, dtype=float)
        n = A.shape[0] if A.ndim == 2 else -1
        Bu = np.asarray(self.B_command, dtype=float).reshape(-1)
        Bv = np.asarray(self.B_velocity, dtype=float).reshape(-1)
        C = np.asarray(self.C_force, dtype=float).reshape(-1)
        if (n < 0 or A.shape != (n, n) or any(v.shape != (n,)
            for v in (Bu, Bv, C)) or not all(np.isfinite(v).all()
            for v in (A, Bu, Bv, C))):
            raise ValueError('Invalid actuator state-space dimensions or coefficients')
        for k, v in (('D_command', self.D_command),
                     ('D_velocity', self.D_velocity),
                     ('command_min', self.command_min),
                     ('command_max', self.command_max),
                     ('force_limit_n', self.force_limit_n),
                     ('delay_s', self.delay_s),
                     ('tracking_horizon_s', self.tracking_horizon_s)):
            if not np.isfinite(v):
                raise ValueError(f'{k} must be finite')
        if (self.command_min >= 0 or self.command_max <= 0
            or self.force_limit_n <= 0 or self.delay_s < 0
            or self.tracking_horizon_s <= 0
            or not isinstance(self.command_unit, str)
            or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', self.command_unit)):
            raise ValueError('Invalid actuator limits, tracking horizon or command unit')
        object.__setattr__(self, 'A', A)
        object.__setattr__(self, 'B_command', Bu)
        object.__setattr__(self, 'B_velocity', Bv)
        object.__setattr__(self, 'C_force', C)

    @classmethod
    def from_transfer_functions(cls, command_num, command_den, *,
                                velocity_num=(0,), velocity_den=(1,), **kwargs):
        Au, Bu, Cu, Du = _tf_realization(command_num, command_den)
        Av, Bv, Cv, Dv = _tf_realization(velocity_num, velocity_den)
        A = block_diag(Au, Av)
        zeros_u, zeros_v = np.zeros(len(Av)), np.zeros(len(Au))
        return cls(A, np.r_[Bu.ravel(), zeros_u],
                   np.r_[zeros_v, Bv.ravel()],
                   np.r_[Cu.ravel(), Cv.ravel()], Du, Dv, **kwargs)

    def output(self, state, command, velocity):
        value = (self.C_force@state+self.D_command*command
                 +self.D_velocity*velocity)
        return float(np.clip(value, -self.force_limit_n, self.force_limit_n))

    def discrete(self, h):
        """Exact ZOH transition and average force for constant velocity."""
        n = self.A.shape[0]
        M = np.zeros((n+3, n+3))
        M[:n, :n] = self.A
        M[:n, n+1] = self.B_command
        M[:n, n+2] = self.B_velocity
        M[n, :n] = self.C_force
        M[n, n+1] = self.D_command
        M[n, n+2] = self.D_velocity
        E = expm(M*h)
        return (E[:n, :n], E[:n, n+1], E[:n, n+2],
                E[n, :n]/h, E[n, n+1]/h, E[n, n+2]/h)

    def command_for_force(self, desired, state, velocity):
        Ad, Bu, Bv, _, _, _ = self._tracking_transition
        baseline = (self.C_force@(Ad@state+Bv*velocity)
                    +self.D_velocity*velocity)
        response = float(self.C_force@Bu+self.D_command)
        if abs(response) < 1e-10:
            raise ValueError('Command has negligible force response over tracking horizon')
        command = (desired-baseline)/response
        return float(np.clip(command, self.command_min, self.command_max))

    @cached_property
    def _tracking_transition(self):
        return self.discrete(self.tracking_horizon_s)


def simulate_linear_actuator(model, actuator, controller, th0, *,
                             t_max=8.0, dt=.002, x0=0.0, v0=0.0, w0=None,
                             initial_actuator_state=None, rail_limit=None,
                             predict_delay=False):
    """Mechanical trajectory and actuator state; stop at first rail breach."""
    if (not np.isfinite(t_max) or t_max <= 0 or not np.isfinite(dt) or dt <= 0
        or (rail_limit is not None and
            (not np.isfinite(rail_limit) or rail_limit <= 0))):
        raise ValueError('Invalid simulation parameters')
    lag = int(round(actuator.delay_s/dt))
    if abs(lag*dt-actuator.delay_s) > 1e-9:
        raise ValueError('Actuator delay must be an integer multiple of dt')
    th0 = np.asarray(th0, dtype=float)
    w0 = np.zeros(model.n) if w0 is None else np.asarray(w0, dtype=float)
    n = actuator.A.shape[0]
    a = (np.zeros(n) if initial_actuator_state is None else
         np.asarray(initial_actuator_state, dtype=float))
    if (th0.shape != (model.n,) or w0.shape != (model.n,)
        or a.shape != (n,) or not all(np.isfinite(v).all()
        for v in (th0, w0, a)) or not np.isfinite(x0) or not np.isfinite(v0)):
        raise ValueError('Invalid initial mechanical or actuator state')
    s = model.pack(x0, v0, np.exp(1j*th0), w0)
    queue = [0.0]*lag
    Ad, Bu, Bv, Cm, Du, Dv = actuator.discrete(dt)
    t_values, y_values, a_values, forces, commands, requests = [], [], [], [], [], []
    violation = None

    def advance(mech, state, command):
        speed = mech[1]
        next_a = Ad@state+Bu*command+Bv*speed
        mean = Cm@state+Du*command+Dv*speed
        trial = model.step(mech, float(np.clip(mean,
                    -actuator.force_limit_n, actuator.force_limit_n)), dt)
        speed = .5*(mech[1]+trial[1])
        next_a = Ad@state+Bu*command+Bv*speed
        mean = Cm@state+Du*command+Dv*speed
        next_mech = model.step(mech, float(np.clip(mean,
                    -actuator.force_limit_n, actuator.force_limit_n)), dt)
        return next_mech, next_a

    steps = int(round(t_max/dt))
    for k in range(steps+1):
        t = k*dt
        if predict_delay and lag:
            future_s, future_a = s.copy(), a.copy()
            for pending in queue:
                future_s, future_a = advance(future_s, future_a, pending)
            observed_t, observed_s, observed_a = (
                t+actuator.delay_s, future_s, future_a)
        else:
            observed_t, observed_s, observed_a = t, s, a
        observed_force = actuator.output(observed_a, 0, observed_s[1])
        desired = float(controller(observed_t, observed_s, observed_force))
        command = actuator.command_for_force(desired, observed_a,
                                              observed_s[1])
        queue.append(command)
        applied = queue.pop(0)
        t_values.append(t)
        y_values.append(s.copy())
        a_values.append(a.copy())
        forces.append(actuator.output(a, applied, s[1]))
        commands.append(command)
        requests.append(desired)
        if rail_limit is not None and abs(s[0]) > rail_limit+1e-5:
            violation = t
            break
        if k < steps:
            s, a = advance(s, a, applied)
    return SimpleNamespace(
        t=np.asarray(t_values), y=np.asarray(y_values).T,
        actuator_state=np.asarray(a_values).T,
        u=np.asarray(forces), command=np.asarray(commands),
        requested_force=np.asarray(requests), rail_violation_time=violation)
