"""Brushed DC motor with current, back EMF, gearing and voltage limits.

The wheel rolls without slip. Constant transmission efficiency is a simple
approximation; backlash, static friction and thermal effects are not modeled.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class DCMotor:
    resistance_ohm: float
    inductance_h: float
    torque_constant_nm_a: float
    back_emf_constant_vs_rad: float
    gear_ratio: float
    wheel_radius_m: float
    efficiency: float
    voltage_limit_v: float
    current_limit_a: float
    current_loop_gain_ohm: float = 0.0
    rotor_inertia_kg_m2: float = 0.0

    def __post_init__(self):
        positive = ('resistance_ohm', 'inductance_h', 'torque_constant_nm_a',
                    'back_emf_constant_vs_rad', 'gear_ratio', 'wheel_radius_m',
                    'efficiency', 'voltage_limit_v', 'current_limit_a')
        for name in positive:
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.efficiency > 1:
            raise ValueError('efficiency must be <= 1')
        for name in ('current_loop_gain_ohm', 'rotor_inertia_kg_m2'):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

    @property
    def force_per_amp(self):
        return (self.efficiency*self.gear_ratio*self.torque_constant_nm_a
                / self.wheel_radius_m)

    @property
    def max_force_n(self):
        return self.force_per_amp*self.current_limit_a

    @property
    def reflected_mass_kg(self):
        return self.rotor_inertia_kg_m2*(self.gear_ratio/self.wheel_radius_m)**2

    @property
    def nominal_force_time_constant_s(self):
        return self.inductance_h/(self.resistance_ohm+self.current_loop_gain_ohm)

    def voltage_for_force(self, force, current, cart_speed):
        """Current reference with a proportional inner loop and back-EMF feedforward."""
        i_ref = np.clip(force/self.force_per_amp,
                        -self.current_limit_a, self.current_limit_a)
        omega = self.gear_ratio*cart_speed/self.wheel_radius_m
        voltage = (self.resistance_ohm*i_ref
                   + self.back_emf_constant_vs_rad*omega
                   + self.current_loop_gain_ohm*(i_ref-current))
        return float(np.clip(voltage, -self.voltage_limit_v,
                             self.voltage_limit_v))

    def electrical_step(self, current, voltage, cart_speed, h):
        """Exact current response at fixed cart speed; clamp at driver limit."""
        R, L = self.resistance_ohm, self.inductance_h
        omega = self.gear_ratio*cart_speed/self.wheel_radius_m
        target = (voltage-self.back_emf_constant_vs_rad*omega)/R
        a = np.exp(-R*h/L)
        end = target+(current-target)*a
        mean = target+(current-target)*(-np.expm1(-R*h/L))*L/(R*h)
        limit = self.current_limit_a
        if abs(end) > limit:
            boundary = np.sign(end)*limit
            if abs(current) < limit and (target-boundary)*(target-current) > 0:
                cross = -(L/R)*np.log((target-boundary)/(target-current))
                cross = float(np.clip(cross, 0, h))
                before = target*cross + (current-target)*(L/R)*(
                    -np.expm1(-R*cross/L))
                mean = (before+boundary*(h-cross))/h
            else:
                mean = boundary
            end = boundary
        return float(np.clip(end, -limit, limit)), float(np.clip(mean, -limit, limit))


def simulate_dc_motor(model, motor, controller, th0, *, t_max=8.0,
                      dt=.002, command_delay=0.0, x0=0.0, v0=0.0, w0=None,
                      initial_current=0.0, rail_limit=None,
                      predict_delay=False):
    """Couple motor current to cart dynamics; controller requests force in N.

    Voltage is held each step. Current is integrated analytically using the
    midpoint cart speed, iterating the mechanical step twice. Delayed voltage
    commands and optional queue forecast use the same motor equations.
    """
    if (not np.isfinite(t_max) or t_max <= 0 or not np.isfinite(dt) or dt <= 0
        or not np.isfinite(command_delay) or command_delay < 0
        or not np.isfinite(initial_current)
        or abs(initial_current) > motor.current_limit_a
        or (rail_limit is not None and
            (not np.isfinite(rail_limit) or rail_limit <= 0))):
        raise ValueError('Invalid DC simulation parameters')
    lag = int(round(command_delay/dt))
    if abs(lag*dt-command_delay) > 1e-9:
        raise ValueError('command_delay must be an integer multiple of dt')
    th0 = np.asarray(th0, dtype=float)
    w0 = np.zeros(model.n) if w0 is None else np.asarray(w0, dtype=float)
    if (th0.shape != (model.n,) or w0.shape != (model.n,)
        or not np.isfinite(th0).all() or not np.isfinite(w0).all()
        or not np.isfinite(x0) or not np.isfinite(v0)):
        raise ValueError('Invalid initial mechanical state')
    state = model.pack(x0, v0, np.exp(1j*th0), w0)
    current = float(initial_current)
    queue = [0.0]*lag
    t_values, y_values, force_values = [], [], []
    requested, voltages, applied_voltages, currents = [], [], [], []
    violation = None

    def advance(s, i, voltage):
        speed = s[1]
        trial_i, trial_mean = motor.electrical_step(i, voltage, speed, dt)
        trial_state = model.step(s, motor.force_per_amp*trial_mean, dt)
        mid_speed = (speed+trial_state[1])/2
        next_i, mean_i = motor.electrical_step(i, voltage, mid_speed, dt)
        next_state = model.step(s, motor.force_per_amp*mean_i, dt)
        return next_state, next_i

    steps = int(round(t_max/dt))
    for k in range(steps+1):
        t = k*dt
        if predict_delay and lag:
            forecast, forecast_current = state.copy(), current
            for pending in queue:
                forecast, forecast_current = advance(forecast,
                                                      forecast_current, pending)
            observed_t, observed_state, observed_current = (
                t+command_delay, forecast, forecast_current)
        else:
            observed_t, observed_state, observed_current = t, state, current
        force_request = float(np.clip(controller(
            observed_t, observed_state,
            motor.force_per_amp*observed_current),
            -model.u_max, model.u_max))
        voltage_cmd = motor.voltage_for_force(
            force_request, observed_current, observed_state[1])
        queue.append(voltage_cmd)
        applied = queue.pop(0)
        t_values.append(t)
        y_values.append(state.copy())
        currents.append(current)
        force_values.append(motor.force_per_amp*current)
        requested.append(force_request)
        voltages.append(voltage_cmd)
        applied_voltages.append(applied)
        if rail_limit is not None and abs(state[0]) > rail_limit+1e-5:
            violation = t
            break
        if k < steps:
            state, current = advance(state, current, applied)
    return SimpleNamespace(
        t=np.asarray(t_values), y=np.asarray(y_values).T,
        u=np.asarray(force_values), command=np.asarray(requested),
        voltage_command=np.asarray(voltages),
        applied_voltage=np.asarray(applied_voltages),
        current=np.asarray(currents), rail_violation_time=violation)
