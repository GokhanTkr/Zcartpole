"""One-call engineering run with reproducible inputs and recorded outputs."""
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .actuation import (ActuatorLQRController, ActuatorTVLQRTracker,
                        simulate_actuated)
from .animation import save_animation
from .config import SystemConfig
from .dc_motor import simulate_dc_motor
from .linear_actuator import LinearActuator, simulate_linear_actuator
from .planning import plan_swingup
from ._version import __version__


@dataclass(frozen=True)
class RunResult:
    status: str
    output_dir: Path
    summary: dict


def _trajectory_csv(path, model, t, y, force, command=None, phase=None,
                    angle_origin=None, current=None, voltage=None,
                    applied_voltage=None, motor_resistance=None,
                    command_label='motor_command_n', requested_force=None,
                    actuator_states=None):
    n = model.n
    if phase is None:
        angles = np.unwrap(np.arctan2(y[3::3], y[2::3]), axis=1)
        if angle_origin is not None:
            origin = np.asarray(angle_origin)
            angles += (2*np.pi*np.round((origin-angles[:, 0])/(2*np.pi)))[:, None]
        omegas = y[4::3]
    else:
        angles = phase[2::2]
        omegas = phase[3::2]
    header = ['time_s', 'cart_position_m', 'cart_velocity_m_s']
    for i in range(n):
        header += [f'link_{i+1}_angle_rad', f'link_{i+1}_omega_rad_s',
                   f'link_{i+1}_tip_x_m', f'link_{i+1}_tip_y_m']
    header += ['mechanical_energy_j', 'cart_friction_force_n']
    header += (['nominal_force_n'] if command is None else
               ['applied_force_n', command_label])
    if requested_force is not None:
        header += ['requested_force_n']
    if actuator_states is not None:
        header += [f'actuator_state_{i+1}' for i in range(actuator_states.shape[0])]
    if current is not None:
        header += ['motor_current_a', 'voltage_command_v', 'applied_voltage_v',
                   'electrical_power_w', 'copper_loss_w']
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(t)):
            row = [t[k], y[0, k], y[1, k]]
            tip_x, tip_y = y[0, k], 0.0
            for i in range(n):
                tip_x += model.l[i]*np.sin(angles[i, k])
                tip_y += model.l[i]*np.cos(angles[i, k])
                row += [angles[i, k], omegas[i, k], tip_x, tip_y]
            velocity = y[1, k]
            friction = (-model.cart_damping*velocity
                        - model.cart_coulomb*np.tanh(
                            velocity/model.coulomb_velocity))
            row += [model.total_energy(y[:, k]), friction]
            row += [force[k]] if command is None else [force[k], command[k]]
            if requested_force is not None:
                row += [requested_force[k]]
            if actuator_states is not None:
                row += list(actuator_states[:, k])
            if current is not None:
                row += [current[k], voltage[k], applied_voltage[k],
                        applied_voltage[k]*current[k],
                        motor_resistance*current[k]**2]
            writer.writerow(row)


def run_system(config, output_dir):
    """Run a validated system description and save config, CSV and summary.

    ``output_dir`` must not exist, so earlier runs are never silently replaced.
    A failed plan still writes config and summary, without a fabricated path.
    A rail breach stops dynamics before the unmodeled collision.
    """
    if not isinstance(config, SystemConfig):
        config = (SystemConfig.from_json(config) if isinstance(config, (str, Path))
                  else SystemConfig.from_dict(config))
    d = config.data
    model = config.build_model()
    actuator = config.build_motor()
    is_linear = isinstance(actuator, LinearActuator)
    dc_motor = None if is_linear else actuator
    motor, initial, settings = d['motor'], d['initial'], d['simulation']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir/'config.json').write_text(
        json.dumps(d, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    summary = dict(schema_version=1, software_version=__version__,
                   created_utc=datetime.now(timezone.utc).isoformat(),
                   name=d['name'], mode=settings['mode'],
                   status=None, link_count=model.n,
                   rail_half_travel_m=d['rail']['half_travel_m'],
                   motor_model=('linear_actuator_with_cart_speed_input'
                                if is_linear else 'dc_electrical_with_gearing' if dc_motor else
                                'first_order_force_lag_with_discrete_dead_time'),
                   lqr_weights=d['lqr'], math_backend=model.backend,
                   effective_cart_mass_kg=model.M,
                   friction_model='viscous_plus_smoothed_coulomb',
                   plan=None, simulation=None)
    plan = None
    if settings['mode'] == 'swingup':
        # Plan with an ideal force input; validate the real actuator below.
        plan = plan_swingup(
            model, T=settings['plan_duration_s'], N=settings['plan_nodes'],
            x_max=d['rail']['half_travel_m'],
            cart_margin=settings['cart_margin_m'],
            th0=initial['angles_rad'], x0=initial['cart_position_m'],
            v0=initial['cart_velocity_m_s'],
            w0=initial['angular_velocities_rad_s'],
            seeds=tuple(settings['seeds']))
        summary['plan'] = dict(success=plan.success, message=plan.message,
                               J=plan.J, seed_k=plan.seed_k,
                               windings=plan.windings,
                               attempts=[asdict(a) for a in plan.attempts])
        if not plan.success:
            summary['status'] = 'planning_failed'
            (output_dir/'summary.json').write_text(
                json.dumps(summary, indent=2, ensure_ascii=False)+'\n',
                encoding='utf-8')
            return RunResult(summary['status'], output_dir, summary)
        _trajectory_csv(output_dir/'plan.csv', model, plan.t, plan.X,
                        plan.U, phase=plan.phase_X)
        design_tau = (None if is_linear else
                      dc_motor.nominal_force_time_constant_s if dc_motor else
                      motor['time_constant_s'])
        if design_tau:
            tracker = ActuatorTVLQRTracker(model, plan,
                                          tau=design_tau,
                                          force_weight=d['lqr']['actuator_force_weight'])
            control = tracker.control
        else:
            tracker = plan.tracker(model)
            control = lambda t, s, f: tracker.control(t, s)
        total_time = float(plan.t[-1])+settings['catch_extra_s']
    else:
        design_tau = (None if is_linear else
                      dc_motor.nominal_force_time_constant_s if dc_motor else
                      motor['time_constant_s'])
        if design_tau:
            control = ActuatorLQRController(model,
                         design_tau,
                         force_weight=d['lqr']['actuator_force_weight']).control
        else:
            control = lambda t, s, f: model.lqr_control(s)
        total_time = settings['duration_s']
    sim_kwargs = dict(
        x0=initial['cart_position_m'], v0=initial['cart_velocity_m_s'],
        w0=initial['angular_velocities_rad_s'], t_max=total_time,
        dt=settings['dt_s'], command_delay=motor['command_delay_s'],
        rail_limit=d['rail']['half_travel_m'],
        predict_delay=settings['predict_delay'])
    if is_linear:
        # Apply the transfer function, delay, and limits in closed loop.
        sol = simulate_linear_actuator(
            model, actuator, control, initial['angles_rad'],
            x0=initial['cart_position_m'], v0=initial['cart_velocity_m_s'],
            w0=initial['angular_velocities_rad_s'], t_max=total_time,
            dt=settings['dt_s'],
            initial_actuator_state=initial['actuator_state'],
            rail_limit=d['rail']['half_travel_m'],
            predict_delay=settings['predict_delay'])
    elif dc_motor:
        sol = simulate_dc_motor(model, dc_motor, control,
                                initial['angles_rad'],
                                initial_current=initial['motor_current_a'],
                                **sim_kwargs)
    else:
        sol = simulate_actuated(model, control, initial['angles_rad'],
                                tau=motor['time_constant_s'],
                                prediction_model=model,
                                prediction_tau=motor['time_constant_s'],
                                **sim_kwargs)
    _trajectory_csv(output_dir/'trajectory.csv', model, sol.t, sol.y,
                    sol.u, sol.command, angle_origin=initial['angles_rad'],
                    current=sol.current if dc_motor else None,
                    voltage=sol.voltage_command if dc_motor else None,
                    applied_voltage=sol.applied_voltage if dc_motor else None,
                    motor_resistance=dc_motor.resistance_ohm if dc_motor else None,
                    command_label=(f'actuator_command_{actuator.command_unit}'
                                   if is_linear else 'motor_command_n'),
                    requested_force=sol.requested_force if is_linear else None,
                    actuator_states=sol.actuator_state if is_linear else None)
    rail_breach = sol.rail_violation_time is not None
    if plan and not rail_breach:
        report = plan.evaluate_catch(model, sol)
        settled = report.success
        summary['catch'] = asdict(report)
    elif not rail_breach:
        final = sol.y[:, -1]
        error = model.error(final, model.top_state())
        n = model.n
        settled = (abs(error[0]) <= .05 and
                   np.max(np.abs(error[1:1+n])) <= .1 and
                   abs(error[1+n]) <= .1 and
                   np.max(np.abs(error[2+n:])) <= .2)
        summary['balance_terminal'] = dict(
            position_error_m=float(abs(error[0])),
            angle_error_rad=float(np.max(np.abs(error[1:1+n]))),
            velocity_error_m_s=float(abs(error[1+n])),
            angular_speed_error_rad_s=float(np.max(np.abs(error[2+n:]))),
            within_default_tolerances=bool(settled))
    else:
        settled = False
    summary['status'] = ('rail_violation' if rail_breach else
                         'caught' if settled else 'missed_catch')
    summary['simulation'] = dict(
        samples=int(sol.t.size), final_time_s=float(sol.t[-1]),
        first_rail_violation_s=sol.rail_violation_time,
        peak_cart_m=float(np.max(np.abs(sol.y[0]))),
        peak_applied_force_n=float(np.max(np.abs(sol.u))),
        peak_command_abs=float(np.max(np.abs(sol.command))),
        command_unit=actuator.command_unit if is_linear else
                     'N' if not dc_motor else 'N force request',
        final_cart_m=float(sol.y[0, -1]),
        final_angles_rad=[float(v) for v in np.angle(
            sol.y[2::3, -1]+1j*sol.y[3::3, -1])])
    if is_linear:
        summary['simulation'].update(
            peak_requested_force_n=float(np.max(np.abs(sol.requested_force))),
            command_saturation_fraction=float(np.mean(
                (sol.command <= actuator.command_min+1e-9) |
                (sol.command >= actuator.command_max-1e-9))),
            force_saturation_fraction=float(np.mean(
                np.abs(sol.u) >= actuator.force_limit_n-1e-9)),
            actuator_state_count=int(sol.actuator_state.shape[0]))
    elif dc_motor:
        dt_values = np.diff(sol.t)
        electrical_power = sol.applied_voltage*sol.current
        resistive_power = dc_motor.resistance_ohm*sol.current**2
        def integral(values):
            return float(np.sum(.5*(values[:-1]+values[1:])*dt_values))
        summary['simulation'].update(
            peak_current_a=float(np.max(np.abs(sol.current))),
            peak_voltage_command_v=float(np.max(np.abs(sol.voltage_command))),
            voltage_saturation_fraction=float(np.mean(
                np.abs(sol.voltage_command)>=dc_motor.voltage_limit_v-1e-9)),
            peak_applied_voltage_v=float(np.max(np.abs(sol.applied_voltage))),
            electrical_energy_j=integral(electrical_power),
            copper_loss_j=integral(resistive_power))
    if settings['animation']:
        save_animation(model, sol, output_dir/'simulation.gif',
                       x_limit=d['rail']['half_travel_m'])
    (output_dir/'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+'\n',
        encoding='utf-8')
    return RunResult(summary['status'], output_dir, summary)
