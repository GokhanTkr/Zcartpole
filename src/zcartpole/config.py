"""Validated, versioned JSON description of a cart and serial pendulum."""
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from .core import ZNCartPole
from .dc_motor import DCMotor
from .linear_actuator import LinearActuator


def _object(value, path, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f'{path} must be an object')
    unknown, missing = set(value)-set(allowed), set(required)-set(value)
    if unknown or missing:
        raise ValueError(f'{path}: unknown {sorted(unknown)}, missing {sorted(missing)}')
    return value


def _number(value, path, *, lower=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (
        not np.isfinite(value) or (lower is not None and
        (value <= lower if strict else value < lower))
    ):
        raise ValueError(f'{path} must be finite and '
                         f'{">" if strict else ">="} {lower}' if lower is not None
                         else f'{path} must be a finite number')
    return float(value)


def _array(values, path, count):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f'{path} must be an array of length {count}')
    return tuple(_number(x, f'{path}[{i}]') for i, x in enumerate(values))


def _coefficients(values, path):
    if not isinstance(values, list) or not values:
        raise ValueError(f'{path} must be a nonempty coefficient array')
    return [_number(x, f'{path}[{i}]') for i, x in enumerate(values)]


def _tf_object(value, path):
    d = _object(value, path, ('numerator', 'denominator'),
                ('numerator', 'denominator'))
    return dict(numerator=_coefficients(d['numerator'], path+'.numerator'),
                denominator=_coefficients(d['denominator'], path+'.denominator'))


@dataclass(frozen=True)
class SystemConfig:
    """Physical model, actuator, initial state and run settings in SI units."""
    data: dict

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

    @classmethod
    def from_dict(cls, source):
        root = _object(source, 'config',
                       ('schema_version', 'name', 'rail', 'cart', 'links',
                        'motor', 'initial', 'simulation', 'gravity_m_s2', 'lqr'),
                       ('schema_version', 'rail', 'cart', 'links', 'motor',
                        'initial', 'simulation'))
        if type(root['schema_version']) is not int or root['schema_version'] != 1:
            raise ValueError('schema_version must be 1')
        name = root.get('name', 'cartpole')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('name must be nonempty text')
        rail = _object(root['rail'], 'rail', ('half_travel_m',), ('half_travel_m',))
        half = _number(rail['half_travel_m'], 'rail.half_travel_m', lower=0, strict=True)
        cart = _object(root['cart'], 'cart',
                       ('mass_kg', 'viscous_friction_ns_m', 'coulomb_friction_n',
                        'coulomb_smoothing_speed_m_s'), ('mass_kg',))
        mass = _number(cart['mass_kg'], 'cart.mass_kg', lower=0, strict=True)
        viscous = _number(cart.get('viscous_friction_ns_m', 0),
                          'cart.viscous_friction_ns_m', lower=0)
        coulomb = _number(cart.get('coulomb_friction_n', 0),
                          'cart.coulomb_friction_n', lower=0)
        smoothing = _number(cart.get('coulomb_smoothing_speed_m_s', .01),
                            'cart.coulomb_smoothing_speed_m_s', lower=0,
                            strict=True)
        links = root['links']
        if not isinstance(links, list) or not links:
            raise ValueError('links must be a nonempty array')
        clean_links = []
        for i, link in enumerate(links):
            label = f'links[{i}]'
            _object(link, label,
                    ('length_m', 'tip_mass_kg', 'rod_mass_kg',
                     'rod_com_fraction', 'rod_inertia_kg_m2',
                     'joint_damping_nms_rad'), ('length_m', 'tip_mass_kg'))
            length = _number(link['length_m'], f'{label}.length_m', lower=0,
                             strict=True)
            tip = _number(link['tip_mass_kg'], f'{label}.tip_mass_kg', lower=0)
            rod = _number(link.get('rod_mass_kg', 0),
                          f'{label}.rod_mass_kg', lower=0)
            if tip+rod <= 0:
                raise ValueError(f'{label} requires positive total mass')
            com = _number(link.get('rod_com_fraction', .5),
                          f'{label}.rod_com_fraction', lower=0)
            if com > 1:
                raise ValueError(f'{label}.rod_com_fraction must be <= 1')
            inertia = _number(link.get('rod_inertia_kg_m2', rod*length**2/12),
                              f'{label}.rod_inertia_kg_m2', lower=0)
            damping = _number(link.get('joint_damping_nms_rad', 0),
                              f'{label}.joint_damping_nms_rad', lower=0)
            clean_links.append(dict(length_m=length, tip_mass_kg=tip,
                                    rod_mass_kg=rod, rod_com_fraction=com,
                                    rod_inertia_kg_m2=inertia,
                                    joint_damping_nms_rad=damping))
        # The legacy "motor" key describes the whole drive-to-cart chain.
        motor = _object(root['motor'], 'motor',
                        ('type', 'max_force_n', 'time_constant_s',
                         'command_delay_s', 'resistance_ohm', 'inductance_h',
                         'torque_constant_nm_a', 'back_emf_constant_vs_rad',
                         'gear_ratio', 'wheel_radius_m', 'efficiency',
                         'voltage_limit_v', 'current_limit_a',
                         'current_loop_gain_ohm', 'rotor_inertia_kg_m2',
                         'command_to_force', 'velocity_to_force',
                         'gain_n_per_command', 'velocity_gain_ns_m',
                         'command_unit', 'command_min', 'command_max',
                         'force_limit_n', 'tracking_horizon_s',
                         'A', 'B_command', 'B_velocity', 'C_force',
                         'D_command', 'D_velocity'))
        motor_type = motor.get('type', 'force_lag')
        if motor_type == 'force_lag':
            _object(motor, 'motor', ('type', 'max_force_n', 'time_constant_s',
                                     'command_delay_s'), ('max_force_n',))
            max_force = _number(motor['max_force_n'], 'motor.max_force_n',
                                lower=0, strict=True)
            tau = _number(motor.get('time_constant_s', 0),
                          'motor.time_constant_s', lower=0)
            delay = _number(motor.get('command_delay_s', 0),
                            'motor.command_delay_s', lower=0)
            clean_motor = dict(type='force_lag', max_force_n=max_force,
                               time_constant_s=tau, command_delay_s=delay)
        elif motor_type == 'dc':
            fields = ('resistance_ohm', 'inductance_h',
                      'torque_constant_nm_a', 'back_emf_constant_vs_rad',
                      'gear_ratio', 'wheel_radius_m', 'efficiency',
                      'voltage_limit_v', 'current_limit_a')
            _object(motor, 'motor', ('type', 'command_delay_s',
                                     'current_loop_gain_ohm',
                                     'rotor_inertia_kg_m2', *fields), fields)
            clean_motor = dict(type='dc')
            for field in fields:
                clean_motor[field] = _number(motor[field], f'motor.{field}',
                                             lower=0, strict=True)
            for field in ('current_loop_gain_ohm', 'rotor_inertia_kg_m2',
                          'command_delay_s'):
                clean_motor[field] = _number(motor.get(field, 0),
                                             f'motor.{field}', lower=0)
            DCMotor(**{k:v for k,v in clean_motor.items()
                       if k not in ('type', 'command_delay_s')})
            delay = clean_motor['command_delay_s']
        elif motor_type in ('transfer_function', 'identified_first_order',
                            'state_space'):
            common = ('type', 'command_unit', 'command_min', 'command_max',
                      'force_limit_n', 'command_delay_s', 'tracking_horizon_s')
            fields = ({'transfer_function':('command_to_force',
                                            'velocity_to_force'),
                       'identified_first_order':('gain_n_per_command',
                                                 'time_constant_s',
                                                 'velocity_gain_ns_m'),
                       'state_space':('A', 'B_command', 'B_velocity',
                                      'C_force', 'D_command',
                                      'D_velocity')})[motor_type]
            required = ('command_min', 'command_max', 'force_limit_n')
            required += (('command_to_force',) if motor_type=='transfer_function'
                         else ('gain_n_per_command', 'time_constant_s')
                         if motor_type=='identified_first_order' else
                         ('A', 'B_command', 'C_force'))
            _object(motor, 'motor', common+fields, required)
            unit = motor.get('command_unit', 'command')
            if not isinstance(unit, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', unit):
                raise ValueError('motor.command_unit must be a simple unit label')
            clean_motor = dict(type=motor_type, command_unit=unit)
            for key in ('command_min', 'command_max', 'force_limit_n',
                        'command_delay_s', 'tracking_horizon_s'):
                clean_motor[key] = _number(motor.get(key, 0 if key.endswith('_s')
                                                and key!='tracking_horizon_s'
                                                else .002 if key=='tracking_horizon_s'
                                                else motor[key]), 'motor.'+key)
            delay = clean_motor['command_delay_s']
            if motor_type=='transfer_function':
                clean_motor['command_to_force'] = _tf_object(
                    motor['command_to_force'], 'motor.command_to_force')
                clean_motor['velocity_to_force'] = _tf_object(
                    motor.get('velocity_to_force',
                              {'numerator':[0], 'denominator':[1]}),
                    'motor.velocity_to_force')
            elif motor_type=='identified_first_order':
                clean_motor['gain_n_per_command'] = _number(
                    motor['gain_n_per_command'], 'motor.gain_n_per_command')
                clean_motor['time_constant_s'] = _number(
                    motor['time_constant_s'], 'motor.time_constant_s',
                    lower=0, strict=True)
                clean_motor['velocity_gain_ns_m'] = _number(
                    motor.get('velocity_gain_ns_m', 0),
                    'motor.velocity_gain_ns_m')
            else:
                clean_motor['A'] = motor['A']
                for key in ('B_command', 'B_velocity', 'C_force'):
                    clean_motor[key] = motor.get(key, [] if key=='B_velocity'
                                                  else motor[key])
                for key in ('D_command', 'D_velocity'):
                    clean_motor[key] = _number(motor.get(key, 0), 'motor.'+key)
            # Build once here to reject improper transfers or invalid matrices.
            temporary = cls.__new__(cls)
            object.__setattr__(temporary, 'data', {'motor':clean_motor})
            temporary.build_motor()
        else:
            raise ValueError('motor.type must be force_lag, dc, transfer_function, '
                             'identified_first_order or state_space')
        initial = _object(root['initial'], 'initial',
                          ('angles_rad', 'angular_velocities_rad_s',
                           'cart_position_m', 'cart_velocity_m_s',
                           'motor_current_a', 'actuator_state'),
                          ('angles_rad',))
        angles = _array(initial['angles_rad'], 'initial.angles_rad', len(links))
        omegas = _array(initial.get('angular_velocities_rad_s',
                                    [0]*len(links)),
                        'initial.angular_velocities_rad_s', len(links))
        x0 = _number(initial.get('cart_position_m', 0),
                     'initial.cart_position_m')
        v0 = _number(initial.get('cart_velocity_m_s', 0),
                     'initial.cart_velocity_m_s')
        current0 = _number(initial.get('motor_current_a', 0),
                           'initial.motor_current_a')
        if motor_type == 'dc':
            if abs(current0) > clean_motor['current_limit_a']:
                raise ValueError('initial.motor_current_a exceeds current limit')
        elif current0 != 0:
            raise ValueError('initial.motor_current_a requires a DC motor')
        actuator_state = initial.get('actuator_state')
        if actuator_state is not None:
            if motor_type not in ('transfer_function', 'identified_first_order',
                                  'state_space'):
                raise ValueError('initial.actuator_state requires a linear actuator')
            temporary = cls.__new__(cls)
            object.__setattr__(temporary, 'data', {'motor':clean_motor})
            actuator_state = list(_array(actuator_state, 'initial.actuator_state',
                                         len(temporary.build_motor().A)))
        if abs(x0) >= half:
            raise ValueError('initial.cart_position_m must be inside rail')
        sim = _object(root['simulation'], 'simulation',
                      ('mode', 'dt_s', 'plan_duration_s', 'plan_nodes',
                       'catch_extra_s', 'duration_s', 'cart_margin_m',
                       'seeds', 'predict_delay', 'animation', 'math_backend'),
                      ('mode', 'dt_s'))
        mode = sim['mode']
        if mode not in ('swingup', 'balance'):
            raise ValueError('simulation.mode must be swingup or balance')
        dt = _number(sim['dt_s'], 'simulation.dt_s', lower=0, strict=True)
        if delay and abs(round(delay/dt)*dt-delay) > 1e-9:
            raise ValueError('motor.command_delay_s must be a multiple of dt_s')
        duration = _number(sim.get('duration_s', 6), 'simulation.duration_s',
                           lower=0, strict=True)
        plan_T = _number(sim.get('plan_duration_s', 3*max(len(links)-1, 1)),
                         'simulation.plan_duration_s', lower=0, strict=True)
        extra = _number(sim.get('catch_extra_s', 3),
                        'simulation.catch_extra_s', lower=0)
        margin = _number(sim.get('cart_margin_m', .005),
                         'simulation.cart_margin_m', lower=0)
        if margin >= half:
            raise ValueError('simulation.cart_margin_m must be smaller than rail')
        nodes = sim.get('plan_nodes', 100)
        if type(nodes) is not int or nodes < 2:
            raise ValueError('simulation.plan_nodes must be an integer >= 2')
        seeds = sim.get('seeds', [0, 1])
        if not isinstance(seeds, list) or not seeds or any(
            type(k) is not int or k < 0 for k in seeds
        ):
            raise ValueError('simulation.seeds must be nonnegative integers')
        predict = sim.get('predict_delay', bool(delay))
        animate = sim.get('animation', False)
        if type(predict) is not bool or type(animate) is not bool:
            raise ValueError('simulation.predict_delay/animation must be booleans')
        math_backend = sim.get('math_backend', 'auto')
        if math_backend not in ('auto', 'python', 'numba'):
            raise ValueError('simulation.math_backend must be auto, python or numba')
        gravity = _number(root.get('gravity_m_s2', 9.81),
                          'gravity_m_s2', lower=0, strict=True)
        lqr = _object(root.get('lqr', {}), 'lqr',
                      ('position_weight', 'angle_weights',
                       'cart_velocity_weight', 'angular_velocity_weights',
                       'force_weight', 'actuator_force_weight'))
        n = len(links)
        def weights(key, default):
            values = _array(lqr.get(key, [default]*n), f'lqr.{key}', n)
            if any(v <= 0 for v in values):
                raise ValueError(f'lqr.{key} entries must be positive')
            return list(values)
        clean_lqr = dict(
            position_weight=_number(lqr.get('position_weight', 10),
                                    'lqr.position_weight', lower=0, strict=True),
            angle_weights=weights('angle_weights', 100*n),
            cart_velocity_weight=_number(lqr.get('cart_velocity_weight', 1),
                                         'lqr.cart_velocity_weight', lower=0,
                                         strict=True),
            angular_velocity_weights=weights('angular_velocity_weights', 10),
            force_weight=_number(lqr.get('force_weight', .1),
                                 'lqr.force_weight', lower=0, strict=True),
            actuator_force_weight=_number(lqr.get('actuator_force_weight', .1),
                                          'lqr.actuator_force_weight', lower=0,
                                          strict=True))
        return cls(dict(schema_version=1, name=name, gravity_m_s2=gravity,
                        rail=dict(half_travel_m=half),
                        cart=dict(mass_kg=mass, viscous_friction_ns_m=viscous,
                                  coulomb_friction_n=coulomb,
                                  coulomb_smoothing_speed_m_s=smoothing),
                        links=clean_links,
                        motor=clean_motor, lqr=clean_lqr,
                        initial=dict(angles_rad=list(angles),
                                     angular_velocities_rad_s=list(omegas),
                                     cart_position_m=x0, cart_velocity_m_s=v0,
                                     motor_current_a=current0,
                                     actuator_state=actuator_state),
                        simulation=dict(mode=mode, dt_s=dt, duration_s=duration,
                                        plan_duration_s=plan_T, catch_extra_s=extra,
                                        cart_margin_m=margin, plan_nodes=nodes,
                                        seeds=seeds, predict_delay=predict,
                                        animation=animate,
                                        math_backend=math_backend)))

    def build_model(self):
        d = self.data
        motor = self.build_motor()
        n = len(d['links'])
        tuning = d['lqr']
        Q = np.diag([tuning['position_weight'],
                     *tuning['angle_weights'], tuning['cart_velocity_weight'],
                     *tuning['angular_velocity_weights']])
        return ZNCartPole(
            m=[link['tip_mass_kg'] for link in d['links']],
            l=[link['length_m'] for link in d['links']],
            M=d['cart']['mass_kg']+(getattr(motor, 'reflected_mass_kg', 0)
                                   if motor else 0),
            g=d['gravity_m_s2'],
            u_max=(getattr(motor, 'max_force_n',
                           getattr(motor, 'force_limit_n', None)) if motor else
                   d['motor']['max_force_n']),
            Q=Q, R=np.array([[tuning['force_weight']]]),
            rod_mass=[link['rod_mass_kg'] for link in d['links']],
            rod_com=[link['rod_com_fraction'] for link in d['links']],
            rod_inertia=[link['rod_inertia_kg_m2'] for link in d['links']],
            joint_damping=[link['joint_damping_nms_rad'] for link in d['links']],
            cart_damping=d['cart']['viscous_friction_ns_m'],
            cart_coulomb=d['cart']['coulomb_friction_n'],
            coulomb_velocity=d['cart']['coulomb_smoothing_speed_m_s'],
            backend=d['simulation']['math_backend'])

    def build_motor(self):
        motor = self.data['motor']
        if motor['type'] == 'force_lag':
            return None
        if motor['type'] == 'transfer_function':
            command = motor['command_to_force']
            velocity = motor['velocity_to_force']
            return LinearActuator.from_transfer_functions(
                command['numerator'], command['denominator'],
                velocity_num=velocity['numerator'],
                velocity_den=velocity['denominator'],
                command_min=motor['command_min'],
                command_max=motor['command_max'],
                force_limit_n=motor['force_limit_n'],
                delay_s=motor['command_delay_s'],
                tracking_horizon_s=motor['tracking_horizon_s'],
                command_unit=motor['command_unit'])
        if motor['type'] == 'identified_first_order':
            return LinearActuator.from_transfer_functions(
                [motor['gain_n_per_command']], [motor['time_constant_s'], 1],
                velocity_num=[motor['velocity_gain_ns_m']],
                velocity_den=[1],
                command_min=motor['command_min'],
                command_max=motor['command_max'],
                force_limit_n=motor['force_limit_n'],
                delay_s=motor['command_delay_s'],
                tracking_horizon_s=motor['tracking_horizon_s'],
                command_unit=motor['command_unit'])
        if motor['type'] == 'state_space':
            return LinearActuator(
                motor['A'], motor['B_command'], motor['B_velocity'],
                motor['C_force'], motor['D_command'], motor['D_velocity'],
                motor['command_min'], motor['command_max'],
                motor['force_limit_n'], motor['command_delay_s'],
                motor['tracking_horizon_s'], motor['command_unit'])
        return DCMotor(**{k:v for k,v in motor.items()
                          if k not in ('type', 'command_delay_s')})
