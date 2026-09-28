"""Electrical actuator equations and configured LQR weights."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from zcartpole import DCMotor, SystemConfig, run_system


def _motor():
    return DCMotor(.3, .016, .02, .02, 10, .08, .8, 12, 40, .5,
                   rotor_inertia_kg_m2=1e-6)


def _config():
    path = Path(__file__).resolve().parents[1]/'examples'/'dc_motor_system.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_exact_current_step_back_emf_and_limits():
    motor = _motor()
    dt, voltage = .01, 6.0
    end, mean = motor.electrical_step(0, voltage, 0, dt)
    rate = motor.resistance_ohm/motor.inductance_h
    target = voltage/motor.resistance_ohm
    assert end == pytest.approx(target*(1-np.exp(-rate*dt)))
    assert mean == pytest.approx(target*(1-(1-np.exp(-rate*dt))/(rate*dt)))
    fast_end, _ = motor.electrical_step(0, voltage, 1.0, dt)
    assert fast_end < end
    assert motor.electrical_step(0, 12, 2.0, .1)[0] < (
        motor.electrical_step(0, 12, 0.0, .1)[0])
    limited, _ = motor.electrical_step(0, 12, 0, 1)
    assert limited == pytest.approx(motor.current_limit_a)
    assert motor.max_force_n == pytest.approx(80)
    assert motor.reflected_mass_kg == pytest.approx(.015625)
    assert abs(motor.voltage_for_force(80, 0, 4)) <= 12


def test_configured_lqr_weights_and_dc_balance_record(tmp_path):
    d = _config()
    d['links'] = d['links'][:1]
    d['initial']['angles_rad'] = [.02]
    d['initial']['angular_velocities_rad_s'] = [0]
    d['simulation'].update(mode='balance', duration_s=.4, animation=False)
    d['lqr']['angle_weights'] = [220]
    d['lqr']['angular_velocity_weights'] = [15]
    d['lqr']['force_weight'] = .3
    c = SystemConfig.from_dict(d)
    m = c.build_model()
    np.testing.assert_allclose(np.diag(m.Q), [10, 220, 1, 15])
    assert m.R[0, 0] == pytest.approx(.3)
    assert m.M == pytest.approx(1.015625)
    r = run_system(c, tmp_path/'dc')
    assert r.summary['motor_model'] == 'dc_electrical_with_gearing'
    with (tmp_path/'dc'/'trajectory.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert 'motor_current_a' in rows[0]
    assert 'voltage_command_v' in rows[0]
    assert max(abs(float(row['voltage_command_v'])) for row in rows) <= 12+1e-9
    assert max(abs(float(row['motor_current_a'])) for row in rows) <= 40+1e-9


def test_bad_motor_or_lqr_configuration_rejected():
    d = _config()
    d['motor']['efficiency'] = 1.1
    with pytest.raises(ValueError, match='efficiency'):
        SystemConfig.from_dict(d)
    d = _config()
    d['lqr']['angle_weights'] = [100]
    with pytest.raises(ValueError, match='length'):
        SystemConfig.from_dict(d)
