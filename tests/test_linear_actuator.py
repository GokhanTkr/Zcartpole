"""Meaningful checks for the complete actuator-chain interface."""
import csv
import json

import numpy as np
import pytest

from zcartpole import LinearActuator, SystemConfig, fit_first_order_step, run_system


def test_transfer_response_and_velocity_channel():
    a = LinearActuator.from_transfer_functions(
        [2], [.04, 1], velocity_num=[-3], velocity_den=[1],
        command_min=-10, command_max=10, force_limit_n=20)
    Ad, Bu, Bv, Cm, Du, Dv = a.discrete(.02)
    x = np.zeros(1)
    mean = float(Cm@x+Du*4+Dv*.5)
    assert np.isclose(mean, 8*(1-.04/.02*(1-np.exp(-.02/.04)))-1.5)
    x1 = Ad@x+Bu*4+Bv*.5
    assert np.isclose(a.output(x1, 4, .5), 8*(1-np.exp(-.5))-1.5)
    assert a.command_for_force(1e6, x, .5) == 10


def test_first_order_fit_noisy_held_step():
    t = np.arange(0, 1.001, .005)
    u = np.where(t >= .15, 2., 0.)
    f = np.where(t >= .18, 6*(-np.expm1(-np.maximum(t-.18, 0)/.05)), 0.)
    f += np.random.default_rng(7).normal(0, .025, len(t))
    fit = fit_first_order_step(t, u, f)
    assert abs(fit['gain_n_per_command']-3) < .05
    assert abs(fit['time_constant_s']-.05) < .004
    assert abs(fit['command_delay_s']-.03) < .004
    with pytest.raises(ValueError, match='clean, held step'):
        fit_first_order_step(t, np.sin(20*t), f)


def test_balance_run_records_force_command_and_state(tmp_path):
    source = json.loads(open('examples/actuator_first_order_system.json', encoding='utf-8').read())
    source['links'] = source['links'][:1]
    source['initial']['angles_rad'] = [.04]
    source['initial']['angular_velocities_rad_s'] = [0]
    source['simulation'].update(mode='balance', duration_s=2.0)
    result = run_system(SystemConfig.from_dict(source), tmp_path/'balance')
    assert result.status == 'caught'
    with (tmp_path/'balance'/'trajectory.csv').open(newline='') as f:
        header = next(csv.reader(f))
    assert {'actuator_command_N', 'applied_force_n', 'requested_force_n',
            'actuator_state_1'} <= set(header)
    assert result.summary['simulation']['actuator_state_count'] == 1


def test_invalid_tf_and_state_dimensions():
    with pytest.raises(ValueError, match='proper'):
        LinearActuator.from_transfer_functions([1, 2, 3], [1, 2],
                                                command_min=-1, command_max=1,
                                                force_limit_n=1)
    with pytest.raises(ValueError, match='dimensions'):
        LinearActuator(np.eye(2), [1], [0, 0], [1, 0], 0, 0, -1, 1, 1)
