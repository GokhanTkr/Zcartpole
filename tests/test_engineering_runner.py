"""User-facing configured simulation and recorded data."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from zcartpole import SystemConfig, ZNCartPole, run_system


def _example():
    path = Path(__file__).resolve().parents[1]/'examples'/'engineer_system.json'
    return json.loads(path.read_text(encoding='utf-8'))


def test_config_rejects_typos_and_invalid_delay():
    d = _example()
    d['cart']['mass_kgg'] = 1
    with pytest.raises(ValueError, match='unknown'):
        SystemConfig.from_dict(d)
    del d['cart']['mass_kgg']
    d['motor']['command_delay_s'] = .011
    with pytest.raises(ValueError, match='multiple'):
        SystemConfig.from_dict(d)


def test_coulomb_term_shared_by_numeric_and_symbolic_dynamics():
    ca = pytest.importorskip('casadi')
    from zcartpole.symbolic import symbolic_accel
    model = ZNCartPole(m=(.1,), l=(.4,), cart_coulomb=.2,
                       coulomb_velocity=.03)
    v, z, w, u = .08, np.exp(1j*.2), .3, 1.7
    actual = model.accel(np.array([z]), np.array([w]), u, v)
    symbolic = ca.Function('f', [], [symbolic_accel(
        model, [ca.DM(z.real)], [ca.DM(z.imag)], [ca.DM(w)],
        ca.DM(u), ca.DM(v))])
    np.testing.assert_allclose(np.asarray(symbolic()['o0']).ravel(), actual,
                               atol=1e-12)


def test_balance_writes_replayable_files_and_preserves_angle_branch(tmp_path):
    d = _example()
    d['links'] = d['links'][:1]
    d['initial']['angles_rad'] = [2*np.pi+.03]
    d['initial']['angular_velocities_rad_s'] = [0]
    d['simulation'].update(mode='balance', duration_s=.4, animation=False)
    output = tmp_path/'run'
    result = run_system(d, output)
    assert result.status in ('caught', 'missed_catch')
    assert (output/'summary.json').exists()
    assert (output/'config.json').exists()
    assert not (output/'plan.csv').exists()
    with (output/'trajectory.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]['link_1_angle_rad']) == pytest.approx(2*np.pi+.03)
    assert 'motor_command_n' in rows[0]
    assert json.loads((output/'summary.json').read_text())['link_count'] == 1
    with pytest.raises(FileExistsError):
        run_system(d, output)


def test_swingup_with_nonzero_initial_state(tmp_path):
    pytest.importorskip('casadi')
    d = _example()
    d['links'] = d['links'][:1]
    d['initial']['angles_rad'] = [np.pi+.05]
    d['initial']['angular_velocities_rad_s'] = [.02]
    d['initial']['cart_position_m'] = .01
    d['initial']['cart_velocity_m_s'] = .01
    d['simulation'].update(mode='swingup', plan_nodes=20, plan_duration_s=3,
                           seeds=[0], animation=False)
    result = run_system(d, tmp_path/'run')
    assert result.summary['plan']['success']
    with (tmp_path/'run'/'plan.csv').open(newline='') as f:
        first = next(csv.DictReader(f))
    assert float(first['cart_position_m']) == pytest.approx(.01, abs=1e-5)
    assert float(first['cart_velocity_m_s']) == pytest.approx(.01, abs=1e-5)
    assert float(first['link_1_omega_rad_s']) == pytest.approx(.02, abs=1e-5)
    assert 'nominal_force_n' in first
