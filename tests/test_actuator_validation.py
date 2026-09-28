"""Independent trace validation, including delayed multistep inputs."""
import csv
import hashlib
import json

import numpy as np
import pytest

from zcartpole import validate_actuator


def _experiment(path, *, amplitude=4, gain=1):
    t = np.arange(81)*.01
    u = np.where(t < .1, 0, np.where(t < .45, amplitude, -2.))
    f = gain*(amplitude*(-np.expm1(-np.maximum(t-.11, 0)/.02))
              +(-2-amplitude)*(-np.expm1(-np.maximum(t-.46, 0)/.02)))
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['time_s', 'command', 'force_n'])
        writer.writerows(zip(t, u, f))
    return path


def test_separate_data_multistep_and_threshold(tmp_path):
    train = _experiment(tmp_path/'train.csv', amplitude=5)
    holdout = _experiment(tmp_path/'holdout.csv', amplitude=4)
    model = tmp_path/'fitted.json'
    model.write_text(json.dumps({'gain_n_per_command':1,
                                 'time_constant_s':.02,
                                 'command_delay_s':.01,
                                 'source_sha256':hashlib.sha256(train.read_bytes()).hexdigest()}))
    good = validate_actuator(model, holdout, tmp_path/'good', max_rmse_n=1e-6)
    assert good['status'] == 'threshold_passed'
    assert good['rmse_n'] < 1e-8
    assert good['independence'] == 'different_content_fingerprint'
    assert (tmp_path/'good'/'validation.csv').exists()
    with pytest.raises(ValueError, match='identical'):
        validate_actuator(model, train, tmp_path/'same')
    bad = _experiment(tmp_path/'bad.csv', amplitude=4, gain=1.5)
    failed = validate_actuator(model, bad, tmp_path/'failed', max_rmse_n=.1)
    assert failed['status'] == 'threshold_failed'
    assert failed['rmse_n'] > .1


def test_full_transfer_config_and_missing_speed(tmp_path):
    source = json.loads(open('examples/actuator_transfer_system.json').read())
    holdout = _experiment(tmp_path/'holdout.csv')
    config = tmp_path/'system.json'
    config.write_text(json.dumps(source))
    result = validate_actuator(config, holdout, tmp_path/'config')
    assert result['rmse_n'] < 1e-8
    assert result['independence'] == 'unverified'
    source['motor']['velocity_to_force'] = {'numerator':[-3], 'denominator':[1]}
    config.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='velocity_m_s'):
        validate_actuator(config, holdout, tmp_path/'speed_missing')
