"""Check real PRBS import, unit boundary, and out-of-sample current prediction."""
from pathlib import Path

import pytest

from zcartpole.current_identification import (
    identify_current_loop, load_siemens_current_csv,
)


DATA = (Path(__file__).resolve().parents[1] / 'examples' / 'real_data' /
        'LDD_Transfer_function_Force_controlled.csv')


def test_real_siemens_prbs_is_current_not_force():
    t,u,y = load_siemens_current_csv(DATA)
    result,pred = identify_current_loop(t,u,y)
    assert len(t) == len(pred) == 2047
    assert result['samples'] == 2047
    assert result['force_calibrated'] is False
    assert result['validation_kind'] == 'contiguous_holdout_same_experiment'
    assert result['model_order'] == 2
    assert 0.8 < result['holdout_rmse_a'] < 1.1
    assert result['holdout_rmse_a'] < .2*result['holdout_baseline_rmse_a']
    assert abs(result['sampling_interval_s']-0.0000625) < 1e-12


def test_rejects_unmapped_or_malformed_siemens_columns(tmp_path):
    file = tmp_path / 'fake.csv'
    file.write_text('time_s,command,force_n\n0,0,0\n', encoding='utf-8')
    with pytest.raises(ValueError, match='original Siemens'):
        load_siemens_current_csv(file)
