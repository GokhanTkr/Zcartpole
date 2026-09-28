"""Check compiled mechanics against the Python reference."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from zcartpole import SystemConfig, ZNCartPole

pytest.importorskip('numba')


@pytest.mark.parametrize('n', [1, 2, 3, 5])
def test_compiled_acceleration_and_step_match_python(n):
    rng = np.random.default_rng(n)
    common = dict(
        m=rng.uniform(.05, .2, n), l=rng.uniform(.2, .6, n),
        M=1.0, u_max=80.0,
        rod_mass=rng.uniform(.01, .08, n),
        rod_com=rng.uniform(.3, .7, n),
        joint_damping=rng.uniform(.001, .01, n),
        cart_damping=.15, cart_coulomb=.2, coulomb_velocity=.02,
    )
    reference = ZNCartPole(**common, backend='python')
    compiled = ZNCartPole(**common, backend='numba')
    angles = rng.uniform(-2, 2, n)
    speed = rng.uniform(-1, 1, n)
    state = reference.pack(.1, -.2, np.exp(1j * angles), speed)
    for k in range(50):
        _, v, z, w = reference.unpack(state)
        u = 6 * np.sin(k / 7)
        np.testing.assert_allclose(
            compiled.accel(z, w, u, v), reference.accel(z, w, u, v),
            rtol=1e-12, atol=1e-12)
        a = reference.step(state, u, .002)
        b = compiled.step(state, u, .002)
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)
        state = a


def test_config_selects_and_reports_backend():
    path = Path(__file__).resolve().parents[1] / 'examples' / 'quick_transfer_balance.json'
    d = json.loads(path.read_text())
    d['simulation']['math_backend'] = 'numba'
    config = SystemConfig.from_dict(d)
    assert config.build_model().backend == 'numba'
    d = copy.deepcopy(d)
    d['simulation']['math_backend'] = 'python'
    assert SystemConfig.from_dict(d).build_model().backend == 'python'
    d['simulation']['math_backend'] = 'invalid'
    with pytest.raises(ValueError, match='math_backend'):
        SystemConfig.from_dict(d)
