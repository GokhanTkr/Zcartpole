"""Compare the JavaScript demo dynamics with the Python model."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from zcartpole import ZNCartPole, simulate

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs" / "demo"
RUNNER = Path(__file__).resolve().parent / "demo_runner.js"

CASES = {
    2: dict(m=(0.1, 0.1), l=(0.5, 0.5), u_max=50.0, th0=(0.05, -0.03)),
    3: dict(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4), u_max=80.0, th0=(0.05, -0.03, 0.02)),
}


def run_js(payload):
    proc = subprocess.run(["node", str(RUNNER), str(DEMO_DIR)], input=json.dumps(payload),
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


@pytest.mark.parametrize("n", [2, 3])
def test_js_matches_python(n):
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    if not (DEMO_DIR / "zcartpole.js").exists():
        pytest.skip("docs/demo/zcartpole.js is missing")

    case = CASES[n]
    model = ZNCartPole(m=case["m"], l=case["l"], u_max=case["u_max"])
    rng = np.random.default_rng(n)

    th, w = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    s_in = model.pack(0.1, 0.4, np.exp(1j * th), w)
    steps_loop, dt = 2000, 1e-3
    free_steps, free_dt = 1000, 1e-3
    payload = dict(
        model=dict(m=list(case["m"]), l=list(case["l"]), M=model.M, g=model.g,
                   u_max=case["u_max"], K=model.K.tolist()),
        step=dict(x=0.1, v=0.4, c=np.cos(th).tolist(), s=np.sin(th).tolist(),
                  w=w.tolist(), u=7.5, h=1e-2),
        loop=dict(th0=list(case["th0"]), dt=dt, steps=steps_loop),
        free=dict(th0=th.tolist(), w0=w.tolist(), steps=free_steps, dt=free_dt),
    )
    js = run_js(payload)

    def compare(js_state, py_state, tol):
        x, v, zs, ws = model.unpack(py_state)
        assert js_state["x"] == pytest.approx(x, abs=tol)
        assert js_state["v"] == pytest.approx(v, abs=tol)
        np.testing.assert_allclose(js_state["c"], zs.real, atol=tol)
        np.testing.assert_allclose(js_state["s"], zs.imag, atol=tol)
        np.testing.assert_allclose(js_state["w"], ws, atol=tol)

    
    compare(js["step"], model.step(s_in, 7.5, 1e-2), 1e-11)

    
    sol = simulate(model, lambda t, s: model.lqr_control(s), th0=case["th0"],
                   t_max=steps_loop * dt, dt=dt)
    compare(js["loop"], sol.y[:, -1], 1e-8)
    np.testing.assert_allclose(js["loop"]["U"], sol.u, atol=1e-6)

    
    free = simulate(model, lambda t, s: 0.0, th0=th, w0=w, t_max=free_steps * free_dt, dt=free_dt)
    compare(js["free"], free.y[:, -1], 1e-7)
