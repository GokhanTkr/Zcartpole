# Code examples

The parameter values in these examples are for demonstration. Use the Python API for interactive work and arrays; use the command line for batch runs.

## Simulate and plot in Python

Install `.[viz]` from the source directory. Run `python examples/python_balance.py`, or paste the code below into a Python session:

```python
import numpy as np
import matplotlib.pyplot as plt
from zcartpole import ZNCartPole, simulate

model = ZNCartPole(m=[0.1], l=[0.4], M=1.0, u_max=80.0)
sol = simulate(model, lambda t, state: model.lqr_control(state),
               th0=[0.05], t_max=3.0, dt=0.002)
angle = np.unwrap(np.arctan2(sol.y[3], sol.y[2]))

plt.plot(sol.t, angle)
plt.xlabel('Time (s)')
plt.ylabel('Link angle (rad)')
plt.show()
```

`sol.t`, `sol.y`, and `sol.u` are arrays. This low-level example applies ideal force directly. Use `SystemConfig.from_dict(...)` and `run_system(...)` for a reproducible actuator run; a complete dictionary example is on the [Python API page](api.md#configured-run). `simulate_linear_actuator(...)` returns actuator and mechanical arrays in memory.

## Simulate an actuator in memory

```python
import matplotlib.pyplot as plt
from zcartpole import ZNCartPole, LinearActuator, simulate_linear_actuator

model = ZNCartPole(m=[0.1], l=[0.4], M=1.0, u_max=80.0)
drive = LinearActuator.from_transfer_functions(
    [20.0], [0.05, 1.0],
    command_min=-4.0, command_max=4.0,
    force_limit_n=80.0, delay_s=0.01, command_unit='V',
)
sol = simulate_linear_actuator(
    model, drive,
    controller=lambda t, state, force: model.lqr_control(state),
    th0=[0.05], t_max=3.0, dt=0.002,
    rail_limit=0.5, predict_delay=True,
)
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(sol.t, sol.y[0])
axes[0].set_ylabel('Cart position (m)')
axes[1].plot(sol.t, sol.u)
axes[1].set_ylabel('Applied force (N)')
axes[1].set_xlabel('Time (s)')
fig.tight_layout()
plt.show()
```

This code returns arrays in memory. `sol.command` is the issued voltage command, and `sol.rail_violation_time` records a rail crossing.

## Configured run from Python

```python
from zcartpole import SystemConfig, run_system

config = SystemConfig.from_json('examples/quick_transfer_balance.json')
result = run_system(config, 'balance_python')
print(result.status)
print(result.summary['simulation']['peak_cart_m'])
```

The JSON file is supplied in the source archive. `run_system` records normalized inputs, trajectory, and summary in a new directory. The same function accepts a dictionary via `SystemConfig.from_dict`.

## Command line examples

The commands below run from the extracted source directory. Each output directory must be new.

## Balance with one, two, or three links

Install the base package from the source directory:

```bash
python -m pip install -e .
python examples/run_links.py --links 1 --mode balance -o n1_balance
python examples/run_links.py --links 2 --mode balance -o n2_balance
python examples/run_links.py --links 3 --mode balance -o n3_balance
```

`examples/run_links.py` builds a `SystemConfig` with n link entries, runs `run_system`, and prints the path to `summary.json`. The script is a complete example of the Python API. A completed run writes `config.json`, `trajectory.csv`, and `summary.json`.

## Swing-up with one, two, or three links

Install the optional optimizer. Add `viz` only when saving a GIF:

```bash
python -m pip install -e '.[opt,viz]'
python examples/run_links.py --links 1 --mode swingup -o n1_swingup
python examples/run_links.py --links 2 --mode swingup -o n2_swingup
python examples/run_links.py --links 3 --mode swingup -o n3_swingup --animation
```

The last command writes `n3_swingup/simulation.gif` if planning produces a trajectory. The other results still need `summary.json` inspection: a valid plan can miss the catch. Swing-up solve time depends on the link count, seeds, nodes, and initial state.

## Run a JSON file

```bash
python -m zcartpole examples/quick_transfer_balance.json -o balance_json_run
python -m zcartpole examples/actuator_transfer_system.json -o triple_json_run
```

To change the motor, lengths, masses, initial state, or LQR weights, copy the JSON and edit the input fields. See [configuration](configuration.md) and [actuators](actuators.md). From a wheel installation, use an absolute path to a JSON file; example files are not installed with the wheel.

## Inspect the result with Python

```python
import csv
import json
from pathlib import Path

folder = Path('n2_balance')
summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
print(summary['status'])
print(summary['simulation']['peak_cart_m'])

with (folder / 'trajectory.csv').open(newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
print(rows[-1]['cart_position_m'])
print(rows[-1]['link_2_angle_rad'])
```

For swing-up, compare `plan.csv` (ideal force) with `trajectory.csv` (actuator simulation). The [output guide](results.md) lists all columns and statuses.

## Other source examples

| File | Purpose |
|---|---|
| `examples/actuator_first_order_system.json` | Identified first-order actuator format. |
| `examples/dc_motor_system.json` | Optional electrical motor format. |
| `examples/engineer_system.json` | Legacy force-lag format. |
| `examples/animate_link_counts.py` | GIFs for n = 1, 2, and 3 using an ideal-force simulation. |
| `examples/benchmark_math.py` | Warm mechanical-step timings for Python and Numba. |
| `examples/robustness_checks.py`, `examples/actuated_robustness.py` | Selected parameter scenarios. |
| `examples/actuator_step_measurement.csv`, `examples/actuator_validation_measurement.csv` | Input formats for optional fitting and validation commands. |

The browser [trajectory demo](demo/player.html) uses bundled data. It is separate from the JSON runner.
