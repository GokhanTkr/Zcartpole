# zcartpole

`zcartpole` simulates a cart with one or more serial pendulum links. Describe the rail, cart, links, drive, initial state, and controller in JSON. The runner saves the validated inputs, trajectory, and a result summary. It can balance near upright or plan and track a swing-up.

The actuator interface accepts a transfer function from drive command to horizontal cart force. It can represent a motor, gearbox, drive, and transmission together. A second transfer function can account for cart speed. Command delay, command bounds, and force bounds are applied in the simulation.

**Status:** simulation software. Included parameter values are examples. Check the model against measurements for hardware use. The swing-up optimizer uses an ideal force input; the chosen actuator is simulated afterward.

## Install

Python 3.9 or newer is required.

```bash
python -m pip install zcartpole
```

Swing-up uses CasADi/IPOPT; GIFs use Matplotlib and Pillow. The optional Numba backend speeds up mechanical steps after its first compilation:

```bash
python -m pip install 'zcartpole[opt,viz,speed]'
```

In PowerShell, use `py -m pip` and quote the extra spec in the same way. You can install from this source archive with `python -m pip install -e '.[opt,viz,speed]'` from the extracted folder. Examples and the documentation site are in the source archive; a normal wheel installation contains the Python package and command line tools.

## Use from Python

The mechanical API returns NumPy arrays. Plot them in the same Python session:

```python
import matplotlib.pyplot as plt
from zcartpole import ZNCartPole, simulate

model = ZNCartPole(m=[0.1], l=[0.4], M=1.0, u_max=80.0)
sol = simulate(model, lambda t, state: model.lqr_control(state),
               th0=[0.05], t_max=3.0, dt=0.002)
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time (s)')
plt.ylabel('Cart position (m)')
plt.show()
```

Install `[viz]` for this plot. This short example uses an ideal force input. For a configured actuator, use `SystemConfig` and `run_system` below, or call `simulate_linear_actuator` to keep its arrays in memory. The source archive contains `examples/python_balance.py`.

## Configured run

Save this as `balance.json`:

```json
{
  "schema_version": 1,
  "rail": {"half_travel_m": 0.5},
  "cart": {"mass_kg": 1.0},
  "links": [{"length_m": 0.4, "tip_mass_kg": 0.1}],
  "motor": {
    "type": "transfer_function",
    "command_to_force": {"numerator": [20.0], "denominator": [0.05, 1.0]},
    "command_unit": "V", "command_min": -4.0,
    "command_max": 4.0, "force_limit_n": 80.0,
    "command_delay_s": 0.01
  },
  "initial": {"angles_rad": [0.05]},
  "simulation": {"mode": "balance", "dt_s": 0.002, "duration_s": 3.0}
}
```

```python
from zcartpole import SystemConfig, run_system

config = SystemConfig.from_json('balance.json')
result = run_system(config, 'balance_run')
print(result.status, result.summary['simulation'])
```

The directory must be new. Read `balance_run/summary.json` first. `status` is `caught`, `missed_catch`, `rail_violation`, or `planning_failed`. Inspect `trajectory.csv` for the simulated force, command, cart motion, and link motion. `caught` means the final state meets built-in numerical tolerances; it is not a safety certificate.

The command line wrapper runs the same configuration:

```bash
python -m zcartpole balance.json -o balance_cli_run
```

On Windows, `py -m zcartpole balance.json -o balance_cli_run` works. `zcartpole-run` is equivalent. Use it for batch runs or shell automation.

Set `simulation.mode` to `swingup`, provide a near-hanging initial state, and install `[opt]` to plan a swing-up. For a complete three-link input, use `examples/actuator_transfer_system.json` from the source archive:

```bash
python -m zcartpole examples/actuator_transfer_system.json -o triple_run
```

Set `simulation.animation` to `true` and install `[viz]` to write `simulation.gif`. Planning uses a local nonlinear solver and can fail; a valid nominal plan can also fail when the drive is simulated.

## Documentation

The source archive includes a MkDocs site. From the project folder, install `.[docs]` and run `python -m mkdocs serve`. It covers configuration, actuator models, LQR weights, outputs, Python API, mathematics, and release steps. For code examples with n = 1, 2, or 3, run `python examples/run_links.py --links 2 --mode balance -o n2_balance` from the source folder.

## License

Original code and documentation: MIT (see `LICENSE`). A third-party DaRUS measurement CSV in some source archives has separate CC BY 4.0 terms (see `DATA_LICENSE.md`); it is excluded from PyPI distributions. No measured drive performance is implied by the sample models.
