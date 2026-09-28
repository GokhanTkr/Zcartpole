# Getting started

## 1. Install

Use Python 3.9 or newer. Download and extract the repository ZIP, then run this from its root. For balance near upright, the base install is enough:

```bash
python -m pip install -e .
```

For swing-up, images, and faster mechanics, add the matching extras:

```bash
python -m pip install -e '.[opt,viz,speed]'
```

`opt` installs CasADi/IPOPT. `viz` installs Matplotlib and Pillow. `speed` installs Numba. `docs` installs MkDocs for local site previews. In Windows PowerShell, replace `python -m pip` with `py -m pip` if needed. Quote the bracket expression in both shells.

After the PyPI release, `python -m pip install 'zcartpole[opt,viz,speed]'` will also work. Example JSON files live in the repository, not in the installed wheel.

## 2. Simulate in Python

For arrays and plots in one Python session, install `[viz]` and run:

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

`sol.t`, `sol.y`, and `sol.u` are NumPy arrays. This example applies ideal force without an actuator. The full runnable file is `examples/python_balance.py`. For a configured actuator, see the [Python API](api.md).

## 3. Run a configured balance model

Use the source JSON example:

```bash
python -m zcartpole examples/quick_transfer_balance.json -o balance_run
```

From any directory, you can instead pass an absolute path to your own JSON file. A new output directory is required each run. For balance, start near upright: angle `0` is upright; angle `π` is hanging. Use one angle per link.

For one, two, or three links, run the [Python examples](examples.md). The script builds the JSON-equivalent configuration directly with `SystemConfig.from_dict`.

## 4. Run a three-link swing-up

From the extracted source directory:

```bash
python -m zcartpole examples/actuator_transfer_system.json -o triple_run
```

This example uses three links, a first-order transfer function, a 10 ms command delay, and a six-second nominal plan. It requires `[opt]`. It may take much longer than the balance example because it solves a nonlinear program for each seed. The values are illustrative; this run does not describe a specific motor.

To save a GIF, set `"animation": true` in `simulation` and install `[viz]`. The file will be `triple_run/simulation.gif` when a trajectory exists. The source example `examples/animate_link_counts.py` also generates GIFs for n = 1, 2, and 3.

## 5. Read the run

Start with `summary.json`: check `status`, `plan.success` for swing-up, `simulation.peak_cart_m`, `simulation.peak_applied_force_n`, saturation fractions when present, and `catch` or `balance_terminal`. Then plot `trajectory.csv`. `plan.csv` is the nominal ideal-force path, not the achieved response.

| Status | Meaning |
|---|---|
| `caught` | Final state meets the built-in numerical catch tolerances. |
| `missed_catch` | Simulation ended inside the rail but outside a catch tolerance. |
| `rail_violation` | Cart crossed the rail limit; the run stopped before any collision model. |
| `planning_failed` | No nominal swing-up candidate passed the planner checks. |

The runner returns shell exit code 0 only for `caught`, 1 for another run status, and argparse code 2 for input or file errors. See [outputs](results.md) for each file and field.

## 6. Change the model

Update lengths, masses, friction, and the complete command-to-force transfer function in the JSON. Set `force_limit_n` and command bounds separately. The transfer function can include the motor, gearbox, controller, and transmission. See the [configuration reference](configuration.md) and [actuator guide](actuators.md) for exact fields and units.
