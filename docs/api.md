# Python API

The public names below are imported from `zcartpole`. Use `run_system` for a configured run. Lower-level functions give direct access to the model and planner.

## Configured run

```python
from zcartpole import SystemConfig, run_system

config = SystemConfig.from_dict({
    'schema_version': 1,
    'rail': {'half_travel_m': 0.5},
    'cart': {'mass_kg': 1.0},
    'links': [{'length_m': 0.4, 'tip_mass_kg': 0.1}],
    'motor': {
        'type': 'transfer_function',
        'command_to_force': {
            'numerator': [20.0], 'denominator': [0.05, 1.0]},
        'command_unit': 'V', 'command_min': -4.0,
        'command_max': 4.0, 'force_limit_n': 80.0,
        'command_delay_s': 0.01,
    },
    'initial': {'angles_rad': [0.05]},
    'simulation': {'mode': 'balance', 'dt_s': 0.002, 'duration_s': 3.0},
})
result = run_system(config, 'balance_api_run')
print(result.status, result.output_dir / 'summary.json')
```

`SystemConfig.from_json(path)` and `.from_dict(mapping)` validate values, fill defaults, and reject unknown keys. `.data` is the normalized dictionary; `.build_model()` returns a `ZNCartPole`, and `.build_motor()` returns `LinearActuator`, `DCMotor`, or `None` for force lag. `run_system(config, output_dir)` also accepts a JSON path or a mapping. It requires a new output directory. `RunResult` holds `status: str`, `output_dir: pathlib.Path`, and `summary: dict`. The [output guide](results.md) explains statuses and files.

## Mechanical model

`ZNCartPole(m, l, M=1, g=9.81, u_max=50, Q=None, R=None, rod_mass=None, rod_com=0.5, rod_inertia=None, cart_damping=0, joint_damping=0, cart_coulomb=0, coulomb_velocity=0.01, backend='auto')` uses tip masses `m`, lengths `l`, cart mass `M`, force bound `u_max`, and optional rod and friction parameters. Per-link optional parameters accept scalars or n-vectors. `backend` is `auto`, `python`, or `numba`.

The mechanical state is a float vector of size `2+3*n`:

```text
[x, cart_speed, cos(theta_1), sin(theta_1), omega_1,
                cos(theta_2), sin(theta_2), omega_2, ...]
```

Construct one with `model.pack(x, v, complex_link_states, angular_speeds)`; unpack it with `model.unpack(state)`. `model.step(state, force, dt)` returns one integrated state. `model.accel(link_states, angular_speeds, force, cart_speed)` returns cart and joint accelerations. `model.mass_matrix(link_states)`, `total_energy(state)`, `momentum(state)`, `top_state()`, `error(state, reference)`, `design_lqr()`, and `lqr_control(state)` support analysis. `error` is local in the angles.

`simulate(model, controller, th0, x0=0, t_max=8, dt=0.001, w0=None, v0=0)` calls `controller(t, state)` and clips force to `model.u_max`. It returns an object with `t` of shape `(samples,)`, `y` of shape `(2+3*n, samples)`, and `u` of shape `(samples,)`. This is an ideal-force simulation, without the JSON runner's actuator or rail stop.

## Planning and tracking

```python
from zcartpole import ZNCartPole, plan_swingup

model = ZNCartPole(m=[0.1] * 3, l=[0.4] * 3, u_max=80.0)
plan = plan_swingup(model, T=6, N=100, x_max=0.5,
                    th0=[3.19, 3.17, 3.16], seeds=(0, 1))
if plan.success:
    sol = plan.simulate_catch(model, t_extra=3, dt=0.002)
    report = plan.evaluate_catch(model, sol)
    print(plan.windings, report.success)
else:
    print(plan.message, plan.attempts)
```

`plan_swingup` also accepts `cart_margin`, `x0`, `v0`, `w0`, `w_penalty`, and `max_iter`. CasADi/IPOPT is required. `SwingupResult` has `success`, `message`, `attempts`, and on success `t`, complex-layout `X`, force `U`, unwrapped `phase_X`, `J`, `seed_k`, and `windings`. Check `success` before using these arrays. `plan.tracker(model)` builds `TVLQRTracker` with `.control(t, state)`. `simulate_catch` tracks with an ideal force; `evaluate_catch(model, sol, cart_tol=..., speed_tol=..., angle_tol=..., omega_tol=..., limit_tol=...)` returns a `CatchReport` with terminal and path checks. For the configured real actuator, use `run_system` instead.

`RobustScenario(name, plant, th0=None, tau=0, command_delay=0)`, `assess_plan(model, plan, scenarios, ...)`, and `select_robust_swingup(model, scenarios, candidates, ...)` screen finite scenarios using the force-lag simulation. They do not give a probability of success or replace a transfer-function simulation.

## Actuator and utilities

`LinearActuator.from_transfer_functions(command_num, command_den, velocity_num=(0,), velocity_den=(1,), command_min=..., command_max=..., force_limit_n=..., delay_s=..., tracking_horizon_s=..., command_unit=...)` constructs a state-space realization. `simulate_linear_actuator(model, actuator, controller, th0, ...)` expects `controller(t, mechanical_state, observed_force)` to return a desired force and returns arrays `t`, `y`, `u`, `command`, `requested_force`, `actuator_state`, and `rail_violation_time`. See [actuator models](actuators.md).

`fit_first_order_step(time_s, command, force_n)` and `fit_step_csv(path)` fit a single held step. `validate_actuator(model_json, measurements_csv, output_dir, max_rmse_n=None, plot=False)` checks an independent force trace and writes a report. `save_animation(model, solution, output, fps=15, speed=1.0, ...)` writes a GIF with `[viz]` installed. `DCMotor`, `simulate_dc_motor`, `simulate_actuated`, `ActuatorTVLQRTracker`, and `ActuatorLQRController` support the optional legacy actuator paths. Consult function docstrings for their additional arguments.

Invalid configuration raises `ValueError`; missing optional packages may raise `ImportError`; output paths can raise `OSError`. IPOPT failure is reported through `plan.success` and `attempts`, rather than by treating every nonconvergence as an exception.
