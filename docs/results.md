# Outputs and run status

`python -m zcartpole INPUT.json -o NEW_DIR` makes a new directory and will not replace an existing run. The Python function returns a `RunResult(status, output_dir, summary)`. All runs save `config.json` (normalized input) and `summary.json` (result). A plan failure saves no simulated or nominal CSV.

| File | When written | Contents |
|---|---|---|
| `config.json` | Always after output directory is created | Validated inputs with defaults and schema version. |
| `summary.json` | Every completed runner call | Software version, time, mode, backend, weights, plan attempts, status, peaks and terminal checks. |
| `plan.csv` | Swing-up with a valid nominal plan | Ideal-force path with continuous link angles; it is not an actuator test. |
| `trajectory.csv` | A simulation runs | Actual cart and links, applied force, issued command, energy, friction. |
| `simulation.gif` | A trajectory exists and `simulation.animation=true` | Animated run; needs `[viz]`. |

## Read `summary.json`

`status` is `caught`, `missed_catch`, `rail_violation`, or `planning_failed`. `plan` is null in balance mode; in swing-up it contains `success`, `message`, `J`, `seed_k`, `windings`, and `attempts`. Each attempt contains solver convergence, validation, cost or null, winding count, iterations, runtime, reason, terminal error, and peak bounds. A solver success alone does not establish an acceptable plan.

`simulation` is null after `planning_failed`. Otherwise it records sample count, final time, first rail crossing, peak cart displacement, peak applied force, peak command, command unit, final cart position, and wrapped final angles. Linear-actuator runs also record peak requested force, command and force saturation fractions, and actuator state count. DC runs also record current and voltage peaks, voltage saturation, electrical energy, and copper loss. The balance result has `balance_terminal`; a swing-up tracked without rail breach has `catch`. If rail breach occurs, there is no terminal catch assessment.

The built-in terminal tolerances are 0.05 m cart position, 0.1 rad maximum angle, 0.1 m/s cart speed, and 0.2 rad/s maximum angular speed. `caught` means this numerical test passed at the final sample. It does not imply low peak force, no saturation, physical clearance, or hardware safety. A rail crossing stops the simulation without an impact model.

## Read `trajectory.csv`

All trajectories start with `time_s`, `cart_position_m`, `cart_velocity_m_s`. For each link i they include `link_i_angle_rad`, `link_i_omega_rad_s`, `link_i_tip_x_m`, `link_i_tip_y_m`. Then come `mechanical_energy_j`, `cart_friction_force_n` and force fields.

| Run | Force and command columns |
|---|---|
| `plan.csv` | `nominal_force_n`. |
| Linear actuator | `applied_force_n`, `actuator_command_<command_unit>`, `requested_force_n`, then `actuator_state_1`, ... |
| Force lag | `applied_force_n`, `motor_command_n`. |
| DC model | The same command columns, then `motor_current_a`, `voltage_command_v`, `applied_voltage_v`, `electrical_power_w`, `copper_loss_w`. |

Tip `y` is measured upward from the cart level. Angles in CSV are unwrapped to show turns. In `summary.json`, `final_angles_rad` uses wrapped angles. Applied force and issued command can differ due to actuator dynamics, dead time, and clipping. `requested_force_n` belongs to the feedback controller, not a sensor measurement.

## Diagnose a result

| Observation | Check first |
|---|---|
| `planning_failed` | `plan.attempts[].reason`, initial state, plan duration, nodes, seeds, rail margin, force bound. CasADi/IPOPT is required. |
| `missed_catch` | Final errors, requested vs applied force, command and force saturation, delay, and duration. |
| `rail_violation` | `first_rail_violation_s` and the cart path. No collision is calculated beyond crossing. |
| No GIF | Install `[viz]`, enable animation, and ensure simulation produced a trajectory. |
| `FileExistsError` or CLI input error | Choose a new output folder; fix JSON keys, array lengths, bounds, or delay/step mismatch. |

The CLI returns code 0 for `caught`, 1 for other completed statuses, and 2 on argparse input errors. Do not infer a physical pass from a shell code alone.
