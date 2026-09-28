# Configuration reference

The runner reads one JSON object. `schema_version` must be `1`. Unknown keys are rejected. All numbers must be finite; angles are radians and physical inputs use SI units. `config.json` in each run is the normalized input with defaults filled in.

## Root and mechanical inputs

| Field | Required | Default | Meaning and limits |
|---|---:|---|---|
| `schema_version` | yes | — | Integer `1`. |
| `name` | no | `cartpole` | Nonempty run name. |
| `gravity_m_s2` | no | `9.81` | Positive gravity magnitude. |
| `rail.half_travel_m` | yes | — | Positive allowed distance of cart center from rail origin. |
| `cart.mass_kg` | yes | — | Positive cart mass. The DC model adds reflected rotor mass. |
| `cart.viscous_friction_ns_m` | no | `0` | Nonnegative viscous friction coefficient. |
| `cart.coulomb_friction_n` | no | `0` | Nonnegative smoothed Coulomb friction magnitude. |
| `cart.coulomb_smoothing_speed_m_s` | no | `0.01` | Positive transition speed. No static sticking is modeled. |
| `links` | yes | — | Nonempty ordered array from cart outward; one object per link. |
| `links[i].length_m` | yes | — | Positive length. |
| `links[i].tip_mass_kg` | yes | — | Nonnegative point mass at link end. |
| `links[i].rod_mass_kg` | no | `0` | Nonnegative distributed rod mass; rod plus tip mass must be positive. |
| `links[i].rod_com_fraction` | no | `0.5` | Rod center of mass as fraction of length, between 0 and 1. |
| `links[i].rod_inertia_kg_m2` | no | `rod_mass_kg * length_m² / 12` | Nonnegative rod inertia about its own center of mass. |
| `links[i].joint_damping_nms_rad` | no | `0` | Nonnegative joint damping. |

`rail.half_travel_m` is measured for the cart center, not a tip. Each link angle is absolute relative to vertical; link i is attached to link i−1. The cart friction force is `-b*v - Fc*tanh(v/vs)`.

## Initial state

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `initial.angles_rad` | yes | — | One absolute angle per link; 0 upright and π hanging. |
| `initial.angular_velocities_rad_s` | no | zeros | One value per link. |
| `initial.cart_position_m` | no | `0` | Must start strictly inside the rail. |
| `initial.cart_velocity_m_s` | no | `0` | Initial cart speed. |
| `initial.motor_current_a` | no | `0` | Only for `dc`; magnitude cannot exceed current limit. |
| `initial.actuator_state` | no | model default | State vector for a linear actuator; length must match its realization. |

## Simulation

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `simulation.mode` | yes | — | `balance` or `swingup`. |
| `simulation.dt_s` | yes | — | Positive simulation step. |
| `simulation.duration_s` | no | `6` | Positive balance duration. |
| `simulation.plan_duration_s` | no | `3 * max(n-1, 1)` | Positive nominal swing-up duration. |
| `simulation.plan_nodes` | no | `100` | Integer >= 2; more nodes raise optimization cost. |
| `simulation.catch_extra_s` | no | `3` | Nonnegative tracking time after nominal plan. |
| `simulation.cart_margin_m` | no | `0.005` | Nonnegative margin below half travel, used in planning. |
| `simulation.seeds` | no | `[0, 1]` | Nonempty list of nonnegative integers for distinct initial guesses. |
| `simulation.predict_delay` | no | `true` if delay > 0 | Enable controller delay prediction. |
| `simulation.animation` | no | `false` | Save `simulation.gif`; needs `[viz]`. |
| `simulation.math_backend` | no | `auto` | `auto`, `python`, or `numba`; explicit `numba` needs `[speed]`. |

`motor.command_delay_s` must be an integer multiple of `simulation.dt_s` for the runner. Initial states and weights must have arrays of exactly n entries. See [actuators](actuators.md) for the `motor` object.

## LQR weights

All weights must be positive. `angle_weights` and `angular_velocity_weights` each have n entries.

| Field | Default | Cost term |
|---|---:|---|
| `lqr.position_weight` | `10` | Cart position error squared. |
| `lqr.angle_weights` | `[100*n] * n` | Link angle errors squared. |
| `lqr.cart_velocity_weight` | `1` | Cart speed error squared. |
| `lqr.angular_velocity_weights` | `[10] * n` | Link angular speed errors squared. |
| `lqr.force_weight` | `0.1` | Ideal force request squared. |
| `lqr.actuator_force_weight` | `0.1` | Extra actuator force error term for legacy force-lag and DC controllers. |

These are relative numerical weights, not physical units. The general transfer-function actuator uses a local inversion of its force response; `actuator_force_weight` is not used to tune that inversion. See [LQR](math/lqr.md).
