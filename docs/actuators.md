# Actuator models

The JSON key is `motor` for historical reasons. It can describe the **whole chain** from a drive command to horizontal force on the cart. A stepper with gearbox, a servo, or a linear drive can all use the transfer-function interface if you have an appropriate command-to-cart-force model. NEMA frame size alone does not define that model.

## Transfer function (recommended general interface)

The unsaturated relation is

$$F(s)=G_c(s)C(s)+G_v(s)V(s),$$

where `C` is the drive command and `V` is cart speed. `G_v` is optional and defaults to zero. For example, `numerator: [20]` and `denominator: [0.05, 1]` represent $20/(0.05s+1)$ N/V when `command_unit` is `V`. Polynomial arrays run from highest power to constant. The transfer functions must be proper and have a nonzero leading denominator. They should describe the range of operation being simulated.

```json
"motor": {
  "type": "transfer_function",
  "command_to_force": {"numerator": [20.0], "denominator": [0.05, 1.0]},
  "velocity_to_force": {"numerator": [-2.0], "denominator": [1.0]},
  "command_unit": "V",
  "command_min": -4.0,
  "command_max": 4.0,
  "force_limit_n": 80.0,
  "command_delay_s": 0.01,
  "tracking_horizon_s": 0.002
}
```

The speed term here represents an illustrative `−2 N/(m/s)` effect. The command is clipped to `[command_min, command_max]`; the force is clipped to `±force_limit_n`. The command bounds must straddle zero, the force limit must be positive, and the delay must be nonnegative and an integer multiple of `simulation.dt_s`. `command_unit` is a simple label such as `V`, `A`, `N`, or `pulse_rate`. It does not convert units for you.

`tracking_horizon_s` is a positive horizon used to invert the local force response into a command. It does not add physical lag. If the command has almost no force response over this horizon, the runner raises an error; choose a horizon compatible with the identified dynamics. The current actuator state can be supplied with `initial.actuator_state`; otherwise it starts at zero. An unstable or poorly identified transfer function is not made safe by clipping.

## Other models

| `motor.type` | Required fields | Optional fields and interpretation |
|---|---|---|
| `identified_first_order` | `gain_n_per_command`, positive `time_constant_s`, `command_min`, `command_max`, `force_limit_n` | `velocity_gain_ns_m` (default 0); common linear-actuator fields. Implements $G_c=K/(\tau s+1)$ and a direct speed term. |
| `state_space` | `A`, `B_command`, `C_force`, command bounds and force limit | `B_velocity`, `D_command`, `D_velocity`; common linear-actuator fields. Uses $\dot a=Aa+B_cC+B_vV$, $F=C_fa+D_cC+D_vV$. Matrix and vector sizes must agree. |
| `force_lag` | `max_force_n` | `time_constant_s` and `command_delay_s` default 0. Legacy force-request model; requests and limits are in N. |
| `dc` | `resistance_ohm`, `inductance_h`, `torque_constant_nm_a`, `back_emf_constant_vs_rad`, `gear_ratio`, `wheel_radius_m`, `efficiency`, `voltage_limit_v`, `current_limit_a` | `current_loop_gain_ohm`, `rotor_inertia_kg_m2`, `command_delay_s` default 0. Explicit electrical and gearing approximation. |

The common linear-actuator fields are `command_unit` (default `command`), `command_delay_s` (default 0), and `tracking_horizon_s` (default 0.002 s). All three linear types require command bounds and a force limit. Supply a whole-chain model when using a geared or stepped system; do not derive motor force from a frame designation. The DC model assumes no wheel slip and fixed efficiency. See [model scope](scope.md).

## Choose and check parameters

1. Choose the command you will actually send to the drive. A transfer from controller request to wheel torque is incomplete if the belt and gearbox change the cart force.
2. Enter command and force limits separately. A linear transfer function does not encode saturation or delay.
3. Include the optional speed input if back EMF, loading, or drive behavior measurably changes force with cart speed.
4. For an engineering fit, collect actual cart force in newtons under known drive commands. Current feedback alone is not force calibration.
5. Check the [output summary](results.md) for clipping fractions and a missed catch. Validate on a separate input trace when available.

The swing-up optimizer uses an ideal bounded force. The selected actuator only enters the subsequent closed-loop simulation. This can change whether the cart catches the pendulum.
