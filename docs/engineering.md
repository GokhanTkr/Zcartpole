# Model and control workflow

## Define the model

1. Enter the usable rail half travel for the cart center.
2. Enter cart mass and ordered links from the cart outward. Each link needs a length and tip mass. Add rod mass, inertia, damping, and friction as needed.
3. Set one initial angle per link. Angle `0` is upright; `π` is hanging.
4. Enter a command-to-cart-force [actuator model](actuators.md), command bounds, force limit, and delay.
5. Run a [balance or swing-up example](examples.md) with a new output directory.

The exact JSON keys, units, and defaults are in [configuration](configuration.md). A gearbox can be part of the transfer function. Current feedback alone does not give a command-to-cart-force model.

## Tune balance control

Run `balance` from small initial angles. Check `balance_terminal`, peak cart travel, applied force, command saturation, and force saturation in `summary.json`. Use the per-link `angle_weights` to adjust angle priority, `position_weight` for cart centering, and `force_weight` for force request cost. Change one weight at a time and compare the resulting trajectories.

The mechanical LQR requests force. With a transfer-function actuator, a local inversion converts that request to a bounded command. `actuator_force_weight` applies to the augmented legacy force-lag and DC controllers; it does not tune the transfer-function inversion.

## Run and check swing-up

Set `simulation.mode` to `swingup`, start near hanging, and install `[opt]`. The solver searches from the supplied `seeds` with free final winding. `plan.csv` stores a nominal ideal-force path. `trajectory.csv` stores the result after actuator dynamics, delay, and limits.

Check `plan.success`, each attempt's reason, terminal catch values, peak cart travel, and saturation fractions. Change `plan_duration_s`, `plan_nodes`, `seeds`, or `cart_margin_m` if the nominal solve fails. If the closed-loop catch fails, inspect the requested and applied forces before changing LQR weights or actuator limits.

For numerical sensitivity, reduce `dt_s` and compare the result. The Numba backend speeds mechanical steps; the optimizer remains in CasADi/IPOPT.

## Model limits

Rail impact, static sticking, backlash, flexible rods, sensor noise, thermal limits, and hardware safety logic are outside the model. Parameters in the included examples are not measured hardware specifications. See [scope](scope.md) and [actuator validation](validation.md).
