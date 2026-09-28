# Actuator fitting and validation

A transfer function can come from engineering specifications or measured cart force. The optional fitting command estimates only a **first-order held-step** response at fixed cart speed. Fitting is not needed when you already have a suitable model.

## Fit a measured force step

Prepare CSV columns `time_s,command,force_n`. Collect a settled baseline, one clean held command step, and at least eight samples afterward (at least 12 samples total). The cart should be held at zero speed for this fit.

```bash
zcartpole-fit-actuator measurements_step.csv -o fit.json
```

The fit reports `gain_n_per_command`, `time_constant_s`, `command_delay_s`, baseline force, RMSE, and a source SHA-256. It does **not** estimate command limits, force limits, or speed dependence. Enter those from your system. When using a fitted delay in a simulation JSON, choose `simulation.dt_s` so delay is an integer multiple, or choose a rounded delay and record that approximation.

## Validate on another trace

Use a different measurement CSV, also with `time_s,command,force_n`. Add `velocity_m_s` if the model has a speed-to-force term.

```bash
zcartpole-validate-actuator fit.json separate_measurement.csv -o validation_run --max-rmse-n 5
```

The model argument can also be a full system JSON with a linear actuator. The validation run writes `summary.json`, `validation.csv`, and a copy of `model.json`. Add `--plot` with `[viz]` installed for `validation.png`. The `--max-rmse-n` option sets a user-defined acceptance threshold: exit code 1 means the threshold failed. The validator rejects byte-identical fitting and validation CSVs when the fit carries a source hash. A full model without that hash reports independence as unverified. A different hash does not prove a truly independent experiment.

The validator assumes an initial steady-state actuator and uses the measured pre-step force to set an offset. Check the reported RMSE, peak error, input range, clipping, and warnings in your operating range. Good current-loop prediction is not cart-force validation.

## Optional current-loop sample

The source ZIP includes a third-party Siemens current-loop CSV. `zcartpole-identify-current` accepts its original semicolon-delimited format and writes a current transfer estimate. This is **current setpoint to current feedback**, not command to cart force. See [the data note](real_current_data.md) and [attribution](data-license.md). The CSV is excluded from PyPI source and wheel distributions.
