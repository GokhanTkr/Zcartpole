# Optional Siemens current-loop data

Some project source ZIPs include the original DaRUS CSV from [Hinze (2022), DaRUS V1](https://doi.org/10.18419/DARUS-2563), licensed CC BY 4.0. The CSV and its derived `reference_run` outputs are **excluded from PyPI distributions**. You can obtain the original dataset from the linked record if you do not have the source ZIP.

The dataset describes a force-controlled linear drive. The optional analyzer reads Siemens r0077 and r0078[0] as drive current setpoint and feedback in A RMS. This does not establish a calibrated horizontal cart force. The fitted transfer function has current units and must not be inserted as `motor.command_to_force` in N per command.

With the original CSV in the source ZIP, run:

```bash
zcartpole-identify-current examples/real_data/LDD_Transfer_function_Force_controlled.csv -o current_run
```

It writes `current_loop_fit.json` and `current_loop_prediction.csv`; add `--plot` with `[viz]` to save a PNG. The holdout is a later segment of the same acquisition, not an independent hardware experiment. A separate force sensor and system model are needed to map drive command to cart force. See [data attribution](data-license.md) and [validation](validation.md).
