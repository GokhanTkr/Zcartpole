# Third-party data

The file `examples/real_data/LDD_Transfer_function_Force_controlled.csv`
comes from:

Hinze, Christoph (2022), *Identification data for Linear Direct Drive
Forces compensation*, DaRUS, V1,
https://doi.org/10.18419/DARUS-2563.

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Some project source ZIPs include the CSV unchanged and analysis outputs in
`examples/real_data/reference_run/`. Both are excluded from PyPI source
and wheel distributions. The project's MIT license covers the code and
original documentation, not this third-party dataset.

The dataset describes this export as force-controlled drive data. The
optional analyzer interprets Siemens r0077 and r0078[0] as drive current
signals in A RMS. These labels do not establish a calibrated horizontal
force on the cart. Do not use the fit as `command_to_force` in N per command.
