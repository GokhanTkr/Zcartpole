# Scope and limits

This is a simulation and planning library for serial cart pendulums. It uses
a rigid-link model with tip and rod masses, rod inertia, joint damping,
viscous cart friction, and smoothed Coulomb cart friction. The actuator may
be a command-to-force transfer function, state-space system, first-order
force lag, or an optional DC motor model.

The nominal swing-up optimizer enforces sampled rail and force limits.
It does not put the full actuator, command saturation, or delay inside
the optimization. The closed-loop simulation includes the selected
actuator model and checks the rail. A valid nominal plan may fail in
closed loop. IPOPT finds a local solution, if any.

The rail is a hard stop for the run: simulation ends at the first detected
crossing, without an impact model. There is no static sticking, backlash,
flexible-link motion, sensor noise, temperature, battery, or safety
controller. Force transfer functions are linear models used with separate
limits; they should only be used over a suitable operating range.

The bundled values and GIFs are illustrative, not validation against
hardware. Tests check software behavior and some equivalence with older
two- and three-link code. They do not certify physical accuracy. Compare
your inputs and outputs with measurements when making an engineering
decision.
