# Time integration

For a held force over time step $h$, the mechanical solver uses fourth-order Runge–Kutta for $x$, $v$, $\omega_i$, and incremental angles $\phi_i$. At each stage it evaluates $z_i^{\mathrm{stage}}=z_i^k e^{\mathrm{i}\phi_i^{\mathrm{stage}}}$. After the RK4 increment it stores

$$z_i^{k+1}=z_i^k e^{\mathrm{i}\Delta\theta_i}.$$

This keeps $|z_i|$ close to one without wrapping angles or projecting the state. It is **not** an exact energy-conserving integrator. Damping, applied force, discrete actuator updates, and step size all affect energy. For a numerical sensitivity check, halve `simulation.dt_s` and compare trajectories and terminal results.

Linear actuator states use a zero-order-hold matrix exponential for each step, with constant command and a local speed approximation. The force used for the mechanical step is a clipped average over that interval. Command delay uses an integer-length queue; the runner requires `command_delay_s / dt_s` to be integral. Delay prediction projects the model forward over queued commands. Closed-loop rail crossing is checked at stored step boundaries and ends the run without an impact calculation.

`math_backend='numba'` compiles the mechanical acceleration and step. `auto` uses it if installed; otherwise it uses the NumPy reference implementation. Compilation adds one-time cost. It does not compile or speed up the CasADi/IPOPT swing-up solve. The recorded `summary.json` says which backend was used.
