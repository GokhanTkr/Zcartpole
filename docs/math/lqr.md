# LQR and trajectory tracking

Near upright, the controller represents errors as

$$e=[\Delta x,\Delta\theta_1,\ldots,\Delta\theta_n,
\Delta v,\Delta\omega_1,\ldots,\Delta\omega_n]^\top.$$

The mechanical model is linearized about upright to give $\dot e\simeq A e+B\Delta F$. A continuous algebraic Riccati equation with positive diagonal $Q$ and scalar $R$ gives feedback $F=-Ke$ (clipped at the model force bound). JSON `lqr` fields set the state and force weights; see [configuration](../configuration.md). This design is local and does not lift a hanging pendulum by itself.

For a successful swing-up path, TVLQR linearizes along the time-varying nominal state and force, integrates a Riccati equation along the path, and requests $F_{\mathrm{nominal}}(t)-K(t)e(t)$. After the planned time, it uses upright feedback. The requested force then passes through the selected actuator and its command, delay, and force limits in the JSON runner. Large tracking error and saturation can make an otherwise valid nominal plan miss the catch.

For force lag or the DC model, the runner can use an augmented actuator-aware controller; `actuator_force_weight` tunes its extra force state. For a general transfer-function or state-space actuator, the runner uses the mechanical force controller and a local command inversion over `tracking_horizon_s`. This is an approximation, not an LQR synthesis for every actuator internal state. Retune and test against the full actuator simulation.
