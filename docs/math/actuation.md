# Actuation equations

The general linear actuator accepts a command $C$ and cart speed $V$ and produces horizontal force $F$. Before limits,

$$F(s)=G_c(s)C(s)+G_v(s)V(s).$$

A proper transfer function is converted to a real state-space form:

$$\dot a=Aa+B_cC+B_vV,\qquad F_0=C_fa+D_cC+D_vV.$$

The issued command is clipped to its entered minimum and maximum, delayed by an integer number of simulation steps, and the calculated output is clipped to $\pm F_{\max}$. These nonlinear limits mean the final actuator behavior is not itself a linear transfer function. The simulation advances the actuator using a zero-order-hold state transition. The feedback controller first requests force, then estimates the command needed to reach it over `tracking_horizon_s` based on current actuator state and cart speed.

The optional `identified_first_order` type is $G_c(s)=K/(\tau s+1)$ plus a direct speed coefficient. The older `force_lag` type delays and filters a force request. The optional `dc` type uses an electrical winding and gearing approximation with voltage and current limits. None of these models is inferred from a motor frame label. For complete JSON fields and practical interpretation, see [actuators](../actuators.md).
