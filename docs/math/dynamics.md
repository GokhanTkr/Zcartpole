# Mechanical dynamics

Let $q=[x,\theta_1,\ldots,\theta_n]$ and $\dot q=[v,\omega_1,\ldots,\omega_n]$. Each rigid link can have a point mass $m_i$ at its tip, rod mass $r_i$, rod center of mass at fraction $c_i$ of its length $l_i$, and rod inertia $I_i$ about that center. The cart has mass $M_c$; the optional DC drive adds reflected rotor mass to the effective cart mass.

Define the total mass beyond link $i$ as $B_i=\sum_{j=i+1}^n(m_j+r_j)$, and $A_i=B_i+m_i+r_i c_i$, $D_i=B_i+m_i+r_i c_i^2$. The symmetric mass matrix used by the mechanical model has entries

$$\mathcal M_{00}=M_c+\sum_i(m_i+r_i), \qquad
\mathcal M_{0i}=A_i l_i\cos\theta_i,$$

$$\mathcal M_{ii}=D_i l_i^2+I_i, \qquad
\mathcal M_{ij}=A_j l_i l_j\cos(\theta_i-\theta_j)\quad(i<j).$$

The model solves $\mathcal M(q)\ddot q=b(q,\dot q,F)$ each step. $F$ is the horizontal force actually applied to the cart. The right-hand side contains gravity, motion coupling, joint damping, and cart friction. With upward-positive potential convention, $V=g\sum_i A_i l_i\cos\theta_i$ and $T=\tfrac12\dot q^\top\mathcal M\dot q$. `model.total_energy(state)` returns $T+V$.

Cart friction is $-b_x v-F_c\tanh(v/v_s)$. Joint damping uses relative angular speed across each hinge (the cart hinge has zero base angular speed). This smooth friction approximation has no static sticking or breakaway. Rod defaults correspond to a centered uniform rod: $c_i=1/2$ and $I_i=r_i l_i^2/12$. When rod mass and damping are zero, the model reduces to point masses at link tips. See [configuration](../configuration.md) for physical units.
