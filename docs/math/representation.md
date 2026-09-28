# Angles and state

The cart moves horizontally. Each link angle $\theta_i$ is measured **from upward vertical**, with positive angle turning the tip toward positive cart x. Angles are absolute, even though each link is hinged to the previous one. Thus 0 is upright and $\pi$ is hanging. Angles differing by $2\pi k$ have the same pose.

The mechanical simulator stores $z_i=e^{\mathrm{i}\theta_i}=(\cos\theta_i,\sin\theta_i)$ and angular speed $\omega_i$. Its real array layout is

$$s=[x,v,\Re z_1,\Im z_1,\omega_1,\ldots,\Re z_n,\Im z_n,\omega_n].$$

The tip position of link $i$ is $x_i=x+\sum_{j=1}^i l_j\sin\theta_j$ and $y_i=\sum_{j=1}^i l_j\cos\theta_j$. The controller computes local angle error with the principal angle of $z_i\overline{z_{i,\mathrm{ref}}}$. That error has a branch at $\pm\pi$; feedback is local around a reference path.

The swing-up optimizer instead uses unwrapped $[x,v,\theta_1,\omega_1,\ldots]$. This preserves whole turns in its candidate paths. It converts successful paths to the unit-complex simulator layout for TVLQR. `plan.phase_X` retains the unwrapped angles and `plan.windings` records their final complete turns.
