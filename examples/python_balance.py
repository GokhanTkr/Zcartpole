"""Simulate one link in memory and plot cart and angle trajectories.

Install: python -m pip install -e '.[viz]'
Run: python examples/python_balance.py
This example applies ideal force directly, without actuator dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt

from zcartpole import ZNCartPole, simulate


model = ZNCartPole(m=[0.1], l=[0.4], M=1.0, u_max=80.0)
solution = simulate(
    model,
    controller=lambda t, state: model.lqr_control(state),
    th0=[0.05],
    t_max=3.0,
    dt=0.002,
)
angle = np.unwrap(np.arctan2(solution.y[3], solution.y[2]))

fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(solution.t, solution.y[0])
axes[0].set_ylabel('Cart position (m)')
axes[1].plot(solution.t, angle)
axes[1].set_ylabel('Link angle (rad)')
axes[1].set_xlabel('Time (s)')
fig.tight_layout()
plt.show()
