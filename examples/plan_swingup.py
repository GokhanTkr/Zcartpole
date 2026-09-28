"""Plan, validate and animate swing-up. Install: pip install -e '.[opt,viz]'"""
import numpy as np

from zcartpole import ZNCartPole, plan_swingup, save_animation


model = ZNCartPole(m=(0.1, 0.1, 0.1), l=(0.4, 0.4, 0.4), u_max=80.0)
initial = (np.pi + 0.05, np.pi + 0.03, np.pi + 0.02)
plan = plan_swingup(model, T=6, N=100, x_max=0.5, th0=initial)
for attempt in plan.attempts:
    print(attempt)
if not plan.success:
    raise SystemExit(plan.message)

print(f"selected seed={plan.seed_k}, windings={plan.windings}, J={plan.J:.6g}")
sol = plan.simulate_catch(model, t_extra=3, dt=0.002)
print(f"last cart x={sol.y[0, -1]:.6g}, last link cos={sol.y[2::3, -1]}")
print(plan.evaluate_catch(model, sol))
print(f"animation: {save_animation(model, sol, 'swingup.gif', fps=15, speed=1.5, x_limit=plan.x_max)}")
