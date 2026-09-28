# Numerical comparisons

The scripts under `experiments/` compare coordinate and integration
choices on selected cart-pendulum cases. They are reproducible examples,
not benchmarks of every physical system.

- **E1** compares angle RK4, ordinary Cartesian $(c,s)$ RK4, and the
  exponential-map update. See `experiments/e1_integration.py`.
- **E2-A** compares angle and complex-coordinate swing-up programs from
  matched seeds. E2-B uses a fixed total solve budget. See
  `experiments/e2_fair_benchmark.py`.
- **E3** tries a free-winding angle formulation without a prescribed final
  turn count. See `experiments/e3_free_winding_phase.py`.

Saved results under `experiments/results/` record the selected inputs
and outcomes. NLP solvers can settle in different local solutions. A
successful nominal path still needs closed-loop validation with the chosen
actuator, limits, and delay.
