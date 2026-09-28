# zcartpole documentation

Model a cart with serial pendulum links, plan a swing-up, and record a closed-loop simulation with a chosen actuator. All physical inputs use SI units. The number of links is the length of the `links` array.

## Choose a path

| Goal | Read |
|---|---|
| Install and run | [Getting started](getting-started.md) |
| Run code for n = 1, 2, or 3 | [Code examples](examples.md) |
| Set parameters and control weights | [Configuration](configuration.md), [Actuators](actuators.md), [Model and control workflow](engineering.md) |
| Read a result or diagnose a failure | [Outputs](results.md), [Scope](scope.md) |
| Call Python functions directly | [Python API](api.md) |
| Check equations and numerical methods | [Math: angles](math/representation.md), [dynamics](math/dynamics.md), [integration](math/integration.md), [LQR](math/lqr.md), [swing-up](math/swingup.md) |
| Build a package for PyPI | [Release guide](publishing.md) |

The [browser demo](demo/player.html) displays sample trajectories. Its parameters are not measured hardware data. The planner assumes ideal force; the subsequent simulation includes actuator dynamics and limits.
