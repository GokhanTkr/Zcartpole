"""Closed-loop simulation with a bounded, lagging cart force actuator."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_continuous_are


def simulate_actuated(model, controller, th0, *, t_max=8.0, dt=0.002,
                      tau=0.0, command_delay=0.0, x0=0.0, v0=0.0, w0=None,
                      rail_limit=None, predict_delay=False,
                      prediction_model=None, prediction_tau=None):
    """Simulate the cart with ``tau * force_dot = command - force``.

    Commands are clipped to ``u_max`` and optionally held in a discrete
    dead-time queue. ``controller(t, state, force)`` observes the instantaneous
    force. For finite ``tau``, the mechanical RK4 step uses the exact mean
    force of the first-order response to the held command. An optional rail
    limit stops integration at the first violation; no impact is modeled.
    With ``predict_delay``, the controller receives a nominal model forecast
    over commands already waiting in the delay queue. Specify the predictor
    model and tau explicitly when they differ from the simulated plant.
    Return ``t``, ``y``, ``u`` (applied force), ``command`` and violation time.
    """
    if (not np.isfinite(t_max) or t_max <= 0 or not np.isfinite(dt) or dt <= 0
        or not np.isfinite(tau) or tau < 0 or not np.isfinite(command_delay)
        or command_delay < 0 or (rail_limit is not None and
                                  (not np.isfinite(rail_limit) or rail_limit <= 0))):
        raise ValueError('Invalid actuator or simulation parameters')
    steps = int(round(t_max/dt))
    lag = int(round(command_delay/dt))
    if abs(lag*dt-command_delay) > 1e-9:
        raise ValueError('command_delay must be an integer multiple of dt')
    prediction_model = model if prediction_model is None else prediction_model
    prediction_tau = tau if prediction_tau is None else prediction_tau
    if (prediction_model.n != model.n or not np.isfinite(prediction_tau)
            or prediction_tau < 0):
        raise ValueError('Invalid predictor model or time constant')
    th0 = np.asarray(th0, dtype=float)
    w0 = np.zeros(model.n) if w0 is None else np.asarray(w0, dtype=float)
    if th0.shape != (model.n,) or w0.shape != (model.n,) or not (
        np.isfinite(th0).all() and np.isfinite(w0).all()
        and np.isfinite(x0) and np.isfinite(v0)
    ):
        raise ValueError('Initial state must be finite, with n-vector angles and speeds')
    state = model.pack(x0, v0, np.exp(1j*th0), w0)
    force = 0.0
    queue = [0.0]*lag
    t_values, y_values, u_values, commands = [], [], [], []
    violation = None

    def advance(plant, current, actual, demand, time_constant):
        if time_constant > 0:
            fraction = -np.expm1(-dt/time_constant)
            mean = demand+(actual-demand)*(time_constant/dt)*fraction
            updated = actual+(demand-actual)*fraction
        else:
            mean = updated = demand
        return plant.step(current, mean, dt), updated

    for k in range(steps+1):
        t = k*dt
        if predict_delay and lag:
            forecast, forecast_force = state.copy(), force
            for pending in queue:
                forecast, forecast_force = advance(
                    prediction_model, forecast, forecast_force,
                    float(np.clip(pending, -prediction_model.u_max,
                                  prediction_model.u_max)), prediction_tau)
            observed_t, observed_state, observed_force = (
                t+command_delay, forecast, forecast_force)
        else:
            observed_t, observed_state, observed_force = t, state, force
        command = float(np.clip(controller(observed_t, observed_state,
                                           observed_force),
                                -model.u_max, model.u_max))
        queue.append(command)
        delayed = queue.pop(0)
        if tau == 0:
            force = delayed
        t_values.append(t)
        y_values.append(state.copy())
        u_values.append(force)
        commands.append(command)
        if rail_limit is not None and abs(state[0]) > rail_limit+1e-5:
            violation = t
            break
        if k < steps:
            state, force = advance(model, state, force, delayed, tau)
    return SimpleNamespace(t=np.asarray(t_values), y=np.asarray(y_values).T,
                           u=np.asarray(u_values), command=np.asarray(commands),
                           rail_violation_time=violation)


class ActuatorTVLQRTracker:
    """TVLQR around a force trajectory with first-order actuator state.

    The nominal force ``plan.U`` is converted to a command with
    ``u_command = force + tau * force_dot``. Feedback observes both mechanical
    state and actual force. This addresses first-order lag; pure dead time is
    not represented in the Riccati design.
    """

    def __init__(self, model, plan, tau, force_weight=0.1):
        if not plan.success or not np.isfinite(tau) or tau <= 0 or (
            not np.isfinite(force_weight) or force_weight <= 0
        ):
            raise ValueError('Need a successful plan and positive tau/force_weight')
        self.model, self.plan, self.tau = model, plan, tau
        tn, X, U = plan.t, plan.X, plan.U
        self.ref = CubicSpline(tn, X, axis=1)
        dim = 2+2*model.n
        AB = []
        for k in range(len(tn)):
            A, B = model.linearize(X[:, k], U[k])
            augmented_A = np.zeros((dim+1, dim+1))
            augmented_A[:dim, :dim] = A
            augmented_A[:dim, dim:] = B
            augmented_A[dim, dim] = -1/tau
            augmented_B = np.zeros((dim+1, 1))
            augmented_B[dim, 0] = 1/tau
            AB.append((augmented_A, augmented_B))
        R = model.R
        Ri = np.linalg.inv(R)
        Q = np.zeros((dim+1, dim+1))
        Q[:dim, :dim] = model.Q
        Q[dim, dim] = force_weight
        terminal = solve_continuous_are(AB[-1][0], AB[-1][1], Q, R)

        def dP(P, A, B):
            return -(A.T@P + P@A - P@B@Ri@B.T@P + Q)

        P = terminal
        gains = [None]*len(tn)
        gains[-1] = (Ri@AB[-1][1].T@P).ravel()
        sub = 10
        for k in range(len(tn)-1, 0, -1):
            h = -(tn[k]-tn[k-1])/sub
            A1, B1 = AB[k]
            A0, B0 = AB[k-1]
            for j in range(sub):
                a, mid, end = j/sub, (j+.5)/sub, (j+1)/sub
                Am, Bm = (1-mid)*A1+mid*A0, (1-mid)*B1+mid*B0
                k1 = dP(P, (1-a)*A1+a*A0, (1-a)*B1+a*B0)
                k2 = dP(P+h*k1/2, Am, Bm)
                k3 = dP(P+h*k2/2, Am, Bm)
                k4 = dP(P+h*k3, (1-end)*A1+end*A0,
                        (1-end)*B1+end*B0)
                P = P+h*(k1+2*k2+2*k3+k4)/6
            gains[k-1] = (Ri@AB[k-1][1].T@P).ravel()
        self.gains = np.asarray(gains)

    def control(self, t, state, force):
        model, plan = self.model, self.plan
        if t >= plan.t[-1]:
            error = model.error(state, model.top_state())
            command = -self.gains[-1]@np.r_[error, force]
        else:
            k = int(np.clip(np.searchsorted(plan.t, t, side='right')-1,
                            0, len(plan.t)-2))
            slope = (plan.U[k+1]-plan.U[k])/(plan.t[k+1]-plan.t[k])
            feedforward = np.interp(t, plan.t, plan.U) + self.tau*slope
            gain = np.array([np.interp(t, plan.t, self.gains[:, i])
                             for i in range(self.gains.shape[1])])
            error = model.error(state, self.ref(t))
            force_error = force-np.interp(t, plan.t, plan.U)
            command = feedforward-gain@np.r_[error, force_error]
        return float(np.clip(command, -model.u_max, model.u_max))


class ActuatorLQRController:
    """Upright local LQR with force as an additional actuator state."""

    def __init__(self, model, tau, force_weight=0.1):
        if not np.isfinite(tau) or tau <= 0 or (
            not np.isfinite(force_weight) or force_weight <= 0
        ):
            raise ValueError('tau and force_weight must be positive')
        self.model = model
        A, B = model.linearize(model.top_state(), 0)
        dim = A.shape[0]
        augmented_A = np.zeros((dim+1, dim+1))
        augmented_A[:dim, :dim] = A
        augmented_A[:dim, dim:] = B
        augmented_A[dim, dim] = -1/tau
        augmented_B = np.zeros((dim+1, 1))
        augmented_B[dim, 0] = 1/tau
        Q = np.zeros((dim+1, dim+1))
        Q[:dim, :dim] = model.Q
        Q[dim, dim] = force_weight
        P = solve_continuous_are(augmented_A, augmented_B, Q, model.R)
        self.gain = np.linalg.solve(model.R, augmented_B.T@P).ravel()

    def control(self, t, state, force):
        error = self.model.error(state, self.model.top_state())
        return float(np.clip(-self.gain@np.r_[error, force],
                             -self.model.u_max, self.model.u_max))
