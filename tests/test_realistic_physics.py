"""Independent kinematic and energy checks for rigid rods and hinge damping."""
import numpy as np
import pytest

from zcartpole import ZNCartPole


def _model(n, damped=False):
    return ZNCartPole(m=np.linspace(.08, .12, n), l=np.linspace(.3, .5, n),
                       rod_mass=np.linspace(.04, .07, n),
                       rod_com=np.linspace(.4, .6, n),
                       rod_inertia=np.linspace(.0002, .0004, n),
                       cart_damping=.2 if damped else 0,
                       joint_damping=np.linspace(.01, .03, n) if damped else 0,
                       u_max=80)


def _independent_mass_and_potential(model, angles):
    """Assemble translational mass from each body's Cartesian Jacobian."""
    n = model.n
    mass = np.diag([model.M] + [0.0]*n)
    potential = 0.0
    for j in range(n):
        for weight, fraction in ((model.m[j], 1.0),
                                 (model.rod_mass[j], model.rod_com[j])):
            J = np.zeros((2, n+1))
            J[0, 0] = 1.0
            for i in range(j):
                J[:, i+1] = (model.l[i]*np.cos(angles[i]),
                             -model.l[i]*np.sin(angles[i]))
            J[:, j+1] = (fraction*model.l[j]*np.cos(angles[j]),
                          -fraction*model.l[j]*np.sin(angles[j]))
            mass += weight * J.T @ J
            height = np.sum(model.l[:j]*np.cos(angles[:j]))
            height += fraction*model.l[j]*np.cos(angles[j])
            potential += model.g*weight*height
        mass[j+1, j+1] += model.rod_inertia[j]
    return mass, potential


def test_unphysical_link_parameters_rejected():
    for kwargs in ({'rod_mass': -.01}, {'rod_com': 1.2},
                   {'joint_damping': -.01},
                   {'m': (0.0,), 'rod_mass': 0.1,
                    'rod_com': 0.0, 'rod_inertia': 0.0}):
        params = dict(m=(.1,), l=(.4,))
        params.update(kwargs)
        with pytest.raises(ValueError):
            ZNCartPole(**params)


@pytest.mark.parametrize('n', (1, 2, 3))
def test_rigid_body_mass_matrix_and_energy(n):
    model = _model(n)
    rng = np.random.default_rng(n)
    for _ in range(5):
        th = rng.uniform(-2, 2, n)
        qdot = rng.uniform(-1, 1, n+1)
        zs = np.exp(1j*th)
        expected_mass, V = _independent_mass_and_potential(model, th)
        np.testing.assert_allclose(model.mass_matrix(zs), expected_mass, atol=1e-13)
        assert np.min(np.linalg.eigvalsh(expected_mass)) > 0
        state = model.pack(0, qdot[0], zs, qdot[1:])
        assert model.total_energy(state) == pytest.approx(
            .5*qdot@expected_mass@qdot + V, abs=1e-12)


def test_damping_obeys_power_balance():
    model = _model(3, damped=True)
    th, ws = np.array([.2, -.5, .9]), np.array([.7, -.4, .3])
    v, u = .6, 2.0
    state = model.pack(.1, v, np.exp(1j*th), ws)
    dt = 1e-6
    next_state = model.step(state, u, dt)
    measured = (model.total_energy(next_state)-model.total_energy(state))/dt
    relative = np.diff(np.r_[0.0, ws])
    expected = u*v - model.cart_damping*v*v - np.sum(model.joint_damping*relative**2)
    assert measured == pytest.approx(expected, abs=2e-5)


@pytest.mark.parametrize('n', (1, 2, 3))
def test_acceleration_matches_independent_euler_lagrange(n):
    model = _model(n, damped=True)
    rng = np.random.default_rng(100+n)
    th = rng.uniform(-1.5, 1.5, n)
    qdot = rng.uniform(-.8, .8, n+1)
    u = 1.3
    mass, _ = _independent_mass_and_potential(model, th)
    dM = np.zeros((n+1, n+1, n+1))
    dV = np.zeros(n+1)
    h = 1e-6
    for i in range(n):
        step = np.zeros(n)
        step[i] = h
        Mp, Vp = _independent_mass_and_potential(model, th+step)
        Mm, Vm = _independent_mass_and_potential(model, th-step)
        dM[i+1] = (Mp-Mm)/(2*h)
        dV[i+1] = (Vp-Vm)/(2*h)
    mdot = np.einsum('k,kij->ij', qdot, dM)
    partial_T = np.einsum('i,kij,j->k', qdot, dM, qdot)/2
    relative = np.diff(np.r_[0.0, qdot[1:]])
    Q = np.r_[u-model.cart_damping*qdot[0],
              -model.joint_damping*relative]
    Q[1:-1] += model.joint_damping[1:]*relative[1:]
    expected = np.linalg.solve(mass, Q + partial_T-dV-mdot@qdot)
    actual = model.accel(np.exp(1j*th), qdot[1:], u, qdot[0])
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_symbolic_accelerations_match_simulator():
    ca = pytest.importorskip('casadi')
    from zcartpole.symbolic import symbolic_accel

    model = _model(3, damped=True)
    q = ca.MX.sym('q', 8)  # x, v, angle1, omega1, ...
    u = ca.MX.sym('u')
    th = [q[2+2*i] for i in range(model.n)]
    ws = [q[3+2*i] for i in range(model.n)]
    f = ca.Function('accel', [q, u], [symbolic_accel(
        model, [ca.cos(a) for a in th], [ca.sin(a) for a in th], ws, u, q[1])])
    rng = np.random.default_rng(42)
    for _ in range(5):
        values = rng.uniform(-1, 1, 8)
        angles = values[2::2]
        omega = values[3::2]
        expected = model.accel(np.exp(1j*angles), omega, 1.2, values[1])
        np.testing.assert_allclose(np.asarray(f(values, 1.2)).ravel(), expected,
                                   atol=1e-12)
