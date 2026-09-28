"""A rotation update should keep solved z nodes on the unit circle."""
import numpy as np
import pytest

from zcartpole import ZNCartPole
from zcartpole.opt import _angle_quadrature, optimize_swingup, unit_circle_error


def test_angle_quadrature_for_quadratic_angular_velocity():
    # omega(t) = 2 + 3*t + 4*t**2 on [0, h].
    h = 0.2
    omega = lambda t: 2 + 3*t + 4*t*t
    half, full = _angle_quadrature(omega(0), omega(h/2), omega(h), h)
    primitive = lambda t: 2*t + 1.5*t*t + (4/3)*t**3
    assert half == pytest.approx(primitive(h/2))
    assert full == pytest.approx(primitive(h))


def test_lie_swingup_solution_stays_on_circle():
    pytest.importorskip('casadi')
    model = ZNCartPole(m=(0.1,), l=(0.4,), u_max=80)
    ok, _, X, _, J = optimize_swingup(
        model, T=3, N=12, x_max=0.5, th0=(np.pi + .05,),
        w_penalty=0.0, integrator='lie')
    assert ok and np.isfinite(J)
    assert unit_circle_error(model, X) < 1e-5


def test_integrator_name_is_checked():
    pytest.importorskip('casadi')
    model = ZNCartPole(m=(0.1,), l=(0.4,))
    with pytest.raises(ValueError, match='integrator'):
        optimize_swingup(model, integrator='unknown')
