import numpy as np
import pytest

from pinn.physics import Oscillator

T = np.linspace(0, 1, 400)


def central(f, t, h=1e-5):
    return (f(t + h) - f(t - h)) / (2 * h)


def test_starts_where_it_should():
    osc = Oscillator(x0=0.7, v0=-2.0)
    assert osc.position(0.0) == pytest.approx(0.7)
    assert osc.velocity(0.0) == pytest.approx(-2.0)


def test_velocity_is_the_slope_of_position():
    osc = Oscillator()
    np.testing.assert_allclose(osc.velocity(T), central(osc.position, T), atol=1e-6)


def test_closed_form_satisfies_the_equation():
    # acceleration() is defined from the ODE, so compare it with a numerical
    # derivative of velocity(). Agreement means the closed form is a solution.
    osc = Oscillator()
    np.testing.assert_allclose(osc.acceleration(T), central(osc.velocity, T), atol=1e-4)


def test_energy_never_rises():
    energy = Oscillator().energy(T)
    assert np.all(np.diff(energy) <= 1e-12)


def test_no_friction_keeps_energy_constant():
    energy = Oscillator(c=0.0).energy(T)
    np.testing.assert_allclose(energy, energy[0], rtol=1e-12)


def test_overdamped_is_refused():
    with pytest.raises(ValueError):
        Oscillator(c=40.0)
