import numpy as np

from pinn.data import T_DATA, T_END, future_grid, measure
from pinn.physics import Oscillator


def test_same_seed_same_dots():
    a, b = measure(Oscillator(), seed=3), measure(Oscillator(), seed=3)
    np.testing.assert_array_equal(a.x, b.x)


def test_different_seed_different_dots():
    a, b = measure(Oscillator(), seed=3), measure(Oscillator(), seed=4)
    assert not np.array_equal(a.x, b.x)


def test_every_dot_is_inside_the_window():
    m = measure(Oscillator())
    assert len(m) == 10
    assert m.t.min() == 0.0 and m.t.max() == T_DATA


def test_scoring_times_never_touch_the_dots():
    grid = future_grid()
    assert grid.min() > T_DATA and grid.max() == T_END


def test_noise_has_the_advertised_size():
    osc = Oscillator()
    m = measure(osc, n=20_000, noise=0.05)
    assert abs(np.std(m.x - osc.position(m.t)) - 0.05) < 0.002
