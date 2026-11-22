import numpy as np

from pinn import classical
from pinn.data import future_grid, measure
from pinn.physics import Oscillator


def test_perfect_dots_give_a_perfect_future():
    osc = Oscillator()
    predict = classical.fit(osc, measure(osc, noise=0.0))
    np.testing.assert_allclose(predict(future_grid()), osc.position(future_grid()), atol=1e-12)


def test_predicted_error_matches_simulated_error():
    osc, t = Oscillator(), future_grid()
    errs = []
    for seed in range(2000):
        predict = classical.fit(osc, measure(osc, seed=seed))
        errs.append(np.mean((predict(t) - osc.position(t)) ** 2))
    expected = classical.expected_future_mse(osc, measure(osc), t)
    assert abs(np.mean(errs) / expected - 1) < 0.1
