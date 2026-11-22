from pinn.data import measure
from pinn.inverse import discover
from pinn.physics import Oscillator


def test_the_guess_moves_towards_the_truth():
    osc = Oscillator()
    found = discover(measure(osc), adam_steps=1500, lbfgs_steps=500)
    assert abs(found.k - osc.k) < 0.2 * osc.k
    assert abs(found.trace[0][2] - 100) < 1e-9
