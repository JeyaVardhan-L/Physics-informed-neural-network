import numpy as np

from pinn import classical
from pinn.data import measure
from pinn.models import future_mse, train_pinn, train_plain
from pinn.physics import Oscillator

OSC = Oscillator()


def test_training_lowers_the_loss():
    result = train_plain(measure(OSC), adam_steps=300)
    assert result.history[-1]["total"] < 0.3 * result.history[0]["total"]


def test_same_seed_same_result():
    options = dict(adam_steps=50, lbfgs_steps=20)
    a = train_pinn(measure(OSC), OSC, **options).predict([0.5])
    b = train_pinn(measure(OSC), OSC, **options).predict([0.5])
    np.testing.assert_array_equal(a, b)


def test_scoring_the_truth_gives_zero():
    assert future_mse(OSC.position, OSC) == 0


def test_physics_beats_no_physics():
    # Shortened training, so the bar is loose. The full numbers live in results/.
    data = measure(OSC)
    plain = future_mse(train_plain(data, adam_steps=1000).predict, OSC)
    pinn = future_mse(train_pinn(data, OSC, adam_steps=1000, lbfgs_steps=1000).predict, OSC)
    assert pinn < plain / 50
