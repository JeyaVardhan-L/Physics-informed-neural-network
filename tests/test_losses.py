import torch

from pinn.losses import data_loss, physics_loss, residual
from pinn.network import MLP, column
from pinn.physics import Oscillator

OSC = Oscillator()
T = column(torch.linspace(0, 1, 100))


def true_motion(t):
    d, w = OSC.delta, OSC.omega_d
    return torch.exp(-d * t) * (torch.cos(w * t) + d / w * torch.sin(w * t))


def test_true_motion_has_no_residual():
    assert physics_loss(true_motion, T, OSC.m, OSC.c, OSC.k) < 1e-25


def test_wrong_stiffness_shows_up_in_the_residual():
    assert physics_loss(true_motion, T, OSC.m, OSC.c, 1.1 * OSC.k) > 1e-4


def test_flat_zero_also_satisfies_the_physics():
    # The equation alone cannot pick a solution. The dots have to.
    assert physics_loss(lambda t: 0 * t, T, OSC.m, OSC.c, OSC.k) == 0


def test_residual_is_scaled_to_order_one():
    # A constant curve x = 1 breaks the equation by exactly k x / k = 1.
    r = residual(lambda t: torch.ones_like(t) + 0 * t, T, OSC.m, OSC.c, OSC.k)
    assert torch.allclose(r, torch.ones_like(r))


def test_data_loss_is_a_mean_square():
    net = MLP(seed=0)
    x = net(T).detach() + 0.5
    assert torch.allclose(data_loss(net, T, x), torch.tensor(0.25, dtype=torch.float64))
