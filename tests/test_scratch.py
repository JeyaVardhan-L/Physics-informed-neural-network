"""The hand-written gradients against autograd."""

import numpy as np
import torch

from pinn.data import measure
from pinn.derivatives import derivatives
from pinn.losses import data_loss, physics_loss
from pinn.network import MLP, column
from pinn.physics import Oscillator
from pinn.scratch import Adam, ScratchMLP, tanh_and_derivatives

OSC = Oscillator()
T_COL = np.linspace(0, 1, 25)


def twins():
    torch_net, scratch = MLP(width=12, seed=2), ScratchMLP(width=12)
    linears = [layer for layer in torch_net.layers if hasattr(layer, "weight")]
    scratch.W = [layer.weight.detach().numpy().copy() for layer in linears]
    scratch.b = [layer.bias.detach().numpy().copy() for layer in linears]
    return torch_net, linears, scratch


def test_tanh_derivatives():
    z, h = np.linspace(-2, 2, 50), 1e-5
    for order in range(3):
        numeric = (tanh_and_derivatives(z + h)[order] - tanh_and_derivatives(z - h)[order]) / (2 * h)
        np.testing.assert_allclose(tanh_and_derivatives(z)[order + 1], numeric, atol=1e-8)


def test_forward_matches_autograd():
    torch_net, _, scratch = twins()
    ours = scratch.forward(T_COL)[:3]
    for mine, theirs in zip(ours, derivatives(torch_net, column(T_COL))):
        np.testing.assert_allclose(mine, theirs.detach().numpy(), atol=1e-11)


def test_every_weight_gradient_matches_autograd():
    torch_net, linears, scratch = twins()
    data = measure(OSC)
    loss = data_loss(torch_net, column(data.t), column(data.x))
    loss = loss + 0.3 * physics_loss(torch_net, column(T_COL), OSC.m, OSC.c, OSC.k)
    loss.backward()
    theirs = [layer.weight.grad.numpy() for layer in linears]
    theirs += [layer.bias.grad.numpy() for layer in linears]
    _, ours = scratch.loss_and_grads(data.t, data.x, T_COL, OSC.m, OSC.c, OSC.k, weight=0.3)
    assert len(ours) == len(theirs) == 8
    for mine, other in zip(ours, theirs):
        np.testing.assert_allclose(mine, other, rtol=1e-9, atol=1e-12)


def test_adam_matches_torch_adam():
    start, grad = np.array([1.0, -2.0]), np.array([0.3, -0.1])
    mine = start.copy()
    ours = Adam([mine], lr=0.01)
    theirs = torch.tensor(start, requires_grad=True)
    optimiser = torch.optim.Adam([theirs], lr=0.01)
    for _ in range(5):
        ours.step([grad])
        theirs.grad = torch.tensor(grad)
        optimiser.step()
    np.testing.assert_allclose(mine, theirs.detach().numpy(), rtol=1e-9)
