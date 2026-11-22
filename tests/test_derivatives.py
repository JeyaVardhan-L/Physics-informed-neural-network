import torch

from pinn.derivatives import derivatives
from pinn.network import MLP, column

T = column(torch.linspace(0.05, 0.95, 40))


def test_known_function():
    x, v, a = derivatives(lambda t: t**3, T)
    assert torch.allclose(v, 3 * T**2)
    assert torch.allclose(a, 6 * T)


def test_network_against_finite_differences():
    net, h = MLP(seed=1), 1e-4
    x, v, a = derivatives(net, T)
    assert torch.allclose(v, (net(T + h) - net(T - h)) / (2 * h), atol=1e-6)
    assert torch.allclose(a, (net(T + h) - 2 * net(T) + net(T - h)) / h**2, atol=1e-5)


def test_second_derivative_still_reaches_the_weights():
    net = MLP(seed=1)
    derivatives(net, T)[2].pow(2).mean().backward()
    assert net.layers[0].weight.grad.abs().sum() > 0


def test_relu_has_no_second_derivative():
    net = MLP(activation="relu", seed=1)
    assert torch.all(derivatives(net, T)[2] == 0)
