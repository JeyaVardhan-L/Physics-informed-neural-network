"""The two terms of the loss.

data_loss     how far the curve is from the dots
physics_loss  how badly the curve breaks m x'' + c x' + k x = 0
"""

import torch

from .derivatives import derivatives


def data_loss(net, t, x):
    return torch.mean((net(t) - x) ** 2)


def residual(net, t, m, c, k):
    """What is left of the equation when the network is plugged in.

    Dividing by k turns force into position. x is about 1 and k is 225, so
    without this the physics term would start around 225^2 = 50,000 times
    larger than the data term and the dots would be ignored.
    """
    x, v, a = derivatives(net, t)
    return (m * a + c * v + k * x) / k


def physics_loss(net, t, m, c, k):
    return torch.mean(residual(net, t, m, c, k) ** 2)
