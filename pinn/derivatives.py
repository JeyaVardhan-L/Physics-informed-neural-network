"""x, x' and x'' of a network with respect to its *input*.

Ordinary training differentiates the loss with respect to the weights. A PINN
needs a second kind of derivative first: the output with respect to time.
"""

import torch


def derivatives(net, t):
    """Return (x, v, a) at the times in t, all still attached to the graph."""
    t = t.detach().clone().requires_grad_(True)
    x = net(t)
    v = _slope(x, t)
    a = _slope(v, t)
    return x, v, a


def _slope(y, t):
    # A straight line has a constant slope, and a constant has no graph to
    # differentiate. Its derivative is zero.
    if not y.requires_grad:
        return torch.zeros_like(t)
    # Each y[i] depends only on its own t[i], so summing y before differentiating
    # (which is what grad_outputs of ones does) still gives dy[i]/dt[i].
    #
    # create_graph=True keeps the derivative differentiable. Without it the loss
    # could not send gradients back through x'' to the weights.
    #
    # materialize_grads=True returns zeros where y does not depend on t at all.
    # That really happens: the second derivative of a ReLU net is zero.
    (dy,) = torch.autograd.grad(
        y, t, grad_outputs=torch.ones_like(y), create_graph=True, materialize_grads=True
    )
    return dy
