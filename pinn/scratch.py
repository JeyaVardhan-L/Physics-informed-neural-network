"""The same PINN with no autograd and no PyTorch, only NumPy.

Two ideas, one for each direction.

Forward   Carry three numbers through every layer: a value, its first time
          derivative and its second. (a, a1, a2) instead of only a.

Backward  Walk the layers in reverse and hand each of those three numbers its
          share of the blame for the loss. The output of the walk is the
          gradient for every weight.

The browser figure on the site runs a line-by-line port of this file, and the
tests compare its gradients with autograd's.
"""

import numpy as np


def tanh_and_derivatives(z):
    """tanh(z) and its first three derivatives."""
    h = np.tanh(z)
    h1 = 1 - h**2
    h2 = -2 * h * h1
    h3 = -2 * h1 * (1 - 3 * h**2)
    return h, h1, h2, h3


class ScratchMLP:
    def __init__(self, width=32, depth=3, t_max=1.0, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [1] + [width] * depth + [1]
        self.t_max = t_max
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6 / (n_in + n_out))  # Xavier uniform
            self.W.append(rng.uniform(-bound, bound, (n_out, n_in)))
            self.b.append(np.zeros(n_out))

    def forward(self, t):
        """Return x, x', x'' at times t, and what backward() will need."""
        t = np.asarray(t, dtype=float).reshape(-1, 1)
        # s = 2 t / t_max - 1, so ds/dt = 2 / t_max and d2s/dt2 = 0.
        a = 2 * t / self.t_max - 1
        a1 = np.full_like(t, 2 / self.t_max)
        a2 = np.zeros_like(t)

        cache, last = [], len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            # A linear layer is the same matrix applied to all three. The bias
            # is a constant, so it disappears from both derivatives.
            z, z1, z2 = a @ W.T + b, a1 @ W.T, a2 @ W.T
            cache.append((a, a1, a2, z, z1, z2))
            if i == last:
                a, a1, a2 = z, z1, z2
            else:
                h, h1, h2, _ = tanh_and_derivatives(z)
                a = h
                a1 = h1 * z1  # chain rule
                a2 = h2 * z1**2 + h1 * z2  # chain rule again, then product rule
        return a, a1, a2, cache

    def backward(self, cache, g, g1, g2):
        """Gradients of a loss, given its derivative by x, x' and x''.

        g is dL/dx, g1 is dL/dx', g2 is dL/dx''. Each has shape (N, 1).
        """
        dW, db = [None] * len(self.W), [None] * len(self.b)
        last = len(self.W) - 1
        for i in range(last, -1, -1):
            a, a1, a2, z, z1, z2 = cache[i]
            if i != last:
                _, h1, h2, h3 = tanh_and_derivatives(z)
                # z appears in all three outputs, z1 in two of them, z2 in one.
                gz = g * h1 + g1 * h2 * z1 + g2 * (h3 * z1**2 + h2 * z2)
                gz1 = g1 * h1 + g2 * 2 * h2 * z1
                gz2 = g2 * h1
            else:
                gz, gz1, gz2 = g, g1, g2
            dW[i] = gz.T @ a + gz1.T @ a1 + gz2.T @ a2
            db[i] = gz.sum(axis=0)
            W = self.W[i]
            g, g1, g2 = gz @ W, gz1 @ W, gz2 @ W
        return dW, db

    def loss_and_grads(self, t_data, x_data, t_col, m, c, k, weight=10.0):
        x_data = np.asarray(x_data, dtype=float).reshape(-1, 1)

        # Data term: mean((x - dots)^2). Only x matters, so g1 = g2 = 0.
        x, _, _, cache = self.forward(t_data)
        err = x - x_data
        zero = np.zeros_like(err)
        dW_d, db_d = self.backward(cache, 2 * err / len(err), zero, zero)

        # Physics term: mean(r^2) with r = (m x'' + c x' + k x) / k.
        x, v, a, cache = self.forward(t_col)
        r = (m * a + c * v + k * x) / k
        g = weight * 2 * r / len(r)
        dW_p, db_p = self.backward(cache, g, g * c / k, g * m / k)

        parts = {"data": float(np.mean(err**2)), "physics": float(np.mean(r**2))}
        grads = [p + q for p, q in zip(dW_d + db_d, dW_p + db_p)]
        return parts, grads

    def parameters(self):
        return self.W + self.b


class Adam:
    def __init__(self, params, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr, self.beta1, self.beta2, self.eps = params, lr, beta1, beta2, eps
        self.mean = [np.zeros_like(p) for p in params]
        self.var = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, mean, var in zip(self.params, grads, self.mean, self.var):
            mean += (1 - self.beta1) * (g - mean)
            var += (1 - self.beta2) * (g**2 - var)
            mean_hat = mean / (1 - self.beta1**self.t)
            var_hat = var / (1 - self.beta2**self.t)
            p -= self.lr * mean_hat / (np.sqrt(var_hat) + self.eps)  # in place


def train(net, data, osc, steps=15000, lr=3e-3, weight=10.0, n_collocation=60, t_end=1.0):
    adam, t_col, history = Adam(net.parameters(), lr), np.linspace(0, t_end, n_collocation), []
    for step in range(steps):
        adam.lr = lr / 30 + (lr - lr / 30) * 0.5 * (1 + np.cos(np.pi * step / steps))
        parts, grads = net.loss_and_grads(data.t, data.x, t_col, osc.m, osc.c, osc.k, weight)
        adam.step(grads)
        if step % 500 == 0:
            history.append({"step": step, **parts})
    return history
