"""The network: one number in (time), one number out (position).

Nothing in this file knows about springs. The plain net and the PINN are both
built here with the same seed, so they start with identical weights. The only
difference between them is the loss they are trained on.
"""

import torch
from torch import nn


class Sine(nn.Module):
    def forward(self, z):
        return torch.sin(z)


ACTIVATIONS = {"tanh": nn.Tanh, "sin": Sine, "softplus": nn.Softplus, "relu": nn.ReLU}


class MLP(nn.Module):
    def __init__(
        self,
        width=32,
        depth=3,
        activation="tanh",
        input_scale=1.0,
        t_min=0.0,
        t_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.t_min, self.t_max = t_min, t_max
        # Above 1, the first layer starts out with faster wiggles. See the
        # frequency experiment for when that is needed.
        self.input_scale = input_scale
        generator = torch.Generator().manual_seed(seed)

        layers, size_in = [], 1
        for _ in range(depth):
            layers += [self._linear(size_in, width, generator), ACTIVATIONS[activation]()]
            size_in = width
        layers.append(self._linear(size_in, 1, generator))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def _linear(size_in, size_out, generator):
        # float64 throughout. Second derivatives amplify rounding error, and
        # L-BFGS compares losses that differ in the seventh digit.
        # skip_init stops nn.Linear drawing its own default weights, which would
        # silently advance the global random state.
        layer = nn.utils.skip_init(nn.Linear, size_in, size_out, dtype=torch.float64)
        nn.init.xavier_uniform_(layer.weight, generator=generator)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, t):
        # Map [t_min, t_max] to [-1, 1], where tanh is neither flat nor linear.
        # This happens inside forward so autograd differentiates with respect
        # to real seconds and the chain rule is applied for us.
        s = 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1
        return self.layers(self.input_scale * s)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


def column(values):
    """Turn a list or array of N numbers into the (N, 1) tensor a net expects."""
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)
