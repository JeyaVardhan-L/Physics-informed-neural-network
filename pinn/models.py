"""The two contestants, built from the same parts."""

from dataclasses import dataclass, field

import numpy as np
import torch

from .data import T_DATA, T_END, future_grid
from .losses import data_loss, physics_loss
from .network import MLP, column
from .train import fit


@dataclass
class Result:
    net: MLP
    history: list = field(repr=False)

    def predict(self, t):
        with torch.no_grad():
            return self.net(column(t)).numpy().ravel()


def train_plain(data, seed=0, width=32, depth=3, lbfgs_steps=0, **fit_options):
    """The dots and nothing else.

    L-BFGS is off by default because it hurts here. It drives the data loss
    far below the noise level, so it fits the noise, and the future gets worse.
    """
    net = MLP(width=width, depth=depth, t_max=T_END, seed=seed)
    t, x = column(data.t), column(data.x)
    history = fit(
        net.parameters(),
        lambda: {"data": data_loss(net, t, x)},
        {"data": 1.0},
        lbfgs_steps=lbfgs_steps,
        **fit_options,
    )
    return Result(net, history)


def train_pinn(
    data,
    osc,
    seed=0,
    width=32,
    depth=3,
    activation="tanh",
    input_scale=1.0,
    n_collocation=60,
    t_physics=(0.0, T_END),
    weight=10.0,
    **fit_options,
):
    """The dots, plus the equation at n_collocation times that have no dots.

    osc is used for m, c and k only. Its x0 and v0, and so the true curve,
    never reach the loss.
    """
    net = MLP(
        width=width,
        depth=depth,
        activation=activation,
        input_scale=input_scale,
        t_max=T_END,
        seed=seed,
    )
    t, x = column(data.t), column(data.x)
    t_col = column(np.linspace(*t_physics, n_collocation))
    history = fit(
        net.parameters(),
        lambda: {
            "data": data_loss(net, t, x),
            "physics": physics_loss(net, t_col, osc.m, osc.c, osc.k),
        },
        {"data": 1.0, "physics": weight},
        **fit_options,
    )
    return Result(net, history)


def future_mse(predict, osc):
    """Mean squared error on (T_DATA, T_END], against the noiseless truth."""
    t = future_grid(start=T_DATA, stop=T_END)
    return float(np.mean((predict(t) - osc.position(t)) ** 2))
