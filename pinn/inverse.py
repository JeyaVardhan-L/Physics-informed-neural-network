"""Run the equation backwards: the dots are known, c and k are not.

c and k become two more trainable numbers. They sit inside the residual, so the
same gradients that shape the curve also pull them towards values that make the
curve and the equation agree.

m is fixed at 1. Multiplying m, c and k by the same number leaves the equation
unchanged, so motion alone can never reveal all three.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from .data import T_END
from .derivatives import derivatives
from .losses import data_loss
from .models import Result
from .network import MLP, column
from .train import fit


@dataclass
class Discovery(Result):
    c: float = 0.0
    k: float = 0.0
    trace: list = field(default_factory=list, repr=False)


def discover(data, c_guess=1.0, k_guess=100.0, seed=0, n_collocation=60, weight=10.0, **fit_options):
    net = MLP(t_max=T_END, seed=seed)
    t, x = column(data.t), column(data.x)
    t_col = column(np.linspace(0, T_END, n_collocation))

    # Train the logarithms. c and k then stay positive whatever the optimiser
    # does, and a step means "a few percent" for both even though k is 75 times
    # larger than c.
    log_c = torch.tensor(np.log(c_guess), dtype=torch.float64, requires_grad=True)
    log_k = torch.tensor(np.log(k_guess), dtype=torch.float64, requires_grad=True)
    trace, calls = [], 0

    def terms():
        nonlocal calls
        c, k = log_c.exp(), log_k.exp()
        if calls % 50 == 0:
            trace.append([calls, float(c.detach()), float(k.detach())])
        calls += 1
        pos, vel, acc = derivatives(net, t_col)
        # Scaled by the fixed guess. Dividing by the trainable k would change the
        # loudness of the physics term while k is still being searched for.
        r = (acc + c * vel + k * pos) / k_guess
        return {"data": data_loss(net, t, x), "physics": torch.mean(r**2)}

    history = fit(
        [*net.parameters(), log_c, log_k], terms, {"data": 1.0, "physics": weight}, **fit_options
    )
    c, k = float(log_c.detach().exp()), float(log_k.detach().exp())
    return Discovery(net, history, c=c, k=k, trace=trace)
