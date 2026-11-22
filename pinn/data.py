"""Ten noisy measurements, all taken early.

The whole experiment rests on one split. Measurements exist only in
[0, T_DATA]. The question is asked on (T_DATA, T_END].
"""

from dataclasses import dataclass

import numpy as np

T_DATA = 0.3
T_END = 1.0
N_POINTS = 10
NOISE = 0.01  # standard deviation, in the units of x. x(0) is 1, so this is 1%.


@dataclass(frozen=True)
class Measurements:
    t: np.ndarray
    x: np.ndarray
    noise: float
    seed: int

    def __len__(self):
        return len(self.t)


def measure(osc, n=N_POINTS, t_max=T_DATA, noise=NOISE, seed=0):
    """Sample the true motion at n evenly spaced times and add Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, t_max, n)
    x = osc.position(t) + noise * rng.standard_normal(n)
    return Measurements(t=t, x=x, noise=noise, seed=seed)


def future_grid(n=500, start=T_DATA, stop=T_END):
    """Times used for scoring. None of them is ever shown to a network."""
    return np.linspace(start, stop, n + 1)[1:]
