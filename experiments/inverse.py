"""Recover c and k from the ten dots.

    python -m experiments.inverse
"""

from pinn.data import measure
from pinn.inverse import discover
from pinn.models import future_mse
from pinn.physics import Oscillator

from .common import FEW_SEEDS, curve, run, save

OSC = Oscillator()


def job(noise, seed):
    result = discover(measure(OSC, seed=seed, noise=noise), seed=seed)
    row = {
        "noise": noise,
        "seed": seed,
        "c": result.c,
        "k": result.k,
        "mse": future_mse(result.predict, OSC),
    }
    if seed == 0:
        row["curve"] = curve(result.predict)
        row["trace"] = [[n, round(c, 4), round(k, 3)] for n, c, k in result.trace]
    return row


if __name__ == "__main__":
    save("inverse", run(job, [(noise, s) for noise in (0.0, 0.01, 0.03) for s in FEW_SEEDS]))
