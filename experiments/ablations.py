"""Change one thing at a time and measure what it was doing.

    python -m experiments.ablations             all of them
    python -m experiments.ablations coverage    just one
"""

import sys

from pinn import classical
from pinn.data import measure
from pinn.models import dense_residual, future_mse, train_pinn
from pinn.physics import Oscillator

from .common import FEW_SEEDS, curve, run, save

OSC = Oscillator()


def job(setting, seed, options, noise, believed, osc=OSC):
    """believed holds the m, c, k the loss is told. Usually the true ones."""
    data = measure(osc, seed=seed, noise=noise)
    result = train_pinn(data, believed, seed=seed, **options)
    last = result.history[-1]
    row = {
        "setting": setting,
        "seed": seed,
        "mse": future_mse(result.predict, osc),
        "classical": future_mse(classical.fit(osc, data), osc),
        "data_loss": last["data"],
        "physics_loss": last["physics"],
        "dense_residual": dense_residual(result.net, believed),
        "seconds": last["seconds"],
    }
    if seed == 0:
        row["curve"] = curve(result.predict)
    return row


def variants():
    adam = lambda n: dict(adam_steps=n, lbfgs_steps=0)  # noqa: E731
    return {
        # How loud is the equation compared with the dots?
        "weight": [(w, dict(weight=w), 0.01, OSC) for w in (0.001, 0.1, 1.0, 10.0, 1000.0)],
        # Where is the equation enforced? Always 60 points, spread up to t.
        "coverage": [(t, dict(t_physics=(0.0, t)), 0.01, OSC) for t in (0.3, 0.5, 0.75, 1.0)],
        # At how many times is it enforced?
        "collocation": [(n, dict(n_collocation=n), 0.01, OSC) for n in (4, 6, 8, 12, 20, 60)],
        # Which optimiser does the work?
        "optimiser": [
            ("adam 3000", adam(3000), 0.01, OSC),
            ("adam 6000", adam(6000), 0.01, OSC),
            ("adam 3000 + lbfgs", {}, 0.01, OSC),
            ("lbfgs only", dict(adam_steps=0), 0.01, OSC),
        ],
        # Does the activation need a second derivative?
        "activation": [(a, dict(activation=a), 0.01, OSC) for a in ("tanh", "sin", "softplus", "relu")],
        # How far does it degrade as the dots get worse?
        "noise": [(s, {}, s, OSC) for s in (0.0, 0.003, 0.01, 0.03, 0.1)],
        # What if the equation is wrong?
        "wrong_physics": [
            ("right", {}, 0.01, OSC),
            ("k 5% high", {}, 0.01, Oscillator(k=1.05 * OSC.k)),
            ("k 20% high", {}, 0.01, Oscillator(k=1.2 * OSC.k)),
            ("no friction", {}, 0.01, Oscillator(c=0.0)),
        ],
    }


if __name__ == "__main__":
    chosen = sys.argv[1:] or list(variants())
    for name in chosen:
        print("==", name)
        jobs = [
            (setting, seed, options, noise, believed)
            for setting, options, noise, believed in variants()[name]
            for seed in FEW_SEEDS
        ]
        save(f"ablation_{name}", run(job, jobs))
