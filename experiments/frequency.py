"""A faster spring, where the training loss stops being trustworthy.

    python -m experiments.frequency
"""

from pinn.physics import Oscillator

from .ablations import job
from .common import FEW_SEEDS, curve, run, save

SPRINGS = {15: Oscillator(k=225.0), 30: Oscillator(k=900.0), 60: Oscillator(k=3600.0)}

VARIANTS = [
    ("omega 15, 60 points", 15, {}),
    ("omega 30, 60 points", 30, {}),
    ("omega 60, 60 points", 60, {}),
    ("omega 60, 240 points", 60, dict(n_collocation=240)),
    ("omega 60, 240 points, sin x4", 60, dict(n_collocation=240, activation="sin", input_scale=4.0)),
    ("omega 60, 60 points, sin x4", 60, dict(activation="sin", input_scale=4.0)),
]

if __name__ == "__main__":
    jobs = [
        (name, seed, options, 0.01, SPRINGS[omega], SPRINGS[omega])
        for name, omega, options in VARIANTS
        for seed in FEW_SEEDS
    ]
    rows = run(job, jobs)
    truth = {str(w): curve(osc.position) for w, osc in SPRINGS.items()}
    save("frequency", {"truth": truth, "rows": rows})
