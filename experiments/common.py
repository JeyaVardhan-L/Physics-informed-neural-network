"""Shared plumbing: run jobs in parallel, save rows as JSON."""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pinn.data import T_END

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Seeds 100, 101 and 102 were used to choose weight = 10 and the optimiser
# settings. The seeds below were first run after those choices were fixed.
SEEDS = list(range(10))
FEW_SEEDS = [0, 1, 2]

CURVE_T = np.linspace(0, T_END, 201)


def curve(predict):
    return [round(float(x), 5) for x in predict(CURVE_T)]


def run(job, arguments, workers=None):
    """Call job(*args) for every args tuple, one process each, in order."""
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(workers, initializer=_single_thread) as pool:
        futures = [pool.submit(job, *args) for args in arguments]
        rows = []
        for future in futures:
            rows.append(future.result())
            shown = {k: v for k, v in rows[-1].items() if not isinstance(v, (list, dict))}
            print(shown, flush=True)
    return rows


def _single_thread():
    # The tensors are tiny. Several threads per process only fight each other.
    import torch

    torch.set_num_threads(1)


def save(name, payload):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", path.name)


def median(rows, key="mse", **match):
    values = [r[key] for r in rows if all(r[k] == v for k, v in match.items())]
    return float(np.median(values))
