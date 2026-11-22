"""Write NumPy gradients to JSON, so a Node test can compare the browser engine.

    python -m tools.make_engine_fixture
"""

import json
from pathlib import Path

import numpy as np

from pinn.data import measure
from pinn.physics import Oscillator
from pinn.scratch import ScratchMLP

if __name__ == "__main__":
    osc, net = Oscillator(), ScratchMLP(width=6, depth=3, seed=4)
    data, t_col = measure(osc), np.linspace(0, 1, 9)
    parts, grads = net.loss_and_grads(data.t, data.x, t_col, osc.m, osc.c, osc.k, weight=0.3)
    n = len(net.W)
    payload = {
        "W": [w.ravel().tolist() for w in net.W],
        "b": [b.tolist() for b in net.b],
        "data": {"t": data.t.tolist(), "x": data.x.tolist()},
        "t_col": t_col.tolist(),
        "parts": parts,
        "gW": [g.ravel().tolist() for g in grads[:n]],
        "gb": [g.tolist() for g in grads[n:]],
        "truth": [[t, float(osc.position(t)), float(osc.velocity(t))] for t in t_col],
    }
    path = Path("tests/engine_fixture.json")
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8", newline="\n")
    print("wrote", path)
