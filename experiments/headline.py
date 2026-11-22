"""Plain net against PINN, ten seeds, against two yardsticks.

    python -m experiments.headline
"""

from pinn import classical
from pinn.data import future_grid, measure
from pinn.models import future_mse, train_pinn, train_plain
from pinn.physics import Oscillator

from .common import CURVE_T, SEEDS, curve, median, run, save

OSC = Oscillator()


def job(method, seed):
    data = measure(OSC, seed=seed)
    history = []
    if method == "plain":
        result = train_plain(data, seed=seed)
        predict, history = result.predict, result.history
    elif method == "plain_lbfgs":
        result = train_plain(data, seed=seed, lbfgs_steps=3000)
        predict, history = result.predict, result.history
    elif method == "pinn":
        result = train_pinn(data, OSC, seed=seed)
        predict, history = result.predict, result.history
    elif method == "classical":
        predict = classical.fit(OSC, data)
    elif method == "zero":
        predict = lambda t: 0 * t  # noqa: E731  (a net that says "it has stopped")
    row = {"method": method, "seed": seed, "mse": future_mse(predict, OSC)}
    if history:
        row["data_loss"] = history[-1]["data"]
        row["seconds"] = history[-1]["seconds"]
    if seed < 3:
        row["curve"] = curve(predict)
        row["history"] = history
        row["dots"] = {"t": data.t.tolist(), "x": data.x.tolist()}
    return row


if __name__ == "__main__":
    methods = ["plain", "plain_lbfgs", "pinn", "classical", "zero"]
    rows = run(job, [(m, s) for m in methods for s in SEEDS])
    floor = classical.expected_future_mse(OSC, measure(OSC), future_grid())
    summary = {m: median(rows, method=m) for m in methods}
    by_seed = {m: {r["seed"]: r["mse"] for r in rows if r["method"] == m} for m in methods}
    ratios = sorted(by_seed["plain"][s] / by_seed["pinn"][s] for s in SEEDS)
    save(
        "headline",
        {
            "median_mse": summary,
            "noise_floor": floor,
            "ratio_of_medians": summary["plain"] / summary["pinn"],
            "ratio_per_seed": ratios,
            "t": CURVE_T.round(4).tolist(),
            "truth": curve(OSC.position),
            "rows": rows,
        },
    )
    print(summary, "floor", floor, "ratio", summary["plain"] / summary["pinn"])
