"""Training: Adam to get into the right valley, L-BFGS to reach the bottom."""

import time

import torch


def fit(params, terms, weights, adam_steps=3000, lr=3e-3, lbfgs_steps=3000, log_every=100):
    """Minimise a weighted sum of loss terms.

    params   tensors to train
    terms    function returning {"data": tensor, "physics": tensor, ...}
    weights  {"data": 1.0, "physics": 0.1, ...}
    Returns a list of log rows, one per log_every steps.
    """
    params = list(params)
    history, start = [], time.perf_counter()

    def total(parts):
        return sum(weights[name] * value for name, value in parts.items())

    def log(step, stage, parts):
        history.append(
            {
                "step": step,
                "stage": stage,
                "seconds": round(time.perf_counter() - start, 3),
                "total": float(total(parts).detach()),
                **{name: float(value.detach()) for name, value in parts.items()},
            }
        )

    adam = torch.optim.Adam(params, lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(adam, max(adam_steps, 1), eta_min=lr / 30)
    for step in range(adam_steps):
        adam.zero_grad()
        parts = terms()
        total(parts).backward()
        if step % log_every == 0:
            log(step, "adam", parts)
        adam.step()
        schedule.step()

    if lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            params,
            max_iter=lbfgs_steps,
            max_eval=lbfgs_steps * 2,
            history_size=50,
            tolerance_grad=1e-12,
            tolerance_change=1e-16,
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure():
            # L-BFGS evaluates the loss several times per step while it searches
            # along a line, so it needs a function and not a single number.
            nonlocal calls
            lbfgs.zero_grad()
            parts = terms()
            loss = total(parts)
            loss.backward()
            if calls % log_every == 0:
                log(adam_steps + calls, "lbfgs", parts)
            calls += 1
            return loss

        lbfgs.step(closure)
        log(adam_steps + calls, "lbfgs", terms())
    else:
        log(adam_steps, "adam", terms())
    return history
