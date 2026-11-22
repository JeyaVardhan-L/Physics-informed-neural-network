"""What a physicist would do without any network.

If m, c and k are known, every solution of the equation has the form

    x(t) = a * exp(-delta t) cos(wd t) + b * exp(-delta t) sin(wd t)

Only a and b are unknown and x is linear in both, so fitting ten dots is an
ordinary 2-parameter least-squares problem. This is the fair yardstick for the
PINN: it uses the same dots and the same equation, and it is optimal for
Gaussian noise.
"""

import numpy as np


def basis(osc, t):
    t = np.asarray(t, dtype=float)
    env = np.exp(-osc.delta * t)
    return np.stack([env * np.cos(osc.omega_d * t), env * np.sin(osc.omega_d * t)], axis=1)


def fit(osc, data):
    """Return a function t -> x(t) with the two constants fitted to the dots."""
    coef, *_ = np.linalg.lstsq(basis(osc, data.t), data.x, rcond=None)
    return lambda t: basis(osc, t) @ coef


def expected_future_mse(osc, data, t_future):
    """Average error of that fit, over all the noise it could have seen.

    Least squares turns noise of variance s^2 into coefficient covariance
    s^2 (B'B)^-1. Push that through the basis at the future times. No unbiased
    linear method using these dots can do better (Gauss-Markov).
    """
    B, F = basis(osc, data.t), basis(osc, t_future)
    cov = data.noise**2 * np.linalg.inv(B.T @ B)
    return float(np.mean(np.sum((F @ cov) * F, axis=1)))
