"""The physical system: a mass on a spring with friction.

    m x''(t) + c x'(t) + k x(t) = 0

m is the mass (kg), c the damping (kg/s), k the stiffness (N/m). Everything
else in the project trusts this file, so it has the most direct tests.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Oscillator:
    m: float = 1.0
    c: float = 3.0
    k: float = 225.0
    x0: float = 1.0  # position at t = 0
    v0: float = 0.0  # velocity at t = 0

    def __post_init__(self):
        if self.m <= 0 or self.k <= 0 or self.c < 0:
            raise ValueError("need m > 0, k > 0 and c >= 0")
        if self.zeta >= 1:
            raise ValueError(
                f"damping ratio {self.zeta:.2f} >= 1: the closed form here "
                "only covers the underdamped case"
            )

    @property
    def omega0(self):
        """Natural frequency, the rate it would swing at without friction."""
        return np.sqrt(self.k / self.m)

    @property
    def delta(self):
        """Decay rate of the envelope exp(-delta t)."""
        return self.c / (2 * self.m)

    @property
    def zeta(self):
        """Damping ratio. Below 1 the mass oscillates, at 1 or above it creeps."""
        return self.c / (2 * np.sqrt(self.m * self.k))

    @property
    def omega_d(self):
        """The frequency it really swings at, slightly below omega0."""
        return np.sqrt(self.omega0**2 - self.delta**2)

    def position(self, t):
        t = np.asarray(t, dtype=float)
        a, b = self._coefficients()
        w = self.omega_d
        return np.exp(-self.delta * t) * (a * np.cos(w * t) + b * np.sin(w * t))

    def velocity(self, t):
        t = np.asarray(t, dtype=float)
        a, b = self._coefficients()
        w, d = self.omega_d, self.delta
        return np.exp(-d * t) * (
            (b * w - d * a) * np.cos(w * t) - (a * w + d * b) * np.sin(w * t)
        )

    def acceleration(self, t):
        # Newton's second law. The equation itself says what x'' must be.
        return -(self.c * self.velocity(t) + self.k * self.position(t)) / self.m

    def energy(self, t):
        return 0.5 * self.m * self.velocity(t) ** 2 + 0.5 * self.k * self.position(t) ** 2

    def _coefficients(self):
        # a fixes x(0) = x0. b fixes x'(0) = v0, and is not zero even when
        # v0 = 0, because the envelope is already falling at t = 0.
        return self.x0, (self.v0 + self.delta * self.x0) / self.omega_d
