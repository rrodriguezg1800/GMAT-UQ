# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Self-contained reference dynamics, integrated with SciPy.

These models need no GMAT installation.  They exist so that the engines can be
tested and demonstrated on any machine, and they give the validation suite
problems with known properties (conserved energy, an analytic Jacobi constant).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .base import Dynamics, PropagationError

__all__ = ["OdeDynamics", "TwoBody", "TwoBodyJ2", "CR3BP", "EARTH_MU", "EARTH_RADIUS", "EARTH_J2"]

EARTH_MU = 398600.4415  # km^3/s^2 (GMAT default for Earth)
EARTH_RADIUS = 6378.1363  # km
EARTH_J2 = 1.0826269e-3


@dataclass(frozen=True)
class OdeDynamics(Dynamics):
    """Base class for dynamics given as a first-order ODE with an analytic Jacobian.

    The integrator tolerances are keyword-only, so that the first positional
    argument of every model is its physical parameter (``TwoBody(mu)``).
    """

    rtol: float = field(default=1e-12, kw_only=True)
    atol_position: float = field(default=1e-9, kw_only=True)
    atol_velocity: float = field(default=1e-12, kw_only=True)

    def __post_init__(self) -> None:
        if not 0.0 < self.rtol < 1e-3:
            raise ValueError("rtol must lie between 0 and 1e-3")
        if not (self.atol_position > 0.0 and self.atol_velocity > 0.0):
            raise ValueError("absolute tolerances must be positive")

    def rhs(self, t: float, state: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def jacobian(self, t: float, state: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _integrate(self, function, initial, t0, t1, atol):
        if t1 == t0:
            return np.array(initial, dtype=float)
        solution = solve_ivp(function, (t0, t1), initial, method="DOP853", rtol=self.rtol, atol=atol)
        if not solution.success:
            raise PropagationError(f"integration from {t0} to {t1} failed: {solution.message}")
        return solution.y[:, -1]

    def _state_atol(self) -> np.ndarray:
        return np.array([self.atol_position] * 3 + [self.atol_velocity] * 3)

    def propagate(self, state, t0: float, t1: float) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(6)
        return self._integrate(self.rhs, state, float(t0), float(t1), self._state_atol())

    def propagate_stm(self, state, t0: float, t1: float):
        state = np.asarray(state, dtype=float).reshape(6)

        def augmented(t, y):
            derivative = np.empty(42)
            derivative[:6] = self.rhs(t, y[:6])
            derivative[6:] = (self.jacobian(t, y[:6]) @ y[6:].reshape(6, 6)).ravel()
            return derivative

        initial = np.concatenate([state, np.eye(6).ravel()])
        # The STM is integrated to a relative tolerance only; its elements span many
        # orders of magnitude, so a single absolute tolerance would not be meaningful.
        atol = np.concatenate([self._state_atol(), np.full(36, 1e-14)])
        final = self._integrate(augmented, initial, float(t0), float(t1), atol)
        return final[:6], final[6:].reshape(6, 6)


@dataclass(frozen=True)
class TwoBody(OdeDynamics):
    """Point-mass gravity of a single body."""

    mu: float = EARTH_MU

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.mu > 0.0:
            raise ValueError("mu must be positive")

    def rhs(self, t, state):
        position = state[:3]
        r3 = np.dot(position, position) ** 1.5
        return np.concatenate([state[3:], -self.mu * position / r3])

    def _gravity_gradient(self, position):
        r2 = float(np.dot(position, position))
        r = np.sqrt(r2)
        return self.mu * (3.0 * np.outer(position, position) / r**5 - np.eye(3) / r**3)

    def jacobian(self, t, state):
        matrix = np.zeros((6, 6))
        matrix[:3, 3:] = np.eye(3)
        matrix[3:, :3] = self._gravity_gradient(state[:3])
        return matrix

    def describe(self) -> dict:
        return {"type": "TwoBody", "mu": self.mu}


@dataclass(frozen=True)
class TwoBodyJ2(TwoBody):
    """Point-mass gravity plus the J2 zonal harmonic, with the body's pole along +z."""

    j2: float = EARTH_J2
    radius: float = EARTH_RADIUS

    def _j2_acceleration(self, position):
        x, y, z = position
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)
        factor = 1.5 * self.j2 * self.mu * self.radius**2 / r**5
        ratio = 5.0 * z * z / r2
        return factor * np.array([x * (ratio - 1.0), y * (ratio - 1.0), z * (ratio - 3.0)])

    def rhs(self, t, state):
        base = super().rhs(t, state)
        base[3:] += self._j2_acceleration(state[:3])
        return base

    def jacobian(self, t, state):
        matrix = super().jacobian(t, state)
        x, y, z = state[:3]
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)
        k = 1.5 * self.j2 * self.mu * self.radius**2
        r5 = r**5
        r7 = r**7
        r9 = r**9
        # Analytic gradient of the J2 acceleration.
        gradient = np.empty((3, 3))
        gradient[0, 0] = k * ((5 * z * z / r2 - 1) / r5 + x * x * (-35 * z * z / r9 + 5 / r7))
        gradient[1, 1] = k * ((5 * z * z / r2 - 1) / r5 + y * y * (-35 * z * z / r9 + 5 / r7))
        gradient[0, 1] = gradient[1, 0] = k * x * y * (-35 * z * z / r9 + 5 / r7)
        gradient[0, 2] = gradient[2, 0] = k * x * z * (-35 * z * z / r9 + 15 / r7)
        gradient[1, 2] = gradient[2, 1] = k * y * z * (-35 * z * z / r9 + 15 / r7)
        gradient[2, 2] = k * ((5 * z * z / r2 - 3) / r5 + z * z * (-35 * z * z / r9 + 25 / r7))
        matrix[3:, :3] += gradient
        return matrix

    def describe(self) -> dict:
        return {"type": "TwoBodyJ2", "mu": self.mu, "j2": self.j2, "radius": self.radius}


@dataclass(frozen=True)
class CR3BP(OdeDynamics):
    """Circular restricted three-body problem in the rotating frame.

    The model is non-dimensional: the primaries are one distance unit apart,
    their mean motion is one, and ``mass_ratio`` is the mass of the smaller
    primary over the total mass.  ``mu`` is left undefined because two-body
    orbital elements have no meaning here.
    """

    mass_ratio: float = 0.012150585609624
    atol_position: float = field(default=1e-13, kw_only=True)
    atol_velocity: float = field(default=1e-13, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.mass_ratio <= 0.5:
            raise ValueError("mass_ratio must lie between 0 and 0.5")

    def _distances(self, x, y, z):
        m = self.mass_ratio
        d1 = np.sqrt((x + m) ** 2 + y * y + z * z)
        d2 = np.sqrt((x - 1.0 + m) ** 2 + y * y + z * z)
        return d1, d2

    def rhs(self, t, state):
        x, y, z, vx, vy, vz = state
        m = self.mass_ratio
        d1, d2 = self._distances(x, y, z)
        ax = 2.0 * vy + x - (1.0 - m) * (x + m) / d1**3 - m * (x - 1.0 + m) / d2**3
        ay = -2.0 * vx + y - (1.0 - m) * y / d1**3 - m * y / d2**3
        az = -(1.0 - m) * z / d1**3 - m * z / d2**3
        return np.array([vx, vy, vz, ax, ay, az])

    def jacobian(self, t, state):
        x, y, z = state[:3]
        m = self.mass_ratio
        d1, d2 = self._distances(x, y, z)
        p1 = np.array([x + m, y, z])
        p2 = np.array([x - 1.0 + m, y, z])
        pseudo = (
            np.diag([1.0, 1.0, 0.0])
            - (1.0 - m) * (np.eye(3) / d1**3 - 3.0 * np.outer(p1, p1) / d1**5)
            - m * (np.eye(3) / d2**3 - 3.0 * np.outer(p2, p2) / d2**5)
        )
        matrix = np.zeros((6, 6))
        matrix[:3, 3:] = np.eye(3)
        matrix[3:, :3] = pseudo
        matrix[3, 4] = 2.0
        matrix[4, 3] = -2.0
        return matrix

    def jacobi_constant(self, state) -> float:
        """Jacobi integral, conserved along any trajectory."""
        x, y, z, vx, vy, vz = np.asarray(state, dtype=float)
        m = self.mass_ratio
        d1, d2 = self._distances(x, y, z)
        return float(x * x + y * y + 2.0 * (1.0 - m) / d1 + 2.0 * m / d2 - (vx * vx + vy * vy + vz * vz))

    def describe(self) -> dict:
        return {"type": "CR3BP", "mass_ratio": self.mass_ratio}
