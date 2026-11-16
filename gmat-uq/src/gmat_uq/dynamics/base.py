# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Abstract interface between the GMAT-UQ engines and an orbit propagator."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["Dynamics", "PropagationError", "finite_difference_stm"]


class PropagationError(RuntimeError):
    """Raised when a propagator cannot reach the requested time."""


def finite_difference_stm(propagate, state, t0: float, t1: float, relative_step: float = 1e-6):
    """State transition matrix by central differences of a propagation function.

    ``propagate(state, t0, t1)`` returns the state at ``t1``.  The perturbation
    of each component is ``relative_step`` times the position or velocity
    magnitude, which keeps the truncation and rounding errors balanced for
    orbit problems.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    position_step = relative_step * max(float(np.linalg.norm(state[:3])), 1e-3)
    velocity_step = relative_step * max(float(np.linalg.norm(state[3:])), 1e-6)
    steps = np.array([position_step] * 3 + [velocity_step] * 3)
    stm = np.empty((6, 6))
    for column in range(6):
        plus = state.copy()
        minus = state.copy()
        plus[column] += steps[column]
        minus[column] -= steps[column]
        stm[:, column] = (propagate(plus, t0, t1) - propagate(minus, t0, t1)) / (2.0 * steps[column])
    return stm


class Dynamics(ABC):
    """Propagates a 6-element orbit state between two times.

    States are ``[x, y, z, vx, vy, vz]`` in km and km/s, in an inertial frame
    centred on the body named by the implementation.  Times are seconds from
    the mission epoch.  Implementations must be picklable so that Monte Carlo
    worker processes can rebuild them.
    """

    #: Gravitational parameter of the central body (km^3/s^2), used by orbital targets.
    mu: float | None = None

    @abstractmethod
    def propagate(self, state, t0: float, t1: float) -> np.ndarray:
        """Return the state at ``t1`` given the state at ``t0``."""

    def propagate_stm(self, state, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the state at ``t1`` and the state transition matrix from ``t0`` to ``t1``.

        The default implementation differentiates :meth:`propagate` numerically.
        """
        final = self.propagate(state, t0, t1)
        return final, finite_difference_stm(self.propagate, state, t0, t1)

    def close(self) -> None:
        """Release external resources.  The default does nothing."""

    def describe(self) -> dict:
        """Return a JSON-serialisable description for reports."""
        return {"type": type(self).__name__}
