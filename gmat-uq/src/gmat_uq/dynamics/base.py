# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Abstract interface between the GMAT-UQ engines and an orbit propagator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np

__all__ = ["Dynamics", "PropagationError", "finite_difference_stm"]


class PropagationError(RuntimeError):
    """Raised when a propagator cannot reach the requested time."""


def finite_difference_stm(propagate, state, t0: float, t1: float, relative_step: float = 1e-6, parameters=None):
    """State transition matrix by central differences of a propagation function.

    ``propagate(state, t0, t1, parameters=...)`` returns the state at ``t1``.
    The perturbation of each component is ``relative_step`` times the position
    or velocity magnitude, which keeps the truncation and rounding errors
    balanced for orbit problems.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    position_step = relative_step * max(float(np.linalg.norm(state[:3])), 1e-3)
    velocity_step = relative_step * max(float(np.linalg.norm(state[3:])), 1e-6)
    steps = np.array([position_step] * 3 + [velocity_step] * 3)
    extra = {"parameters": parameters} if parameters else {}  # keeps simple propagate(state, t0, t1) callables usable
    stm = np.empty((6, 6))
    for column in range(6):
        plus = state.copy()
        minus = state.copy()
        plus[column] += steps[column]
        minus[column] -= steps[column]
        stm[:, column] = (propagate(plus, t0, t1, **extra) - propagate(minus, t0, t1, **extra)) / (2.0 * steps[column])
    return stm


class Dynamics(ABC):
    """Propagates a 6-element orbit state between two times.

    States are ``[x, y, z, vx, vy, vz]`` in km and km/s, in an inertial frame
    centred on the body named by the implementation.  Times are seconds from
    the mission epoch.  Implementations must be picklable so that Monte Carlo
    worker processes can rebuild them.

    Every propagation accepts ``parameters``: a mapping from the name of a
    model parameter (for example ``"Cd"``) to the value to use for that
    propagation only.  ``None`` or an empty mapping means the nominal values.
    This is how uncertain *consider* parameters enter the analysis.
    """

    #: Gravitational parameter of the central body (km^3/s^2), used by orbital targets.
    mu: float | None = None

    @abstractmethod
    def propagate(self, state, t0: float, t1: float, parameters: Mapping[str, float] | None = None) -> np.ndarray:
        """Return the state at ``t1`` given the state at ``t0``."""

    def propagate_stm(
        self, state, t0: float, t1: float, parameters: Mapping[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the state at ``t1`` and the state transition matrix from ``t0`` to ``t1``.

        The default implementation differentiates :meth:`propagate` numerically.
        """
        extra = {"parameters": parameters} if parameters else {}
        final = self.propagate(state, t0, t1, **extra)
        return final, finite_difference_stm(self.propagate, state, t0, t1, parameters=parameters)

    def parameter_value(self, name: str) -> float:
        """Nominal value of a model parameter that can be made uncertain."""
        raise ValueError(f"{type(self).__name__} has no parameter '{name}'")

    def close(self) -> None:
        """Release external resources.  The default does nothing."""

    def describe(self) -> dict:
        """Return a JSON-serialisable description for reports."""
        return {"type": type(self).__name__}


def check_parameters(dynamics: Dynamics, parameters) -> dict[str, float]:
    """Validate a parameter mapping against what ``dynamics`` supports and return a plain dict."""
    if not parameters:
        return {}
    values = {}
    for name, value in dict(parameters).items():
        dynamics.parameter_value(name)  # raises for unknown names
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"parameter '{name}' must be finite")
        values[str(name)] = value
    return values
