# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Target functions: what a correction maneuver aims at, and what dispersions are reported in.

A target is a vector-valued function of the orbit state.  The built-in targets
use the names of the corresponding GMAT calculation parameters (``RMAG``,
``INC``, ``BdotT`` ...) and the same units, so results can be compared with a
GMAT report directly.  Angles are in degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "Target",
    "ParameterTarget",
    "FunctionTarget",
    "PARAMETERS",
    "cartesian_to_keplerian",
    "b_plane",
]

_ANGLES = {"INC", "RAAN", "AOP", "TA"}
_PERIODIC = {"RAAN", "AOP", "TA"}


def cartesian_to_keplerian(state, mu: float) -> dict[str, float]:
    """Classical orbital elements (GMAT conventions, degrees) from a Cartesian state."""
    state = np.asarray(state, dtype=float).reshape(6)
    r_vec, v_vec = state[:3], state[3:]
    r = np.linalg.norm(r_vec)
    v2 = float(v_vec @ v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    energy = 0.5 * v2 - mu / r
    e_vec = ((v2 - mu / r) * r_vec - (r_vec @ v_vec) * v_vec) / mu
    ecc = np.linalg.norm(e_vec)
    sma = -mu / (2.0 * energy) if abs(energy) > 0.0 else np.inf
    inc = np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))
    node = np.array([-h_vec[1], h_vec[0], 0.0])
    node_norm = np.linalg.norm(node)
    equatorial = node_norm < 1e-11 * h
    circular = ecc < 1e-11
    if equatorial:
        raan = 0.0
        node_dir = np.array([1.0, 0.0, 0.0])
    else:
        node_dir = node / node_norm
        raan = np.arctan2(node_dir[1], node_dir[0])
    # In-plane axis perpendicular to the node, used to measure angles in the orbit plane.
    plane_y = np.cross(h_vec / h, node_dir)
    if circular:
        aop = 0.0
        peri_dir = node_dir
    else:
        peri_dir = e_vec / ecc
        aop = np.arctan2(peri_dir @ plane_y, peri_dir @ node_dir)
    peri_y = np.cross(h_vec / h, peri_dir)
    ta = np.arctan2(r_vec @ peri_y, r_vec @ peri_dir)
    to_deg = np.degrees
    return {
        "SMA": float(sma),
        "ECC": float(ecc),
        "INC": float(to_deg(inc)),
        "RAAN": float(to_deg(raan) % 360.0),
        "AOP": float(to_deg(aop) % 360.0),
        "TA": float(to_deg(ta) % 360.0),
    }


def b_plane(state, mu: float) -> tuple[float, float]:
    """B-plane coordinates (``B.T``, ``B.R``) in km for a hyperbolic state.

    ``T`` is perpendicular to the incoming asymptote ``S`` and to the frame's
    z-axis (``T = S x k``), and ``R = S x T``, matching GMAT's ``BdotT`` and
    ``BdotR`` parameters.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    r_vec, v_vec = state[:3], state[3:]
    r = np.linalg.norm(r_vec)
    v2 = float(v_vec @ v_vec)
    h_vec = np.cross(r_vec, v_vec)
    h_hat = h_vec / np.linalg.norm(h_vec)
    e_vec = ((v2 - mu / r) * r_vec - (r_vec @ v_vec) * v_vec) / mu
    ecc = np.linalg.norm(e_vec)
    if ecc <= 1.0:
        raise ValueError("the B-plane is defined for hyperbolic orbits only (eccentricity > 1)")
    energy = 0.5 * v2 - mu / r
    sma = -mu / (2.0 * energy)  # negative for a hyperbola
    b_mag = abs(sma) * np.sqrt(ecc * ecc - 1.0)
    p_hat = e_vec / ecc
    q_hat = np.cross(h_hat, p_hat)
    s_hat = p_hat / ecc + np.sqrt(1.0 - 1.0 / (ecc * ecc)) * q_hat
    b_vec = b_mag * np.cross(s_hat, h_hat)
    t_vec = np.cross(s_hat, np.array([0.0, 0.0, 1.0]))
    t_hat = t_vec / np.linalg.norm(t_vec)
    r_hat = np.cross(s_hat, t_hat)
    return float(b_vec @ t_hat), float(b_vec @ r_hat)


def _kepler(name: str) -> Callable[[np.ndarray, float], float]:
    return lambda state, mu: cartesian_to_keplerian(state, mu)[name]


def _rad_per(state, mu):
    elements = cartesian_to_keplerian(state, mu)
    return elements["SMA"] * (1.0 - elements["ECC"])


def _rad_apo(state, mu):
    elements = cartesian_to_keplerian(state, mu)
    return elements["SMA"] * (1.0 + elements["ECC"])


def _c3(state, mu):
    return float(state[3:] @ state[3:] - 2.0 * mu / np.linalg.norm(state[:3]))


#: Scalar parameters available to :class:`ParameterTarget`, keyed by GMAT parameter name.
PARAMETERS: dict[str, Callable[[np.ndarray, float], float]] = {
    "X": lambda s, mu: float(s[0]),
    "Y": lambda s, mu: float(s[1]),
    "Z": lambda s, mu: float(s[2]),
    "VX": lambda s, mu: float(s[3]),
    "VY": lambda s, mu: float(s[4]),
    "VZ": lambda s, mu: float(s[5]),
    "RMAG": lambda s, mu: float(np.linalg.norm(s[:3])),
    "VMAG": lambda s, mu: float(np.linalg.norm(s[3:])),
    "SMA": _kepler("SMA"),
    "ECC": _kepler("ECC"),
    "INC": _kepler("INC"),
    "RAAN": _kepler("RAAN"),
    "AOP": _kepler("AOP"),
    "TA": _kepler("TA"),
    "RadPer": _rad_per,
    "RadApo": _rad_apo,
    "C3Energy": _c3,
    "BdotT": lambda s, mu: b_plane(s, mu)[0],
    "BdotR": lambda s, mu: b_plane(s, mu)[1],
}

_CANONICAL = {name.upper(): name for name in PARAMETERS}
_NEEDS_MU = set(PARAMETERS) - {"X", "Y", "Z", "VX", "VY", "VZ", "RMAG", "VMAG"}


class Target:
    """Vector-valued function of the orbit state."""

    #: Labels of the components, one per element of the returned vector.
    labels: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.labels)

    def __call__(self, state) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def residual(self, state, goal) -> np.ndarray:
        """Difference between the target value at ``state`` and ``goal``."""
        return self(state) - np.asarray(goal, dtype=float)

    def jacobian(self, state, relative_step: float = 1e-7) -> np.ndarray:
        """Partial derivatives of the target with respect to the state (size x 6).

        Central differences are used, with the residual function so that
        periodic angles are differentiated across their wrap-around.
        """
        state = np.asarray(state, dtype=float).reshape(6)
        position_step = relative_step * max(float(np.linalg.norm(state[:3])), 1e-3)
        velocity_step = relative_step * max(float(np.linalg.norm(state[3:])), 1e-6)
        steps = [position_step] * 3 + [velocity_step] * 3
        reference = self(state)
        jacobian = np.empty((self.size, 6))
        for column, step in enumerate(steps):
            plus = state.copy()
            minus = state.copy()
            plus[column] += step
            minus[column] -= step
            jacobian[:, column] = (self.residual(plus, reference) - self.residual(minus, reference)) / (2.0 * step)
        return jacobian

    def describe(self) -> dict:
        return {"type": type(self).__name__, "labels": list(self.labels)}


@dataclass(frozen=True)
class ParameterTarget(Target):
    """Target made of named orbit parameters, for example ``("BdotT", "BdotR")``.

    Parameters
    ----------
    names:
        Parameter names from :data:`PARAMETERS` (case-insensitive).
    mu:
        Gravitational parameter in km^3/s^2 of the body the parameters refer
        to.  Required by every parameter except the Cartesian components,
        ``RMAG`` and ``VMAG``.
    origin:
        Optional state ``[x, y, z, vx, vy, vz]`` of another body, in the frame
        of the dynamics, at the time the target is evaluated.  When given, the
        parameters are computed from the state relative to that body, which is
        how a B-plane at the Moon is targeted from an Earth-centred
        propagation.  Because targets are evaluated at a fixed time, the
        body's state is a constant.
    """

    names: tuple[str, ...]
    mu: float | None = None
    origin: tuple[float, ...] | None = None
    labels: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        names = (self.names,) if isinstance(self.names, str) else tuple(self.names)
        if not names:
            raise ValueError("a target needs at least one parameter")
        canonical = []
        for name in names:
            key = str(name).upper()
            if key not in _CANONICAL:
                raise ValueError(f"unknown parameter '{name}'; available: {', '.join(sorted(PARAMETERS))}")
            canonical.append(_CANONICAL[key])
        if len(set(canonical)) != len(canonical):
            raise ValueError("a target cannot repeat a parameter")
        if self.mu is None and any(name in _NEEDS_MU for name in canonical):
            raise ValueError("this target needs the gravitational parameter 'mu'")
        if self.mu is not None and not self.mu > 0.0:
            raise ValueError("mu must be positive")
        if self.origin is not None:
            origin = np.asarray(self.origin, dtype=float).reshape(-1)
            if origin.shape != (6,) or not np.all(np.isfinite(origin)):
                raise ValueError("origin must hold six finite values")
            object.__setattr__(self, "origin", tuple(float(v) for v in origin))
        object.__setattr__(self, "names", tuple(canonical))
        object.__setattr__(self, "labels", tuple(canonical))

    def __call__(self, state) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(6)
        if self.origin is not None:
            state = state - np.array(self.origin)
        return np.array([PARAMETERS[name](state, self.mu) for name in self.names])

    def residual(self, state, goal) -> np.ndarray:
        difference = self(state) - np.asarray(goal, dtype=float)
        for index, name in enumerate(self.names):
            if name in _PERIODIC:
                difference[index] = (difference[index] + 180.0) % 360.0 - 180.0
        return difference

    def describe(self) -> dict:
        return {
            "type": "ParameterTarget",
            "names": list(self.names),
            "mu": self.mu,
            "origin": None if self.origin is None else list(self.origin),
        }

    @classmethod
    def position(cls) -> "ParameterTarget":
        """Inertial position vector (fixed-time-of-arrival targeting)."""
        return cls(("X", "Y", "Z"))

    @classmethod
    def b_plane(cls, mu: float, origin=None) -> "ParameterTarget":
        """B-plane aim point (``BdotT``, ``BdotR``), optionally relative to another body."""
        return cls(("BdotT", "BdotR"), mu, origin)


@dataclass(frozen=True)
class FunctionTarget(Target):
    """Target defined by an arbitrary Python function of the state.

    The function must be importable at module level if the mission is used with
    more than one Monte Carlo worker, because it is pickled.
    """

    function: Callable[[np.ndarray], Sequence[float]]
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        labels = tuple(str(label) for label in self.labels)
        if not labels:
            raise ValueError("FunctionTarget needs one label per output component")
        object.__setattr__(self, "labels", labels)

    def __call__(self, state) -> np.ndarray:
        value = np.atleast_1d(np.asarray(self.function(np.asarray(state, dtype=float).reshape(6)), dtype=float))
        if value.shape != (self.size,):
            raise ValueError(f"target function returned shape {value.shape}, expected ({self.size},)")
        return value
