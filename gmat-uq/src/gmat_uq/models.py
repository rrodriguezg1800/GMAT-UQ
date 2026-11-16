# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Error models: initial-state dispersion, maneuver execution error and navigation error.

Units follow GMAT: kilometres, kilometres per second, seconds and radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .linalg import check_covariance, covariance_factor, perpendicular_basis, symmetrize

__all__ = [
    "ManeuverErrorModel",
    "StateDispersion",
    "NavigationErrorModel",
    "ProcessNoise",
    "local_frame_rotation",
    "rotate_covariance_to_inertial",
]

_ZERO_DV = 1e-15  # km/s; below this a commanded delta-V has no usable direction


@lru_cache(maxsize=8)
def _quasi_normal_points(log2_points: int) -> np.ndarray:
    """Deterministic, low-discrepancy standard-normal points in three dimensions."""
    from scipy.stats import norm, qmc

    sampler = qmc.Sobol(d=3, scramble=True, seed=20260101)
    uniform = sampler.random_base2(m=log2_points)
    uniform = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    return norm.ppf(uniform)


@dataclass(frozen=True)
class ManeuverErrorModel:
    """Gates model of impulsive-maneuver execution error.

    The error of a commanded delta-V ``u`` with magnitude ``m`` has a component
    along ``u`` with standard deviation ``sqrt(s1^2 + (s2 m)^2)`` and a component
    on each of the two axes perpendicular to ``u`` with standard deviation
    ``sqrt(s3^2 + (s4 m)^2)``.

    Parameters
    ----------
    fixed_magnitude_sigma:
        ``s1``, km/s.
    proportional_magnitude_sigma:
        ``s2``, dimensionless fraction of the commanded magnitude.
    fixed_pointing_sigma:
        ``s3``, km/s per perpendicular axis.
    proportional_pointing_sigma:
        ``s4``, radians per perpendicular axis.
    """

    fixed_magnitude_sigma: float = 0.0
    proportional_magnitude_sigma: float = 0.0
    fixed_pointing_sigma: float = 0.0
    proportional_pointing_sigma: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "fixed_magnitude_sigma",
            "proportional_magnitude_sigma",
            "fixed_pointing_sigma",
            "proportional_pointing_sigma",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite, non-negative number")
            object.__setattr__(self, name, value)

    @property
    def is_zero(self) -> bool:
        return (
            self.fixed_magnitude_sigma == 0.0
            and self.proportional_magnitude_sigma == 0.0
            and self.fixed_pointing_sigma == 0.0
            and self.proportional_pointing_sigma == 0.0
        )

    def component_sigmas(self, magnitude):
        """Return (along-track sigma, per-axis cross sigma) for a commanded magnitude."""
        magnitude = np.asarray(magnitude, dtype=float)
        along = np.sqrt(self.fixed_magnitude_sigma**2 + (self.proportional_magnitude_sigma * magnitude) ** 2)
        cross = np.sqrt(self.fixed_pointing_sigma**2 + (self.proportional_pointing_sigma * magnitude) ** 2)
        return along, cross

    @property
    def _isotropic_sigma(self) -> float:
        # A zero commanded delta-V has no direction.  The fixed errors are then
        # spread evenly so that the total variance (s1^2 + 2 s3^2) is preserved.
        return float(np.sqrt((self.fixed_magnitude_sigma**2 + 2.0 * self.fixed_pointing_sigma**2) / 3.0))

    def covariance(self, dv) -> np.ndarray:
        """Execution-error covariance (3x3, km^2/s^2) for one commanded delta-V."""
        dv = np.asarray(dv, dtype=float).reshape(3)
        magnitude = float(np.linalg.norm(dv))
        if magnitude < _ZERO_DV:
            return self._isotropic_sigma**2 * np.eye(3)
        unit = dv / magnitude
        along, cross = self.component_sigmas(magnitude)
        projector = np.outer(unit, unit)
        return float(along) ** 2 * projector + float(cross) ** 2 * (np.eye(3) - projector)

    def apply(self, dv, normals) -> np.ndarray:
        """Map standard-normal draws into execution errors.

        ``dv`` and ``normals`` have shape ``(..., 3)``.  The first normal drives
        the magnitude error and the other two drive the pointing error.
        """
        dv = np.asarray(dv, dtype=float)
        normals = np.asarray(normals, dtype=float)
        single = dv.ndim == 1
        dv2 = np.atleast_2d(dv)
        z = np.atleast_2d(normals)
        magnitude = np.linalg.norm(dv2, axis=1)
        zero = magnitude < _ZERO_DV
        safe = np.where(zero, 1.0, magnitude)
        unit = dv2 / safe[:, None]
        unit[zero] = np.array([1.0, 0.0, 0.0])
        first, second = perpendicular_basis(unit)
        along, cross = self.component_sigmas(magnitude)
        error = (
            (along * z[:, 0])[:, None] * unit
            + (cross * z[:, 1])[:, None] * first
            + (cross * z[:, 2])[:, None] * second
        )
        if zero.any():
            error[zero] = self._isotropic_sigma * z[zero]
        return error[0] if single else error

    def sample(self, dv, rng: np.random.Generator) -> np.ndarray:
        """Draw execution errors for the commanded delta-V vector(s) ``dv``."""
        dv = np.asarray(dv, dtype=float)
        return self.apply(dv, rng.standard_normal(dv.shape))

    def expected_covariance(self, dv_mean, dv_covariance=None, log2_points: int = 13) -> np.ndarray:
        """Expected execution-error covariance when the command itself is random.

        The commanded delta-V is modelled as Gaussian with mean ``dv_mean`` and
        covariance ``dv_covariance``.  The proportional terms have a closed
        form; the fixed terms need the expected direction dyad ``E[u u^T / m^2]``,
        which is evaluated with a deterministic quasi-Monte Carlo rule.
        """
        mean = np.asarray(dv_mean, dtype=float).reshape(3)
        if dv_covariance is None:
            return self.covariance(mean)
        cov = symmetrize(np.asarray(dv_covariance, dtype=float).reshape(3, 3))
        mean_sq = float(mean @ mean)
        if np.trace(cov) <= 1e-24 * max(mean_sq, 1e-30) or not cov.any():
            return self.covariance(mean)
        second_moment = cov + np.outer(mean, mean)  # E[u u^T]
        expected_m2 = float(np.trace(second_moment))  # E[m^2]
        needs_direction = self.fixed_magnitude_sigma != 0.0 or self.fixed_pointing_sigma != 0.0
        if needs_direction:
            points = mean + _quasi_normal_points(log2_points) @ covariance_factor(cov).T
            norms = np.linalg.norm(points, axis=1)
            good = norms > _ZERO_DV
            units = points[good] / norms[good, None]
            direction_dyad = units.T @ units / max(int(good.sum()), 1)
        else:
            direction_dyad = np.eye(3) / 3.0
        identity = np.eye(3)
        result = (
            self.fixed_magnitude_sigma**2 * direction_dyad
            + self.proportional_magnitude_sigma**2 * second_moment
            + self.fixed_pointing_sigma**2 * (identity - direction_dyad)
            + self.proportional_pointing_sigma**2 * (expected_m2 * identity - second_moment)
        )
        return symmetrize(result)


def local_frame_rotation(state, frame: str) -> np.ndarray:
    """Rotation matrix whose columns are the local-frame axes expressed inertially.

    ``frame`` is ``"RIC"`` (radial, in-track, cross-track) or ``"VNB"``
    (velocity, orbit normal, bi-normal), both built from the orbit state.
    """
    state = np.asarray(state, dtype=float).reshape(6)
    position, velocity = state[:3], state[3:]
    normal = np.cross(position, velocity)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0.0:
        raise ValueError("local frames are undefined for rectilinear motion")
    normal = normal / normal_norm
    name = frame.upper()
    if name in ("RIC", "RTN", "LVLH"):
        radial = position / np.linalg.norm(position)
        in_track = np.cross(normal, radial)
        return np.column_stack([radial, in_track, normal])
    if name == "VNB":
        along = velocity / np.linalg.norm(velocity)
        binormal = np.cross(along, normal)
        return np.column_stack([along, normal, binormal])
    if name in ("INERTIAL", "MJ2000EQ"):
        return np.eye(3)
    raise ValueError(f"unknown frame '{frame}' (use 'inertial', 'RIC' or 'VNB')")


def rotate_covariance_to_inertial(covariance, state, frame: str) -> np.ndarray:
    """Rotate a 6x6 covariance given in a local orbit frame into the inertial frame.

    The local frame is treated as non-rotating at the instant of ``state``; the
    position and velocity blocks are rotated with the same matrix.
    """
    rotation = local_frame_rotation(state, frame)
    block = np.zeros((6, 6))
    block[:3, :3] = rotation
    block[3:, 3:] = rotation
    return symmetrize(block @ np.asarray(covariance, dtype=float) @ block.T)


@dataclass(frozen=True, eq=False)
class _GaussianState:
    """Zero-mean Gaussian error on the 6-element orbit state."""

    covariance: np.ndarray

    _label = "covariance"

    def __post_init__(self) -> None:
        cov = check_covariance(self.covariance, 6, self._label)
        cov.setflags(write=False)
        object.__setattr__(self, "covariance", cov)

    @classmethod
    def from_sigmas(cls, sigmas):
        """Build a diagonal covariance from six standard deviations (km, km/s)."""
        sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
        if sigmas.size == 2:
            sigmas = np.repeat(sigmas, 3)
        if sigmas.size != 6:
            raise ValueError("expected six sigmas, or two (position, velocity)")
        if np.any(sigmas < 0.0):
            raise ValueError("sigmas must be non-negative")
        return cls(np.diag(sigmas**2))

    @classmethod
    def from_local(cls, covariance, state, frame: str = "RIC"):
        """Build from a covariance expressed in a local orbit frame at ``state``."""
        return cls(rotate_covariance_to_inertial(covariance, state, frame))

    @classmethod
    def from_file(cls, path):
        """Read a 6x6 covariance from a whitespace- or comma-separated text file."""
        text = Path(path).read_text(encoding="utf-8")
        rows = [line.replace(",", " ").split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith(("#", "%"))]
        return cls(np.array(rows, dtype=float))

    @property
    def factor(self) -> np.ndarray:
        """Square factor ``L`` with ``L L^T`` equal to the covariance."""
        return covariance_factor(self.covariance)

    def apply(self, normals) -> np.ndarray:
        """Map standard-normal draws (shape ``(..., 6)``) into state errors."""
        return np.asarray(normals, dtype=float) @ self.factor.T

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        shape = (6,) if size is None else (size, 6)
        return self.apply(rng.standard_normal(shape))

    def __eq__(self, other) -> bool:
        return type(other) is type(self) and np.array_equal(self.covariance, other.covariance)

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.covariance.tobytes()))


class StateDispersion(_GaussianState):
    """Dispersion of the true initial state about the nominal (for example injection errors)."""

    _label = "initial dispersion covariance"


class NavigationErrorModel(_GaussianState):
    """Orbit-knowledge error at the time a correction maneuver is designed.

    The estimate used to design the maneuver is ``truth + e`` with ``e`` drawn
    from this covariance, independently of the true dispersion.
    """

    _label = "navigation knowledge covariance"


@dataclass(frozen=True)
class ProcessNoise:
    """State-noise-compensation process noise (white acceleration noise).

    ``accel_sigma`` holds one value per inertial axis in km/s^(3/2), the same
    unit as GMAT's ``ProcessNoiseModel.AccelNoiseSigma``.  ``max_step`` limits
    the length of the intervals over which the kinematic formula is applied.
    """

    accel_sigma: tuple[float, float, float]
    max_step: float = 600.0

    def __post_init__(self) -> None:
        sigma = tuple(float(v) for v in np.asarray(self.accel_sigma, dtype=float).reshape(3))
        if any(v < 0.0 or not np.isfinite(v) for v in sigma):
            raise ValueError("accel_sigma must hold three finite, non-negative values")
        if not (self.max_step > 0.0):
            raise ValueError("max_step must be positive")
        object.__setattr__(self, "accel_sigma", sigma)
        object.__setattr__(self, "max_step", float(self.max_step))

    def covariance(self, dt: float) -> np.ndarray:
        """Process-noise covariance accumulated over an interval of ``dt`` seconds."""
        dt = abs(float(dt))
        q = np.zeros((6, 6))
        for axis, sigma in enumerate(self.accel_sigma):
            psd = sigma**2
            q[axis, axis] = psd * dt**3 / 3.0
            q[axis, axis + 3] = q[axis + 3, axis] = psd * dt**2 / 2.0
            q[axis + 3, axis + 3] = psd * dt
        return q
