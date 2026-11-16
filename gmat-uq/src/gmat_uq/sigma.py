# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Sigma-point check of the linearization.

For every coast arc the dispersion covariance at the start of the arc is
represented by twelve sigma points (plus the nominal, which is already known:
thirteen trajectories in total).  The points are propagated through the full
nonlinear dynamics and their mean and covariance are compared with the linear
prediction.  The check costs twelve propagations per arc and answers one
question: can the linear covariance be trusted on this arc?
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .dynamics.base import Dynamics
from .lincov import LinCovResult
from .linalg import covariance_factor, symmetrize, whitening_matrix

__all__ = ["SegmentCheck", "SigmaPointResult", "run_sigma_point_check"]


@dataclass(frozen=True)
class SegmentCheck:
    """Nonlinearity measures of one coast arc.

    ``mean_shift_sigma`` is the distance between the sigma-point mean and the
    nominal end state, in units of the linear one-sigma dispersion.
    ``sigma_deviation`` is the largest relative difference between sigma-point
    and linear standard deviations over all directions.  Both are zero for
    linear dynamics.
    """

    start_node: int
    end_node: int
    start_time: float
    end_time: float
    mean_shift_sigma: float
    sigma_deviation: float
    flagged: bool

    @property
    def index(self) -> float:
        """Single nonlinearity index: the larger of the two measures."""
        return max(self.mean_shift_sigma, self.sigma_deviation)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["index"] = self.index
        return result


@dataclass(frozen=True)
class SigmaPointResult:
    segments: tuple[SegmentCheck, ...]
    threshold: float
    propagations: int

    @property
    def max_index(self) -> float:
        return max((segment.index for segment in self.segments), default=0.0)

    @property
    def flagged_segments(self) -> tuple[SegmentCheck, ...]:
        return tuple(segment for segment in self.segments if segment.flagged)

    @property
    def linear_ok(self) -> bool:
        """True when no arc exceeds the threshold."""
        return not self.flagged_segments

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "max_index": self.max_index,
            "linear_ok": self.linear_ok,
            "propagations": self.propagations,
            "segments": [segment.to_dict() for segment in self.segments],
        }


def run_sigma_point_check(lincov: LinCovResult, dynamics: Dynamics, threshold: float = 0.1) -> SigmaPointResult:
    """Check every coast arc of a linear covariance result against sigma points.

    Parameters
    ----------
    lincov:
        Result of :func:`gmat_uq.lincov.run_lincov`.
    dynamics:
        The dynamics used for that analysis.
    threshold:
        An arc is flagged when its nonlinearity index exceeds this value.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")
    nominal = lincov.nominal
    dimension = 6
    spread = np.sqrt(dimension)  # symmetric cubature set: +/- sqrt(n) columns of the factor
    segments = []
    propagations = 0
    for node in range(nominal.n_nodes - 1):
        t0, t1 = nominal.times[node], nominal.times[node + 1]
        covariance = lincov.covariance_after[node]
        linear = symmetrize(nominal.stms[node] @ covariance @ nominal.stms[node].T)
        whitener = whitening_matrix(linear)
        if whitener.shape[0] == 0:
            segments.append(SegmentCheck(node, node + 1, float(t0), float(t1), 0.0, 0.0, False))
            continue
        factor = covariance_factor(covariance)
        start = nominal.states_after[node]
        end_nominal = nominal.states_before[node + 1]
        ends = []
        for column in range(dimension):
            offset = spread * factor[:, column]
            if not offset.any():
                # A zero column contributes two copies of the nominal end state.
                ends.extend([end_nominal, end_nominal])
                continue
            ends.append(np.asarray(dynamics.propagate(start + offset, t0, t1), dtype=float))
            ends.append(np.asarray(dynamics.propagate(start - offset, t0, t1), dtype=float))
            propagations += 2
        ends = np.array(ends)
        mean = ends.mean(axis=0)
        centered = ends - mean
        sample_cov = centered.T @ centered / ends.shape[0]
        white_cov = whitener @ sample_cov @ whitener.T
        ratios = np.sqrt(np.clip(np.linalg.eigvalsh(symmetrize(white_cov)), 0.0, None))
        sigma_deviation = float(np.abs(ratios - 1.0).max())
        mean_shift = float(np.linalg.norm(whitener @ (mean - end_nominal)))
        flagged = max(sigma_deviation, mean_shift) > threshold
        segments.append(SegmentCheck(node, node + 1, float(t0), float(t1), mean_shift, sigma_deviation, flagged))
    return SigmaPointResult(tuple(segments), float(threshold), propagations)
