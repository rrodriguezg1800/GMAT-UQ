# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Statistics used to report dispersions and to compare the two engines."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as _stats

from .linalg import symmetrize, whitening_matrix

__all__ = [
    "QuantileEstimate",
    "quantile_with_interval",
    "wilks_sample_size",
    "CovarianceComparison",
    "compare_covariance",
    "sigma_ratio_interval",
    "ErrorEllipse",
    "error_ellipse",
    "summarize_magnitudes",
]


@dataclass(frozen=True)
class QuantileEstimate:
    """Sample quantile with a distribution-free confidence interval."""

    probability: float
    value: float
    lower: float
    upper: float
    confidence: float
    n_samples: int

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def to_dict(self) -> dict:
        return asdict(self)


def quantile_with_interval(samples, probability: float = 0.99, confidence: float = 0.95) -> QuantileEstimate:
    """Estimate a quantile and bound it with order statistics.

    The interval ``[x_(l), x_(u)]`` covers the true quantile with probability
    at least ``confidence`` for any continuous distribution.  When the sample
    is too small for a finite bound on one side, that bound is infinite.
    """
    data = np.sort(np.asarray(samples, dtype=float).ravel())
    n = data.size
    if n == 0:
        raise ValueError("no samples")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    alpha = 1.0 - confidence
    # P(x_(l) <= q <= x_(u)) = F(u - 1) - F(l - 1) with F the Binomial(n, p) CDF.
    lower_rank = int(_stats.binom.ppf(alpha / 2.0, n, probability))  # F(lower_rank - 1) <= alpha / 2
    upper_rank = int(_stats.binom.ppf(1.0 - alpha / 2.0, n, probability)) + 1  # F(upper_rank - 1) >= 1 - alpha / 2
    lower = data[lower_rank - 1] if lower_rank >= 1 else -np.inf
    upper = data[upper_rank - 1] if upper_rank <= n else np.inf
    return QuantileEstimate(probability, float(np.quantile(data, probability)), float(lower), float(upper), confidence, n)


def wilks_sample_size(probability: float = 0.99, confidence: float = 0.95) -> int:
    """Smallest sample whose maximum bounds the ``probability`` quantile with ``confidence``.

    This is the first-order, one-sided Wilks criterion: ``1 - p^N >= confidence``.
    """
    if not 0.0 < probability < 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("probability and confidence must lie strictly between 0 and 1")
    return int(np.ceil(np.log(1.0 - confidence) / np.log(probability) - 1e-12))


@dataclass(frozen=True)
class CovarianceComparison:
    """Agreement between a sample and a reference (linear) covariance.

    ``sigma_ratio_min`` and ``sigma_ratio_max`` are the smallest and largest
    ratios of sample to reference standard deviation over all directions; they
    equal one when the two covariances are identical.  ``p_value`` is from the
    likelihood-ratio test of equal covariance for Gaussian samples and
    ``mean_p_value`` from the chi-square test of zero mean.
    """

    n_samples: int
    dimension: int
    sigma_ratio_min: float
    sigma_ratio_max: float
    statistic: float
    degrees_of_freedom: int
    p_value: float
    mean_offset_sigma: float
    mean_p_value: float

    @property
    def max_sigma_deviation(self) -> float:
        return max(abs(self.sigma_ratio_min - 1.0), abs(self.sigma_ratio_max - 1.0))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["max_sigma_deviation"] = self.max_sigma_deviation
        return result


def compare_covariance(samples, reference_covariance, reference_mean=None, rtol: float = 1e-10) -> CovarianceComparison:
    """Compare the covariance and mean of ``samples`` (rows) with reference values.

    The comparison is made in whitened coordinates, so it does not depend on
    units, and only in the subspace where the reference covariance is
    non-singular: directions whose reference variance is below ``rtol`` times
    the largest one (a sigma 100,000 times smaller, by default) are treated as
    exactly constrained and left out.
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim != 2:
        raise ValueError("samples must be a two-dimensional array (one row per sample)")
    n = data.shape[0]
    reference = symmetrize(np.asarray(reference_covariance, dtype=float))
    if reference.shape != (data.shape[1], data.shape[1]):
        raise ValueError("reference covariance does not match the sample dimension")
    mean = np.zeros(data.shape[1]) if reference_mean is None else np.asarray(reference_mean, dtype=float)
    whitener = whitening_matrix(reference, rtol)
    p = whitener.shape[0]
    if p == 0:
        raise ValueError("reference covariance is zero")
    if n <= p + 1:
        raise ValueError(f"need more than {p + 1} samples to compare a {p}-dimensional covariance")
    white = (data - mean) @ whitener.T
    sample_mean = white.mean(axis=0)
    sample_cov = np.cov(white, rowvar=False, ddof=1).reshape(p, p)
    eigenvalues = np.clip(np.linalg.eigvalsh(sample_cov), 0.0, None)
    sign, logdet = np.linalg.slogdet(sample_cov)
    if sign <= 0:
        statistic = np.inf
    else:
        statistic = (n - 1) * (np.trace(sample_cov) - logdet - p)
        statistic *= 1.0 - (2.0 * p + 1.0 - 2.0 / (p + 1.0)) / (6.0 * (n - 1))  # Bartlett correction
    dof = p * (p + 1) // 2
    mean_statistic = n * float(sample_mean @ sample_mean)
    return CovarianceComparison(
        n_samples=n,
        dimension=p,
        sigma_ratio_min=float(np.sqrt(eigenvalues.min())),
        sigma_ratio_max=float(np.sqrt(eigenvalues.max())),
        statistic=float(statistic),
        degrees_of_freedom=dof,
        p_value=float(_stats.chi2.sf(statistic, dof)),
        mean_offset_sigma=float(np.sqrt(sample_mean @ sample_mean)),
        mean_p_value=float(_stats.chi2.sf(mean_statistic, p)),
    )


def sigma_ratio_interval(sample_sigma: float, reference_sigma: float, n_samples: int, confidence: float = 0.95):
    """Ratio of a sample standard deviation to a reference one, with its confidence interval.

    Returns ``(ratio, lower, upper)``.  The interval assumes Gaussian samples
    (``(n - 1) s^2 / sigma^2`` is chi-square distributed).
    """
    if reference_sigma <= 0.0:
        raise ValueError("reference_sigma must be positive")
    if n_samples < 2:
        raise ValueError("need at least two samples")
    ratio = sample_sigma / reference_sigma
    alpha = 1.0 - confidence
    dof = n_samples - 1
    lower = ratio * np.sqrt(dof / _stats.chi2.ppf(1.0 - alpha / 2.0, dof))
    upper = ratio * np.sqrt(dof / _stats.chi2.ppf(alpha / 2.0, dof))
    return float(ratio), float(lower), float(upper)


@dataclass(frozen=True)
class ErrorEllipse:
    """Constant-probability ellipse of a two-dimensional Gaussian."""

    semi_major: float
    semi_minor: float
    angle_deg: float  # orientation of the major axis, measured from the first coordinate axis
    n_sigma: float
    probability: float  # probability mass inside the ellipse

    def to_dict(self) -> dict:
        return asdict(self)


def error_ellipse(covariance, n_sigma: float = 1.0) -> ErrorEllipse:
    """Ellipse ``x^T P^-1 x = n_sigma^2`` of a 2x2 covariance."""
    cov = symmetrize(np.asarray(covariance, dtype=float))
    if cov.shape != (2, 2):
        raise ValueError("error_ellipse needs a 2x2 covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    major = eigenvectors[:, 1]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    angle = (angle + 90.0) % 180.0 - 90.0  # an axis has no direction: keep the angle in (-90, 90]
    probability = 1.0 - np.exp(-0.5 * n_sigma**2)
    return ErrorEllipse(
        semi_major=float(n_sigma * np.sqrt(eigenvalues[1])),
        semi_minor=float(n_sigma * np.sqrt(eigenvalues[0])),
        angle_deg=float(angle),
        n_sigma=float(n_sigma),
        probability=float(probability),
    )


def summarize_magnitudes(values, probability: float = 0.99, confidence: float = 0.95) -> dict:
    """Mean, standard deviation, maximum and an upper quantile of non-negative samples."""
    data = np.asarray(values, dtype=float).ravel()
    quantile = quantile_with_interval(data, probability, confidence)
    return {
        "n_samples": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "max": float(data.max()),
        "quantile": quantile.to_dict(),
    }
