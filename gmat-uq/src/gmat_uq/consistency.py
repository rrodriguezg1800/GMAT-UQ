# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Consistency report between the linear covariance and the Monte Carlo engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lincov import LinCovResult
from .montecarlo import MonteCarloResult
from .stats import CovarianceComparison, QuantileEstimate, compare_covariance, quantile_with_interval, sigma_ratio_interval

__all__ = ["ConsistencyReport", "compare_engines"]


@dataclass(frozen=True)
class ConsistencyReport:
    """How well the linear covariance agrees with the Monte Carlo samples.

    ``agrees`` is the overall verdict.  The engines disagree on dispersions
    only when the difference is both large and statistically significant: some
    directional sigma ratio differs from one by more than ``sigma_tolerance``
    *and* the covariance test rejects equality at the ``significance`` level.
    A small sample can show large ratios by chance, and a very large sample
    can detect differences too small to matter; requiring both avoids the two
    false alarms.  The engines disagree on delta-V when the linear quantile is
    outside the Monte Carlo confidence interval and further than
    ``dv_tolerance`` from the Monte Carlo estimate.
    """

    final_state: CovarianceComparison
    component_sigma_ratios: tuple[tuple[float, float, float], ...]  # (ratio, lower, upper) per state component
    target: CovarianceComparison | None
    dv_quantile_linear: float | None
    dv_quantile_monte_carlo: QuantileEstimate | None
    dv_inside_interval: bool | None
    dv_relative_difference: float | None
    sigma_tolerance: float
    dv_tolerance: float
    significance: float
    agrees: bool

    def to_dict(self) -> dict:
        return {
            "agrees": self.agrees,
            "sigma_tolerance": self.sigma_tolerance,
            "dv_tolerance": self.dv_tolerance,
            "significance": self.significance,
            "final_state": self.final_state.to_dict(),
            "component_sigma_ratios": [
                {"ratio": ratio, "lower": lower, "upper": upper} for ratio, lower, upper in self.component_sigma_ratios
            ],
            "target": None if self.target is None else self.target.to_dict(),
            "delta_v_quantile": None
            if self.dv_quantile_linear is None
            else {
                "linear": self.dv_quantile_linear,
                "monte_carlo": self.dv_quantile_monte_carlo.to_dict(),
                "linear_inside_monte_carlo_interval": self.dv_inside_interval,
                "relative_difference": self.dv_relative_difference,
            },
        }


def compare_engines(
    lincov: LinCovResult,
    monte_carlo: MonteCarloResult,
    probability: float = 0.99,
    confidence: float = 0.95,
    sigma_tolerance: float = 0.1,
    dv_tolerance: float = 0.02,
    significance: float = 0.01,
) -> ConsistencyReport:
    """Compare a linear covariance result with a Monte Carlo result of the same mission."""
    valid = monte_carlo.valid
    final = monte_carlo.final_deviation[valid]
    n = final.shape[0]
    final_comparison = compare_covariance(final, lincov.final_covariance)

    sample_sigma = final.std(axis=0, ddof=1)
    linear_sigma = lincov.sigmas()
    ratios = []
    for component in range(6):
        if linear_sigma[component] > 0.0:
            ratios.append(sigma_ratio_interval(sample_sigma[component], linear_sigma[component], n, confidence))
        else:
            ratios.append((float("nan"), float("nan"), float("nan")))

    target_comparison = None
    if monte_carlo.target_deviation is not None and lincov.target_covariance is not None:
        target_comparison = compare_covariance(monte_carlo.target_deviation[valid], lincov.target_covariance)

    dv_linear = dv_mc = inside = relative = None
    dv_ok = True
    if lincov.samples is not None and len(lincov.nominal.plans) > 0:
        dv_linear = float(np.quantile(lincov.samples.total_dv, probability))
        dv_mc = quantile_with_interval(monte_carlo.total_dv, probability, confidence)
        inside = bool(dv_mc.contains(dv_linear))
        relative = float(abs(dv_linear - dv_mc.value) / dv_mc.value) if dv_mc.value > 0.0 else 0.0
        dv_ok = inside or relative <= dv_tolerance

    def consistent(comparison) -> bool:
        return comparison.max_sigma_deviation <= sigma_tolerance or comparison.p_value >= significance

    sigma_ok = consistent(final_comparison)
    if target_comparison is not None:
        sigma_ok = sigma_ok and consistent(target_comparison)
    return ConsistencyReport(
        final_state=final_comparison,
        component_sigma_ratios=tuple(ratios),
        target=target_comparison,
        dv_quantile_linear=dv_linear,
        dv_quantile_monte_carlo=dv_mc,
        dv_inside_interval=inside,
        dv_relative_difference=relative,
        sigma_tolerance=float(sigma_tolerance),
        dv_tolerance=float(dv_tolerance),
        significance=float(significance),
        agrees=bool(sigma_ok and dv_ok),
    )
