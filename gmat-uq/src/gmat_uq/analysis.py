# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Run every requested engine on a configuration and assemble the report."""

from __future__ import annotations

from dataclasses import dataclass

from .config import AnalysisConfig
from .consistency import ConsistencyReport, compare_engines
from .lincov import LinCovResult, run_lincov
from .mission import Nominal, compute_nominal
from .montecarlo import MonteCarloResult, run_monte_carlo
from .report import build_report
from .sigma import SigmaPointResult, run_sigma_point_check

__all__ = ["AnalysisResult", "run_analysis"]


@dataclass(frozen=True)
class AnalysisResult:
    """Raw results of each engine plus the assembled report."""

    nominal: Nominal
    lincov: LinCovResult | None
    sigma_point: SigmaPointResult | None
    monte_carlo: MonteCarloResult | None
    consistency: ConsistencyReport | None
    report: dict


def run_analysis(config: AnalysisConfig, progress=None) -> AnalysisResult:
    """Run the engines listed in ``config.methods``.

    The sigma-point check and the consistency report both need the linear
    covariance, so it is computed whenever either is requested.
    """
    methods = set(config.methods)
    nominal = compute_nominal(config.mission, config.dynamics)

    lincov = None
    if "lincov" in methods or "sigma_point" in methods:
        lincov = run_lincov(config.mission, config.dynamics, nominal, config.dv_samples, config.seed)

    sigma_point = None
    if "sigma_point" in methods:
        sigma_point = run_sigma_point_check(lincov, config.dynamics, config.threshold)

    monte_carlo = None
    if "monte_carlo" in methods:
        monte_carlo = run_monte_carlo(
            config.mission,
            config.dynamics,
            config.samples,
            seed=config.seed,
            workers=config.workers,
            guidance=config.guidance,
            nominal=nominal,
            progress=progress,
        )

    consistency = None
    if lincov is not None and monte_carlo is not None and monte_carlo.valid.sum() > 8:
        consistency = compare_engines(lincov, monte_carlo, config.probability, config.confidence)

    report = build_report(
        config.mission,
        config.dynamics,
        lincov=lincov if "lincov" in methods else None,
        sigma_point=sigma_point,
        monte_carlo=monte_carlo,
        consistency=consistency if "lincov" in methods else None,
        probability=config.probability,
        confidence=config.confidence,
    )
    return AnalysisResult(nominal, lincov, sigma_point, monte_carlo, consistency, report)
