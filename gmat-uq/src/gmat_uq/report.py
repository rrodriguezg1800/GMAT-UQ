# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""JSON report and console summary of an analysis."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .consistency import ConsistencyReport
from .dynamics.base import Dynamics
from .lincov import LinCovResult
from .mission import Mission
from .montecarlo import MonteCarloResult
from .sigma import SigmaPointResult
from .stats import error_ellipse, wilks_sample_size

__all__ = ["build_report", "write_report", "format_summary"]


def _clean(value):
    """Convert NumPy types and non-finite numbers into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return None
        if np.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    return value


def _sigmas(covariance) -> np.ndarray:
    return np.sqrt(np.clip(np.diag(np.atleast_2d(covariance)), 0.0, None))


def _mission_section(mission: Mission) -> dict:
    maneuvers = []
    for maneuver in mission.maneuvers:
        entry = {
            "name": maneuver.name,
            "time": maneuver.time,
            "dv": list(maneuver.dv),
            "closed_loop": maneuver.closed_loop,
            "error_model": None,
            "navigation_sigmas": None,
        }
        if maneuver.error_model is not None:
            model = maneuver.error_model
            entry["error_model"] = {
                "fixed_magnitude_sigma": model.fixed_magnitude_sigma,
                "proportional_magnitude_sigma": model.proportional_magnitude_sigma,
                "fixed_pointing_sigma": model.fixed_pointing_sigma,
                "proportional_pointing_sigma": model.proportional_pointing_sigma,
            }
        if maneuver.closed_loop:
            entry["target"] = maneuver.target.describe()
            entry["target_time"] = maneuver.target_time
            if maneuver.navigation is not None:
                entry["navigation_sigmas"] = _sigmas(maneuver.navigation.covariance)
        maneuvers.append(entry)
    return {
        "initial_time": mission.initial_time,
        "final_time": mission.final_time,
        "initial_state": list(mission.initial_state),
        "initial_dispersion_sigmas": None if mission.initial_dispersion is None else _sigmas(mission.initial_dispersion.covariance),
        "process_noise": None
        if mission.process_noise is None
        else {"accel_sigma": list(mission.process_noise.accel_sigma), "max_step": mission.process_noise.max_step},
        "final_target": None if mission.final_target is None else mission.final_target.describe(),
        "maneuvers": maneuvers,
    }


def _target_section(labels, covariance) -> dict:
    covariance = np.atleast_2d(covariance)
    section = {"labels": list(labels), "covariance": covariance, "sigmas": _sigmas(covariance)}
    if covariance.shape == (2, 2):
        section["ellipse_1sigma"] = error_ellipse(covariance, 1.0).to_dict()
        section["ellipse_3sigma"] = error_ellipse(covariance, 3.0).to_dict()
    return section


def build_report(
    mission: Mission,
    dynamics: Dynamics,
    lincov: LinCovResult | None = None,
    sigma_point: SigmaPointResult | None = None,
    monte_carlo: MonteCarloResult | None = None,
    consistency: ConsistencyReport | None = None,
    probability: float = 0.99,
    confidence: float = 0.95,
) -> dict:
    """Assemble every available result into one JSON-serialisable dictionary.

    Lengths are in km, velocities in km/s, times in seconds and angles in degrees.
    """
    from . import __version__

    report: dict = {
        "gmat_uq_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "units": {"length": "km", "velocity": "km/s", "time": "s", "angle": "deg"},
        "mission": _mission_section(mission),
        "dynamics": dynamics.describe(),
        "sample_size_guidance": {
            "probability": probability,
            "confidence": confidence,
            "wilks_minimum_samples": wilks_sample_size(probability, confidence),
        },
    }

    if lincov is not None:
        nominal = lincov.nominal
        nodes = []
        for node, time in enumerate(nominal.times):
            plan = nominal.plan_at(node)
            nodes.append(
                {
                    "time": float(time),
                    "maneuver": None if plan is None else plan.maneuver.name,
                    "nominal_state_before": nominal.states_before[node],
                    "nominal_state_after": nominal.states_after[node],
                    "sigma_before": lincov.sigmas(node, after=False),
                    "sigma_after": lincov.sigmas(node, after=True),
                    "position_sigma_rss_after": lincov.position_sigma_rss(node),
                    "velocity_sigma_rss_after": lincov.velocity_sigma_rss(node),
                }
            )
        section = {
            "nodes": nodes,
            "final_covariance": lincov.final_covariance,
            "final_sigmas": lincov.sigmas(),
            "maneuvers": [
                {
                    "name": m.name,
                    "time": m.time,
                    "closed_loop": m.closed_loop,
                    "dv_nominal": m.dv_nominal,
                    "correction_covariance": m.correction_covariance,
                    "correction_sigma_rss": float(np.sqrt(np.trace(m.correction_covariance))),
                    "execution_covariance": m.execution_covariance,
                    "execution_sigma_rss": float(np.sqrt(np.trace(m.execution_covariance))),
                }
                for m in lincov.maneuvers
            ],
            "delta_v": lincov.dv_statistics(probability, confidence),
            "delta_v_samples": 0 if lincov.samples is None else int(lincov.samples.indices.size),
            "seed": lincov.seed,
        }
        if lincov.target_covariance is not None:
            section["target"] = _target_section(mission.final_target.labels, lincov.target_covariance)
            section["target"]["nominal_value"] = nominal.final_target_value
        report["lincov"] = section

    if sigma_point is not None:
        report["sigma_point"] = sigma_point.to_dict()

    if monte_carlo is not None:
        section = {
            "n_samples": monte_carlo.n_samples,
            "seed": monte_carlo.seed,
            "guidance": monte_carlo.guidance,
            "workers": monte_carlo.workers,
            "elapsed_seconds": monte_carlo.elapsed_seconds,
            "failed_samples": monte_carlo.n_failed,
            "unconverged_samples": monte_carlo.n_unconverged,
            "final_covariance": None,
            "final_sigmas": None,
            "final_mean": None,
            "delta_v": None,
        }
        # Statistics need at least two samples that reached the final time.
        if int(monte_carlo.valid.sum()) >= 2:
            section["final_covariance"] = monte_carlo.final_covariance
            section["final_sigmas"] = _sigmas(monte_carlo.final_covariance)
            section["final_mean"] = monte_carlo.final_deviation[monte_carlo.valid].mean(axis=0)
            if monte_carlo.nominal.plans:
                section["delta_v"] = monte_carlo.dv_statistics(probability, confidence)
            if monte_carlo.target_covariance is not None:
                section["target"] = _target_section(mission.final_target.labels, monte_carlo.target_covariance)
                section["target"]["mean"] = monte_carlo.target_deviation[monte_carlo.valid].mean(axis=0)
        report["monte_carlo"] = section

    if consistency is not None:
        report["consistency"] = consistency.to_dict()

    return _clean(report)


def write_report(report: dict, path) -> Path:
    """Write a report produced by :func:`build_report` to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _fmt(value, unit_scale: float = 1.0, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value * unit_scale:.{digits}g}"


def format_summary(report: dict) -> str:
    """Human-readable summary of a report (delta-V in m/s, distances in km)."""
    lines = [f"GMAT-UQ {report['gmat_uq_version']} analysis summary", ""]
    probability = report["sample_size_guidance"]["probability"]
    percent = f"{100 * probability:g}"

    lincov = report.get("lincov")
    if lincov:
        sigmas = lincov["final_sigmas"]
        lines.append("Linear covariance")
        lines.append(f"  final position sigma (x, y, z) [km]   : {_fmt(sigmas[0])}, {_fmt(sigmas[1])}, {_fmt(sigmas[2])}")
        lines.append(f"  final velocity sigma (x, y, z) [m/s]  : {_fmt(sigmas[3], 1e3)}, {_fmt(sigmas[4], 1e3)}, {_fmt(sigmas[5], 1e3)}")
        delta_v = lincov.get("delta_v")
        if delta_v:
            total = delta_v["total"]
            lines.append(f"  nominal delta-V [m/s]                  : {_fmt(delta_v['nominal_total'], 1e3, 6)}")
            lines.append(f"  mean total delta-V [m/s]               : {_fmt(total['mean'], 1e3, 6)}")
            lines.append(f"  delta-V {percent} [m/s]                      : {_fmt(total['quantile']['value'], 1e3, 6)}")
        target = lincov.get("target")
        if target:
            pairs = ", ".join(f"{label}={_fmt(sigma)}" for label, sigma in zip(target["labels"], target["sigmas"]))
            lines.append(f"  final target sigmas                    : {pairs}")
        lines.append("")

    sigma_point = report.get("sigma_point")
    if sigma_point:
        verdict = "linear results trusted" if sigma_point["linear_ok"] else "NONLINEAR ARCS FOUND - prefer Monte Carlo"
        lines.append("Sigma-point check")
        lines.append(f"  largest nonlinearity index             : {_fmt(sigma_point['max_index'])} (threshold {_fmt(sigma_point['threshold'])})")
        lines.append(f"  verdict                                : {verdict}")
        for segment in sigma_point["segments"]:
            if segment["flagged"]:
                lines.append(
                    f"    flagged arc {segment['start_time']:.6g} s -> {segment['end_time']:.6g} s, index {_fmt(segment['index'])}"
                )
        lines.append("")

    monte_carlo = report.get("monte_carlo")
    if monte_carlo:
        sigmas = monte_carlo["final_sigmas"]
        lines.append(f"Monte Carlo ({monte_carlo['n_samples']} samples, {monte_carlo['guidance']} guidance, {monte_carlo['workers']} worker(s), {monte_carlo['elapsed_seconds']:.1f} s)")
        if sigmas is None:
            lines.append("  no statistics: fewer than two samples reached the final time")
        else:
            lines.append(f"  final position sigma (x, y, z) [km]   : {_fmt(sigmas[0])}, {_fmt(sigmas[1])}, {_fmt(sigmas[2])}")
            lines.append(f"  final velocity sigma (x, y, z) [m/s]  : {_fmt(sigmas[3], 1e3)}, {_fmt(sigmas[4], 1e3)}, {_fmt(sigmas[5], 1e3)}")
        delta_v = monte_carlo.get("delta_v")
        if delta_v:
            quantile = delta_v["total"]["quantile"]
            lines.append(f"  mean total delta-V [m/s]               : {_fmt(delta_v['total']['mean'], 1e3, 6)}")
            lines.append(
                f"  delta-V {percent} [m/s]                      : {_fmt(quantile['value'], 1e3, 6)} "
                f"({100 * quantile['confidence']:g}% interval {_fmt(quantile['lower'], 1e3, 6)} to {_fmt(quantile['upper'], 1e3, 6)})"
            )
        lines.append(f"  failed / unconverged samples           : {monte_carlo['failed_samples']} / {monte_carlo['unconverged_samples']}")
        lines.append("")

    consistency = report.get("consistency")
    if consistency:
        final = consistency["final_state"]
        lines.append("Linear covariance against Monte Carlo")
        lines.append(f"  sigma ratio over all directions        : {_fmt(final['sigma_ratio_min'])} to {_fmt(final['sigma_ratio_max'])}")
        delta_v = consistency.get("delta_v_quantile")
        if delta_v:
            inside = "inside" if delta_v["linear_inside_monte_carlo_interval"] else "outside"
            lines.append(f"  linear delta-V {percent}                      : {inside} the Monte Carlo interval (difference {100 * delta_v['relative_difference']:.2f}%)")
        lines.append(f"  verdict                                : {'engines agree' if consistency['agrees'] else 'ENGINES DISAGREE'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
