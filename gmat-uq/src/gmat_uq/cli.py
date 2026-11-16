# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Command-line interface: ``gmat-uq run mission.toml``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import run_analysis
from .config import METHODS, ConfigError, load_config
from .dynamics import GmatNotFoundError, PropagationError
from .report import format_summary, write_report
from .stats import wilks_sample_size

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmat-uq",
        description="Linear covariance and navigation-aware Monte Carlo analysis for GMAT missions.",
    )
    parser.add_argument("--version", action="version", version=f"gmat-uq {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the analysis described by a configuration file")
    run.add_argument("config", help="path to a .toml or .json configuration file")
    run.add_argument("--report", help="write the JSON report here (overrides the configuration)")
    run.add_argument("--methods", nargs="+", choices=METHODS, help="engines to run (overrides the configuration)")
    run.add_argument("--samples", type=int, help="number of Monte Carlo samples (overrides the configuration)")
    run.add_argument("--workers", type=int, help="number of worker processes (overrides the configuration)")
    run.add_argument("--seed", type=int, help="master random seed (overrides the configuration)")
    run.add_argument("--guidance", choices=("nonlinear", "linear"), help="Monte Carlo guidance mode")
    run.add_argument("--quiet", action="store_true", help="do not print the summary")
    run.add_argument("--values", help="also write the main results as 'key = value' lines (read by the GMAT plugin)")

    wilks = commands.add_parser("wilks", help="smallest sample that bounds a quantile with a given confidence")
    wilks.add_argument("--probability", type=float, default=0.99)
    wilks.add_argument("--confidence", type=float, default=0.95)
    return parser


def _run(arguments) -> int:
    config = load_config(arguments.config)
    if arguments.methods:
        config.methods = tuple(arguments.methods)
    if arguments.samples is not None:
        config.samples = arguments.samples
    if arguments.workers is not None:
        config.workers = arguments.workers
    if arguments.seed is not None:
        config.seed = arguments.seed
    if arguments.guidance:
        config.guidance = arguments.guidance
    if arguments.report:
        config.report = arguments.report
    if config.samples < 1 or config.workers < 1:
        raise ConfigError("samples and workers must be at least 1")

    progress = None
    if not arguments.quiet and "monte_carlo" in config.methods:
        def progress(done: int, total: int) -> None:
            print(f"\r  Monte Carlo: {done}/{total} samples", end="", file=sys.stderr, flush=True)
            if done >= total:
                print(file=sys.stderr)

    result = run_analysis(config, progress)
    if arguments.values:
        write_values(result, arguments.values)
    if config.report:
        path = write_report(result.report, config.report)
        if not arguments.quiet:
            print(f"Report written to {path}")
    if not arguments.quiet:
        print(format_summary(result.report))
    # Exit code 2 signals a completed analysis whose linear results should not be trusted.
    untrusted = (result.sigma_point is not None and not result.sigma_point.linear_ok) or (
        result.consistency is not None and "lincov" in config.methods and not result.consistency.agrees
    )
    return 2 if untrusted else 0


def write_values(result, path) -> None:
    """Write the headline results as plain ``key = value`` lines (km and km/s).

    Keys whose engine did not run are left out.  ``linear_trusted`` and
    ``engines_agree`` are 1 or 0.
    """
    values = {}
    if result.lincov is not None:
        statistics = result.lincov.dv_statistics()
        if statistics is not None:
            values["delta_v_99"] = statistics["total"]["quantile"]["value"]
            values["delta_v_mean"] = statistics["total"]["mean"]
        values["final_position_sigma"] = result.lincov.position_sigma_rss()
        values["final_velocity_sigma"] = result.lincov.velocity_sigma_rss()
    if result.sigma_point is not None:
        values["nonlinearity_index"] = result.sigma_point.max_index
        values["linear_trusted"] = 1.0 if result.sigma_point.linear_ok else 0.0
    if result.monte_carlo is not None and result.monte_carlo.nominal.plans and result.monte_carlo.valid.sum() > 0:
        values["monte_carlo_delta_v_99"] = result.monte_carlo.dv_statistics()["total"]["quantile"]["value"]
        if result.lincov is None:
            covariance = result.monte_carlo.final_covariance
            values["final_position_sigma"] = float(np.sqrt(np.trace(covariance[:3, :3])))
            values["final_velocity_sigma"] = float(np.sqrt(np.trace(covariance[3:, 3:])))
    if result.consistency is not None:
        values["engines_agree"] = 1.0 if result.consistency.agrees else 0.0
    lines = [f"{key} = {float(value)!r}" for key, value in values.items() if np.isfinite(value)]
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            return _run(arguments)
        if arguments.command == "wilks":
            print(wilks_sample_size(arguments.probability, arguments.confidence))
            return 0
    except (ConfigError, GmatNotFoundError, PropagationError, FileNotFoundError, ValueError) as error:
        print(f"gmat-uq: error: {error}", file=sys.stderr)
        return 1
    return 0  # pragma: no cover - argparse enforces a command


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
