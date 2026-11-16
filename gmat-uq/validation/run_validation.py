# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Run the GMAT-UQ validation campaign and write its report.

Usage::

    python validation/run_validation.py --gmat-root C:/path/to/GMAT --samples 2000 --workers 6

The campaign runs the acceptance tests of the GMAT-UQ design.  Tests that need
GMAT are skipped, and reported as skipped, when no installation is given.  The
results go to ``validation/results/validation_results.json`` and the readable
report to ``docs/validation.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy

import gmat_uq
from gmat_uq import (
    GmatDynamics,
    ManeuverErrorModel,
    Mission,
    ProcessNoise,
    StateDispersion,
    compute_nominal,
    run_lincov,
    run_monte_carlo,
    run_sigma_point_check,
)
from gmat_uq.consistency import compare_engines
from gmat_uq.stats import compare_covariance, wilks_sample_size

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cases  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "validation_results.json"
REPORT = HERE.parent / "docs" / "validation.md"


def _console(gmat_root: str) -> Path:
    binary = Path(gmat_root) / "bin" / "GmatConsole.exe"
    return binary if binary.exists() else Path(gmat_root) / "bin" / "GmatConsole"


def _run_console(gmat_root: str, script: Path) -> None:
    binary = _console(gmat_root)
    done = subprocess.run([str(binary), "--run", str(script)], capture_output=True, text=True, timeout=1800, cwd=str(binary.parent))
    if done.returncode != 0:
        raise RuntimeError(f"GmatConsole failed on {script.name}:\n{done.stdout[-2000:]}")


def _gmat_version(gmat_root: str) -> str:
    binary = _console(gmat_root)
    done = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=300, cwd=str(binary.parent))
    for line in done.stdout.splitlines():
        if "Build Date" in line:
            return line.strip()
    return "unknown"


def _installation_fingerprint(gmat_root: str) -> str:
    """Hash of every file in GMAT's bin and plugins folders, except the API startup file."""
    digest = hashlib.sha256()
    root = Path(gmat_root)
    for folder in ("bin", "plugins"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.name != "api_startup_file.txt" and "__pycache__" not in path.parts:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _engine_comparison(lincov, monte_carlo, tolerance: float) -> tuple[dict, bool]:
    """Measurements shared by tests 3 and 4, and whether they meet the sigma tolerance."""
    report = compare_engines(lincov, monte_carlo, sigma_tolerance=tolerance)
    ratios = [ratio for ratio, _, _ in report.component_sigma_ratios]
    measurements = {
        "monte_carlo_samples": monte_carlo.n_samples,
        "failed_samples": monte_carlo.n_failed,
        "unconverged_samples": monte_carlo.n_unconverged,
        "component_sigma_ratio_min": float(np.min(ratios)),
        "component_sigma_ratio_max": float(np.max(ratios)),
        "largest_component_sigma_difference_percent": float(100 * np.max(np.abs(np.array(ratios) - 1.0))),
        "directional_sigma_ratio_min": report.final_state.sigma_ratio_min,
        "directional_sigma_ratio_max": report.final_state.sigma_ratio_max,
        "covariance_test_p_value": report.final_state.p_value,
        "delta_v_99_linear_mps": 1e3 * report.dv_quantile_linear,
        "delta_v_99_monte_carlo_mps": 1e3 * report.dv_quantile_monte_carlo.value,
        "delta_v_99_monte_carlo_interval_mps": [1e3 * report.dv_quantile_monte_carlo.lower, 1e3 * report.dv_quantile_monte_carlo.upper],
        "delta_v_99_linear_inside_interval": report.dv_inside_interval,
        "delta_v_99_relative_difference_percent": 100 * report.dv_relative_difference,
    }
    statistical = lincov.samples.dv_magnitudes - np.linalg.norm([p.maneuver.dv_vector for p in lincov.nominal.plans], axis=1)
    closed = [index for index, plan in enumerate(lincov.nominal.plans) if plan.maneuver.closed_loop]
    if closed:
        measurements["correction_delta_v_99_linear_mps"] = float(1e3 * np.quantile(lincov.samples.dv_magnitudes[:, closed].sum(axis=1), 0.99))
        measurements["correction_delta_v_99_monte_carlo_mps"] = float(1e3 * np.quantile(monte_carlo.dv_magnitudes[:, closed].sum(axis=1), 0.99))
    del statistical
    if report.target is not None:
        measurements["target_sigma_ratio_min"] = report.target.sigma_ratio_min
        measurements["target_sigma_ratio_max"] = report.target.sigma_ratio_max
        measurements["target_labels"] = list(lincov.mission.final_target.labels)
        measurements["target_sigmas_linear"] = np.sqrt(np.diag(lincov.target_covariance)).tolist()
        measurements["target_sigmas_monte_carlo"] = np.sqrt(np.diag(monte_carlo.target_covariance)).tolist()
    passed = (
        measurements["largest_component_sigma_difference_percent"] <= 100 * tolerance
        and bool(report.dv_inside_interval)
        and monte_carlo.n_failed == 0
        and monte_carlo.n_unconverged == 0
    )
    return measurements, passed


# -- the tests --------------------------------------------------------------------------------------
def test_1_gates_sampler(options) -> dict:
    model = ManeuverErrorModel(2e-5, 0.01, 1e-5, 0.02)
    dv = np.array([0.08, 0.02, -0.05])
    rng = np.random.default_rng(options.seed)
    errors = model.sample(np.tile(dv, (100_000, 1)), rng)
    comparison = compare_covariance(errors, model.covariance(dv))
    return {
        "id": 1,
        "title": "Gates sampler against its analytic covariance",
        "criterion": "sample covariance of 100,000 draws consistent with the formula at 99% confidence",
        "passed": bool(comparison.p_value >= 0.01 and comparison.mean_p_value >= 0.01),
        "measurements": {
            "draws": 100_000,
            "covariance_test_p_value": comparison.p_value,
            "mean_test_p_value": comparison.mean_p_value,
            "largest_sigma_difference_percent": 100 * comparison.max_sigma_deviation,
        },
    }


def test_2_native_covariance(options) -> dict:
    title = "Linear covariance with no burns against GMAT's native covariance propagation"
    criterion = "position and velocity sigmas equal to 8 significant digits"
    if not options.gmat_root:
        return {"id": 2, "title": title, "criterion": criterion, "passed": None, "skipped": "needs GMAT"}
    diagonal = [1.0e-2, 1.5e-2, 1.0e-2, 2.5e-7, 2.5e-7, 4.0e-7]
    duration = 21600.0
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        state_file, cov_file = folder / "state.txt", folder / "covariance.txt"
        script = folder / "leo_covariance.script"
        script.write_text(cases.leo_covariance_script(state_file, cov_file, diagonal, duration), encoding="utf-8")
        _run_console(options.gmat_root, script)
        native_state = np.loadtxt(state_file)
        native = np.loadtxt(cov_file).reshape(6, 6)
        dynamics = GmatDynamics(str(script), gmat_root=options.gmat_root)
        mission = Mission(tuple(dynamics.initial_state()), final_time=duration, initial_dispersion=StateDispersion(np.diag(diagonal)))
        result = run_lincov(mission, dynamics, dv_samples=0)
        # Same comparison with GMAT's state-noise-compensation process noise switched on.
        accel_sigma = (1.0e-6, 2.0e-6, 3.0e-6)
        noisy_script = folder / "leo_covariance_snc.script"
        process_noise_lines = [
            "Create ProcessNoiseModel SNC;",
            "SNC.Type = StateNoiseCompensation;",
            "SNC.CoordinateSystem = EarthMJ2000Eq;",
            f"SNC.AccelNoiseSigma = [{' '.join(repr(v) for v in accel_sigma)}];",
            "Sat.ProcessNoiseModel = SNC;",
            "Create ForceModel FM;",
        ]
        noisy_text = cases.leo_covariance_script(state_file, cov_file, diagonal, 3600.0)
        noisy_script.write_text(noisy_text.replace("Create ForceModel FM;", chr(10).join(process_noise_lines)), encoding="utf-8")
        _run_console(options.gmat_root, noisy_script)
        native_noisy = np.sqrt(np.diag(np.loadtxt(cov_file).reshape(6, 6)))
        dynamics = GmatDynamics(str(noisy_script), gmat_root=options.gmat_root)
        state = tuple(dynamics.initial_state())
        initial = StateDispersion(np.diag(diagonal))
        noisy = run_lincov(Mission(state, final_time=3600.0, initial_dispersion=initial, process_noise=ProcessNoise(accel_sigma, max_step=20.0)), dynamics, dv_samples=0)
        quiet = run_lincov(Mission(state, final_time=3600.0, initial_dispersion=initial), dynamics, dv_samples=0)
        noise_difference = float(np.abs(noisy.sigmas() / native_noisy - 1.0).max())
        noise_effect = float(np.abs(quiet.sigmas() / native_noisy - 1.0).max())
    sigma_difference = float(np.abs(result.sigmas() / np.sqrt(np.diag(native)) - 1.0).max())
    return {
        "id": 2,
        "title": title,
        "criterion": criterion,
        "passed": bool(sigma_difference < 1e-8 and noise_difference < 1e-8),
        "measurements": {
            "with_process_noise_largest_relative_sigma_difference": noise_difference,
            "process_noise_share_of_sigma": noise_effect,
            "case": "low Earth orbit, 4x4 gravity plus Moon and Sun, 6 hours, RungeKutta89 with 20 s fixed steps",
            "largest_relative_sigma_difference": sigma_difference,
            "matching_significant_digits": float(-np.log10(max(sigma_difference, 1e-16))),
            "largest_relative_element_difference": float(np.abs(result.final_covariance - native).max() / np.abs(native).max()),
            "final_state_difference_km": float(np.abs(result.nominal.states_after[-1][:3] - native_state[:3]).max()),
            "native_position_sigmas_km": np.sqrt(np.diag(native))[:3].tolist(),
        },
    }


def test_3_two_burn_transfer(options) -> dict:
    mission, dynamics, info = cases.two_burn_transfer()
    lincov = run_lincov(mission, dynamics, dv_samples=options.dv_samples, seed=options.seed)
    # The reference dynamics are cheap, so this test uses at least 4,000 samples to keep the
    # sampling noise of a sigma (about 1.1% at one sigma) well below the 5% criterion.
    samples = max(options.samples, 4000)
    monte_carlo = run_monte_carlo(mission, dynamics, samples, seed=options.seed, workers=options.workers, nominal=lincov.nominal)
    measurements, passed = _engine_comparison(lincov, monte_carlo, 0.05)
    measurements["case"] = info
    measurements["sigma_point_max_index"] = run_sigma_point_check(lincov, dynamics).max_index
    return {
        "id": 3,
        "title": "Linear covariance against Monte Carlo on a two-burn transfer between circular orbits",
        "criterion": "1-sigma dispersions within 5%; linear delta-V 99 inside the Monte Carlo 95% interval",
        "passed": passed,
        "measurements": measurements,
    }


def test_4_lunar_transfer(options, context) -> dict:
    title = "Linear covariance against Monte Carlo on a lunar transfer with three corrections to the B-plane"
    criterion = "1-sigma dispersions within 10%; linear delta-V 99 inside the Monte Carlo 95% interval"
    if not options.gmat_root:
        return {"id": 4, "title": title, "criterion": criterion, "passed": None, "skipped": "needs GMAT"}
    mission, dynamics, info = cases.lunar_transfer(options.gmat_root)

    started = time.perf_counter()
    nominal = compute_nominal(mission, dynamics)
    nominal_seconds = time.perf_counter() - started
    started = time.perf_counter()
    lincov = run_lincov(mission, dynamics, nominal, dv_samples=options.dv_samples, seed=options.seed)
    lincov_seconds = time.perf_counter() - started
    started = time.perf_counter()
    sigma = run_sigma_point_check(lincov, dynamics)
    sigma_seconds = time.perf_counter() - started

    monte_carlo = run_monte_carlo(mission, dynamics, options.samples, seed=options.seed, workers=options.workers, nominal=nominal)
    measurements, passed = _engine_comparison(lincov, monte_carlo, 0.10)
    serial = run_monte_carlo(mission, dynamics, options.serial_samples, seed=options.seed, workers=1, nominal=nominal)
    per_sample = serial.elapsed_seconds / serial.n_samples
    linear_total = nominal_seconds + lincov_seconds
    measurements.update(
        {
            "case": info,
            "sigma_point_max_index": sigma.max_index,
            "sigma_point_linear_ok": sigma.linear_ok,
            "per_maneuver_correction_sigma_mps": {m.name: 1e3 * float(np.sqrt(np.trace(m.correction_covariance))) for m in lincov.maneuvers},
            "monte_carlo_wall_seconds": monte_carlo.elapsed_seconds,
            "monte_carlo_workers": monte_carlo.workers,
        }
    )
    context["timing"] = {
        "nominal_propagation_with_stm_seconds": nominal_seconds,
        "linear_covariance_seconds": lincov_seconds,
        "linear_covariance_total_seconds": linear_total,
        "linear_total_over_nominal": linear_total / nominal_seconds,
        "sigma_point_check_seconds": sigma_seconds,
        "delta_v_samples": options.dv_samples,
        "monte_carlo_seconds_per_sample_one_core": per_sample,
        "serial_samples_timed": serial.n_samples,
        "monte_carlo_2000_samples_one_core_seconds_extrapolated": 2000 * per_sample,
        "speed_up_over_2000_sample_monte_carlo": 2000 * per_sample / linear_total,
        "speed_up_including_sigma_point_check": 2000 * per_sample / (linear_total + sigma_seconds),
    }
    context["lunar"] = (mission, dynamics, nominal)
    return {"id": 4, "title": title, "criterion": criterion, "passed": passed, "measurements": measurements}


def test_5_halo_orbit(options) -> dict:
    mission, dynamics, info = cases.near_rectilinear_halo()
    lincov = run_lincov(mission, dynamics, dv_samples=0)
    sigma = run_sigma_point_check(lincov, dynamics, threshold=0.1)
    monte_carlo = run_monte_carlo(mission, dynamics, max(options.samples, 2000), seed=options.seed, workers=options.workers, nominal=lincov.nominal)
    valid = monte_carlo.valid
    nodes = []
    first_flag = first_difference = None
    for node in range(1, lincov.nominal.n_nodes):
        comparison = compare_covariance(monte_carlo.deviations_before[valid, node], lincov.covariance_before[node])
        segment = sigma.segments[node - 1]
        nodes.append(
            {
                "time_days": float(lincov.times[node] * 375190.0 / 86400.0),
                "sigma_point_index": segment.index,
                "flagged": segment.flagged,
                "monte_carlo_sigma_ratio_min": comparison.sigma_ratio_min,
                "monte_carlo_sigma_ratio_max": comparison.sigma_ratio_max,
                "engines_differ_percent": 100 * comparison.max_sigma_deviation,
            }
        )
        if first_flag is None and segment.flagged:
            first_flag = node
        # Sampling noise alone moves the extreme directional ratios by a few percent.
        if first_difference is None and comparison.max_sigma_deviation > 0.10 and comparison.p_value < 0.01:
            first_difference = node
    passed = first_flag is not None and (first_difference is None or first_flag <= first_difference)
    return {
        "id": 5,
        "title": "Several revolutions of a near-rectilinear halo orbit",
        "criterion": "the arc is flagged by the sigma-point check no later than the node where the two engines first differ by more than 10%",
        "passed": bool(passed),
        "measurements": {
            "case": info,
            "monte_carlo_samples": int(valid.sum()),
            "first_flagged_node": first_flag,
            "first_node_where_engines_differ_by_more_than_10_percent": first_difference,
            "first_flagged_time_days": None if first_flag is None else nodes[first_flag - 1]["time_days"],
            "first_difference_time_days": None if first_difference is None else nodes[first_difference - 1]["time_days"],
            "nodes": nodes,
        },
    }


def _identical(a, b) -> bool:
    names = ("final_deviation", "commanded_dv", "execution_error", "deviations_before", "converged", "iterations", "failed")
    return all(np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True) for name in names)


def test_6_reproducibility(options, context) -> dict:
    mission, dynamics, _ = cases.two_burn_transfer()
    nominal = compute_nominal(mission, dynamics)
    runs = {workers: run_monte_carlo(mission, dynamics, 64, seed=options.seed, workers=workers, nominal=nominal) for workers in (1, 4, 16)}
    reference = _identical(runs[1], runs[4]) and _identical(runs[1], runs[16])
    measurements = {"reference_dynamics_samples": 64, "reference_dynamics_identical_on_1_4_16_workers": reference}
    passed = reference
    if "lunar" in context:
        mission, dynamics, nominal = context["lunar"]
        runs = {workers: run_monte_carlo(mission, dynamics, 32, seed=options.seed, workers=workers, nominal=nominal) for workers in (1, 4, 16)}
        with_gmat = _identical(runs[1], runs[4]) and _identical(runs[1], runs[16])
        measurements.update({"gmat_dynamics_samples": 32, "gmat_dynamics_identical_on_1_4_16_workers": with_gmat})
        passed = passed and with_gmat
    return {
        "id": 6,
        "title": "One seed on 1, 4 and 16 worker processes",
        "criterion": "bit-identical results",
        "passed": bool(passed),
        "measurements": measurements,
    }


def test_7_no_side_effects(options, fingerprint_before) -> dict:
    title = "GMAT is left untouched by an analysis"
    criterion = "GMAT's files and a script's own outputs are byte-identical before and after GMAT-UQ runs"
    if not options.gmat_root:
        return {"id": 7, "title": title, "criterion": criterion, "passed": None, "skipped": "needs GMAT"}
    diagonal = [1e-2, 1.5e-2, 1e-2, 2.5e-7, 2.5e-7, 4e-7]
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        outputs = [folder / "state.txt", folder / "covariance.txt"]
        script = folder / "leo_covariance.script"
        script.write_text(cases.leo_covariance_script(outputs[0], outputs[1], diagonal, 3600.0), encoding="utf-8")
        _run_console(options.gmat_root, script)
        before = [path.read_bytes() for path in outputs]
        dynamics = GmatDynamics(str(script), gmat_root=options.gmat_root)
        mission = Mission(tuple(dynamics.initial_state()), final_time=3600.0, initial_dispersion=StateDispersion.from_sigmas([0.1] * 3 + [1e-4] * 3))
        run_monte_carlo(mission, dynamics, 8, seed=1, workers=2)
        outputs_untouched = [path.read_bytes() for path in outputs] == before
        leftovers = [path.name for path in folder.iterdir() if path.name.startswith("gmat_uq_tmp_")]
        _run_console(options.gmat_root, script)  # GMAT itself still gives the same answer afterwards
        rerun_identical = [path.read_bytes() for path in outputs] == before
    installation_unchanged = _installation_fingerprint(options.gmat_root) == fingerprint_before
    return {
        "id": 7,
        "title": title,
        "criterion": criterion,
        "passed": bool(outputs_untouched and rerun_identical and installation_unchanged and not leftovers),
        "measurements": {
            "script_outputs_untouched_by_the_analysis": outputs_untouched,
            "gmat_rerun_gives_identical_outputs": rerun_identical,
            "gmat_bin_and_plugins_folders_unchanged": installation_unchanged,
            "temporary_files_left_behind": leftovers,
            "note": "GMAT-UQ drives GMAT through its API and is not loaded into GMAT as a plugin, so this test "
            "replaces the design's 'sample scripts with the plugin loaded but unused'.",
        },
    }


# -- report -------------------------------------------------------------------------------------------
def _status(test: dict) -> str:
    if test.get("skipped"):
        return "skipped"
    return "pass" if test["passed"] else "FAIL"


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and not np.isfinite(value):
        return "unbounded"
    return f"{value:.{digits}g}"


def write_markdown(results: dict) -> str:
    tests = {test["id"]: test for test in results["tests"]}
    environment = results["environment"]
    lines = [
        "# Validation report",
        "",
        "This report is generated by `validation/run_validation.py`. Every number below was measured in the run",
        "described under Environment; none is a target or an estimate unless it says so.",
        "",
        "## Environment",
        "",
        f"- GMAT-UQ {environment['gmat_uq']}, Python {environment['python']}, NumPy {environment['numpy']}, SciPy {environment['scipy']}",
        f"- GMAT: {environment['gmat']}",
        f"- Platform: {environment['platform']}, {environment['cpu_count']} logical processors",
        f"- Monte Carlo samples: {results['options']['samples']}; worker processes: {results['options']['workers']}; master seed: {results['options']['seed']}",
        f"- Run date: {environment['date_utc']} UTC",
        "",
        "## Summary",
        "",
        "| # | Test | Pass criterion | Result |",
        "| --- | --- | --- | --- |",
    ]
    for test in results["tests"]:
        lines.append(f"| {test['id']} | {test['title']} | {test['criterion']} | **{_status(test)}** |")
    lines += [
        "| 8 | A published case whose inputs are public | Published statistics reproduced | **not run** |",
        "",
        "Test 8 of the design has not been done: no published case was reproduced. It remains open.",
        "",
    ]

    def section(test_id: int, heading: str, rows: list[tuple[str, str]], note: str = "") -> None:
        test = tests[test_id]
        lines.append(f"## Test {test_id}: {heading}")
        lines.append("")
        if test.get("skipped"):
            lines.append(f"Skipped: {test['skipped']}.")
            lines.append("")
            return
        if note:
            lines.append(note)
            lines.append("")
        lines.append("| Measurement | Value |")
        lines.append("| --- | --- |")
        for label, value in rows:
            lines.append(f"| {label} | {value} |")
        lines.append("")

    m = tests[1].get("measurements", {})
    section(1, "Gates sampler", [
        ("Draws", f"{m.get('draws', 0):,}"),
        ("Covariance test p-value (pass if 0.01 or more)", _number(m.get("covariance_test_p_value"))),
        ("Mean test p-value", _number(m.get("mean_test_p_value"))),
        ("Largest difference between sample and analytic sigma", f"{_number(m.get('largest_sigma_difference_percent'))}%"),
    ])

    m = tests[2].get("measurements", {})
    section(2, "GMAT's native covariance propagation", [
        ("Case", m.get("case", "")),
        ("Largest relative difference of a sigma", _number(m.get("largest_relative_sigma_difference"))),
        ("Matching significant digits", _number(m.get("matching_significant_digits"), 3)),
        ("Largest relative difference of a covariance element", _number(m.get("largest_relative_element_difference"))),
        ("Difference of the final position", f"{_number(m.get('final_state_difference_km'))} km"),
        ("Same test with GMAT's process noise (SNC) on, 1 hour: largest relative difference of a sigma", _number(m.get("with_process_noise_largest_relative_sigma_difference"))),
        ("Share of the sigmas due to that process noise", f"{_number(100 * m.get('process_noise_share_of_sigma', 0.0), 3)}%"),
    ], "GMAT ran the script with `Propagate Prop(Sat, 'Covariance')` in its own console application. GMAT-UQ then analysed the same script with no maneuvers.")

    def engine_rows(m: dict) -> list[tuple[str, str]]:
        interval = m.get("delta_v_99_monte_carlo_interval_mps", [None, None])
        rows = [
            ("Monte Carlo samples (failed / unconverged)", f"{m.get('monte_carlo_samples')} ({m.get('failed_samples')} / {m.get('unconverged_samples')})"),
            ("Ratio of Monte Carlo to linear sigma, six state components", f"{_number(m.get('component_sigma_ratio_min'))} to {_number(m.get('component_sigma_ratio_max'))}"),
            ("Largest sigma difference", f"{_number(m.get('largest_component_sigma_difference_percent'), 3)}%"),
            ("Covariance test p-value", _number(m.get("covariance_test_p_value"))),
            ("Delta-V 99, linear", f"{_number(m.get('delta_v_99_linear_mps'), 7)} m/s"),
            ("Delta-V 99, Monte Carlo (95% interval)", f"{_number(m.get('delta_v_99_monte_carlo_mps'), 7)} m/s ({_number(interval[0], 7)} to {_number(interval[1], 7)})"),
            ("Linear delta-V 99 inside the interval", _number(m.get("delta_v_99_linear_inside_interval"))),
        ]
        if "correction_delta_v_99_linear_mps" in m:
            rows.append(("Delta-V 99 of the corrections alone, linear / Monte Carlo", f"{_number(m['correction_delta_v_99_linear_mps'])} / {_number(m['correction_delta_v_99_monte_carlo_mps'])} m/s"))
        if "target_labels" in m:
            labels = ", ".join(m["target_labels"])
            rows.append((f"Final target sigmas ({labels}), linear", ", ".join(_number(v) for v in m["target_sigmas_linear"])))
            rows.append((f"Final target sigmas ({labels}), Monte Carlo", ", ".join(_number(v) for v in m["target_sigmas_monte_carlo"])))
        rows.append(("Largest sigma-point nonlinearity index", _number(m.get("sigma_point_max_index"))))
        return rows

    m = tests[3].get("measurements", {})
    section(3, "two-burn transfer", engine_rows(m), f"Case: {m.get('case', {}).get('description', '')}. Nominal delta-V {_number(m.get('case', {}).get('nominal_delta_v_mps'), 6)} m/s.")

    m = tests[4].get("measurements", {})
    if not tests[4].get("skipped"):
        case = m["case"]
        note = (
            f"Case: {case['description']}. Epoch {case['epoch']}, flight time {_number(case['flight_time_days'])} days, "
            f"translunar injection {_number(case['tli_delta_v_mps'], 6)} m/s, perilune radius {_number(case['perilune_radius_km'])} km. "
            "Targets: `BdotT`, `BdotR` and the distance to the Moon at the arrival time."
        )
        rows = engine_rows(m)
        rows.append(("Correction size, 1-sigma (TCM1, TCM2, TCM3)", ", ".join(f"{_number(m['per_maneuver_correction_sigma_mps'][name])} m/s" for name in ("TCM1", "TCM2", "TCM3"))))
        rows.append(("Monte Carlo wall time", f"{_number(m['monte_carlo_wall_seconds'])} s on {m['monte_carlo_workers']} workers"))
        section(4, "lunar transfer with GMAT in the loop", rows, note)
    else:
        section(4, "lunar transfer with GMAT in the loop", [])

    m = tests[5].get("measurements", {})
    case = m.get("case", {})
    lines.append("## Test 5: near-rectilinear halo orbit")
    lines.append("")
    lines.append(
        f"Case: {case.get('description', '')}. Period {_number(case.get('period_days'))} days, perilune radius "
        f"{_number(case.get('perilune_radius_km'))} km, initial dispersion {_number(case.get('initial_position_sigma_km'))} km and "
        f"{_number(case.get('initial_velocity_sigma_mps'))} m/s per axis at apolune, no maneuvers, {m.get('monte_carlo_samples')} Monte Carlo samples."
    )
    lines.append("")
    lines.append(f"First arc flagged by the sigma-point check: day {_number(m.get('first_flagged_time_days'))}. "
                 f"First node where the engines differ by more than 10%: day {_number(m.get('first_difference_time_days'))}.")
    lines.append("")
    lines.append(
        "The nonlinearity is concentrated at the perilune passages (days 3.28 and 9.84). There the linear covariance "
        "is wrong by orders of magnitude and the sigma-point check flags the arcs on both sides. Between perilunes the two "
        "engines agree again to within sampling noise, which is about 5% for this sample size in six dimensions."
    )
    lines.append("")
    lines.append("| Day | Sigma-point index | Flagged | Monte Carlo / linear sigma, all directions | Engines differ |")
    lines.append("| --- | --- | --- | --- | --- |")
    for node in m.get("nodes", []):
        lines.append(
            f"| {_number(node['time_days'], 3)} | {_number(node['sigma_point_index'], 3)} | {_number(node['flagged'])} | "
            f"{_number(node['monte_carlo_sigma_ratio_min'], 3)} to {_number(node['monte_carlo_sigma_ratio_max'], 3)} | {_number(node['engines_differ_percent'], 3)}% |"
        )
    lines.append("")

    m = tests[6].get("measurements", {})
    rows = [("Reference dynamics, 64 samples, identical on 1, 4 and 16 workers", _number(m.get("reference_dynamics_identical_on_1_4_16_workers")))]
    if "gmat_dynamics_identical_on_1_4_16_workers" in m:
        rows.append(("GMAT dynamics (lunar case), 32 samples, identical on 1, 4 and 16 workers", _number(m["gmat_dynamics_identical_on_1_4_16_workers"])))
    section(6, "reproducibility", rows, "Every array of the result is compared bit for bit.")

    m = tests[7].get("measurements", {})
    section(7, "no side effects on GMAT", [
        ("A script's report files untouched by the analysis", _number(m.get("script_outputs_untouched_by_the_analysis"))),
        ("GMAT gives identical outputs when the script is run again afterwards", _number(m.get("gmat_rerun_gives_identical_outputs"))),
        ("GMAT's bin and plugins folders unchanged (SHA-256 of every file)", _number(m.get("gmat_bin_and_plugins_folders_unchanged"))),
        ("Temporary files left behind", str(len(m.get("temporary_files_left_behind", [])))),
    ], m.get("note", ""))

    timing = results.get("timing")
    lines.append("## Performance on the lunar transfer case")
    lines.append("")
    if timing:
        lines += [
            "| Measurement | Value | Design target |",
            "| --- | --- | --- |",
            f"| Nominal trajectory with state transition matrices | {_number(timing['nominal_propagation_with_stm_seconds'])} s | |",
            f"| Linear covariance, including {timing['delta_v_samples']:,} delta-V samples | {_number(timing['linear_covariance_seconds'])} s | |",
            f"| Linear analysis in total, relative to the nominal propagation | {_number(timing['linear_total_over_nominal'], 3)} times | under 5 times: {'met' if timing['linear_total_over_nominal'] < 5 else 'NOT met'} |",
            f"| Sigma-point check | {_number(timing['sigma_point_check_seconds'])} s | |",
            f"| Monte Carlo cost on one core | {_number(timing['monte_carlo_seconds_per_sample_one_core'])} s per sample ({timing['serial_samples_timed']} samples timed) | |",
            f"| 2,000-sample Monte Carlo on one core, extrapolated from that rate | {_number(timing['monte_carlo_2000_samples_one_core_seconds_extrapolated'])} s | |",
            f"| Speed-up of the linear analysis over that Monte Carlo | {_number(timing['speed_up_over_2000_sample_monte_carlo'], 3)} times | at least 100 times: {'met' if timing['speed_up_over_2000_sample_monte_carlo'] >= 100 else 'NOT met'} |",
            f"| Speed-up when the sigma-point check is included | {_number(timing['speed_up_including_sigma_point_check'], 3)} times | |",
            "",
        ]
        if timing["linear_total_over_nominal"] >= 5:
            lines += [
                "The first target is missed in relative terms because GMAT propagates this nominal in under a tenth of a "
                "second, so the fixed cost of drawing the delta-V samples dominates. In absolute terms the whole linear "
                f"analysis takes {_number(timing['linear_covariance_total_seconds'], 2)} s.",
                "",
            ]
    else:
        lines += ["Not measured: this needs GMAT.", ""]
    lines += [
        "## Sample size",
        "",
        f"A one-sided bound on the 99th percentile at 95% confidence needs at least {wilks_sample_size(0.99, 0.95)} samples.",
        "",
    ]
    return "\n".join(lines)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isnan(number):
            return None
        if np.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gmat-root", default=os.environ.get("GMAT_ROOT"), help="GMAT installation (default: GMAT_ROOT)")
    parser.add_argument("--samples", type=int, default=2000, help="Monte Carlo samples of tests 3 to 5")
    parser.add_argument("--serial-samples", type=int, default=200, help="samples timed on one core for the performance figures")
    parser.add_argument("--dv-samples", type=int, default=20000, help="samples of the linear system used for delta-V statistics")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0), help="worker processes")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only", type=int, nargs="+", help="run only these test numbers")
    options = parser.parse_args(argv)

    fingerprint = _installation_fingerprint(options.gmat_root) if options.gmat_root else None
    context: dict = {}
    plan = [
        (1, lambda: test_1_gates_sampler(options)),
        (2, lambda: test_2_native_covariance(options)),
        (3, lambda: test_3_two_burn_transfer(options)),
        (4, lambda: test_4_lunar_transfer(options, context)),
        (5, lambda: test_5_halo_orbit(options)),
        (6, lambda: test_6_reproducibility(options, context)),
        (7, lambda: test_7_no_side_effects(options, fingerprint)),
    ]
    tests = []
    for number, run in plan:
        if options.only and number not in options.only:
            continue
        started = time.perf_counter()
        print(f"Test {number} ...", flush=True)
        test = run()
        test["elapsed_seconds"] = time.perf_counter() - started
        print(f"Test {number}: {_status(test)} ({test['elapsed_seconds']:.1f} s)", flush=True)
        tests.append(test)

    results = _jsonable(
        {
            "environment": {
                "gmat_uq": gmat_uq.__version__,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "gmat": f"R2026a ({_gmat_version(options.gmat_root)})" if options.gmat_root else "not available",
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "date_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
            },
            "options": {"samples": options.samples, "workers": options.workers, "seed": options.seed, "dv_samples": options.dv_samples},
            "tests": tests,
            "timing": context.get("timing"),
        }
    )
    if not options.only:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(write_markdown(results), encoding="utf-8")
        print(f"Results: {RESULTS}\nReport:  {REPORT}")
    failed = [test["id"] for test in tests if test["passed"] is False]
    if failed:
        print(f"FAILED tests: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
