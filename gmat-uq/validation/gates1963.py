# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Reproduction of C. R. Gates, "A Simplified Model of Midcourse Maneuver Execution
Errors", JPL Technical Report 32-504, 15 October 1963 (NASA NTRS 19640003365).

Gates builds the execution error of a commanded velocity change v (magnitude V)
from four independent errors:

* shutoff error     e_s = s v,         s ~ N(0, sigma_s^2)          (proportional, along v)
* resolution error  e_r = r v / V,     r ~ N(0, sigma_r^2)          (fixed, along v)
* pointing error    e_p = u x v,       u ~ N(0, sigma_p^2 I), rad   (proportional, across v)
* autopilot error   e_a = w x v / V,   w ~ N(0, sigma_a^2 I)        (fixed, across v)

and derives the second moment of e = e_s + e_r + e_p + e_a when the command
itself is random:

    L_e = sigma_s^2 L_v + sigma_r^2 G + sigma_p^2 (Vbar^2 I - L_v) + sigma_a^2 (I - G)

with L_v = E[v v^T], G = E[v v^T / V^2] and Vbar^2 = E[V^2].  He also states two
properties: equal proportional errors (sigma_s = sigma_p) give a spherical
contribution sigma_s^2 Vbar^2 I, and so do equal fixed errors (sigma_r = sigma_a).

GMAT-UQ's ManeuverErrorModel uses the names fixed/proportional magnitude and
pointing sigma: sigma_r, sigma_s, sigma_a and sigma_p respectively.
"""

from __future__ import annotations

import numpy as np

from gmat_uq import ManeuverErrorModel
from gmat_uq.stats import compare_covariance

REFERENCE = (
    'C. R. Gates, "A Simplified Model of Midcourse Maneuver Execution Errors", '
    "JPL Technical Report 32-504, 1963 (NASA NTRS 19640003365)"
)


def gates_construction(commands: np.ndarray, sigmas: dict, rng: np.random.Generator) -> np.ndarray:
    """Execution errors built exactly as in the report, for each commanded vector (rows)."""
    n = commands.shape[0]
    magnitude = np.linalg.norm(commands, axis=1, keepdims=True)
    s = rng.normal(0.0, sigmas["shutoff"], (n, 1))
    r = rng.normal(0.0, sigmas["resolution"], (n, 1))
    u = rng.normal(0.0, sigmas["pointing"], (n, 3))
    w = rng.normal(0.0, sigmas["autopilot"], (n, 3))
    return s * commands + r * commands / magnitude + np.cross(u, commands) + np.cross(w, commands / magnitude)


def gates_second_moment(commands: np.ndarray, sigmas: dict) -> np.ndarray:
    """The report's closed form, with L_v, G and Vbar^2 taken over the given commands."""
    second = commands.T @ commands / commands.shape[0]
    units = commands / np.linalg.norm(commands, axis=1, keepdims=True)
    direction = units.T @ units / commands.shape[0]
    identity = np.eye(3)
    return (
        sigmas["shutoff"] ** 2 * second
        + sigmas["resolution"] ** 2 * direction
        + sigmas["pointing"] ** 2 * (np.trace(second) * identity - second)
        + sigmas["autopilot"] ** 2 * (identity - direction)
    )


def model_for(sigmas: dict) -> ManeuverErrorModel:
    return ManeuverErrorModel(
        fixed_magnitude_sigma=sigmas["resolution"],
        proportional_magnitude_sigma=sigmas["shutoff"],
        fixed_pointing_sigma=sigmas["autopilot"],
        proportional_pointing_sigma=sigmas["pointing"],
    )


def reproduce(samples: int = 400_000, seed: int = 1963) -> dict:
    """Run every check; returns the measurements and whether each passed."""
    rng = np.random.default_rng(seed)
    # A correction whose command is itself random: mean 2 m/s, anisotropic spread (km/s).
    mean = np.array([0.002, -0.001, 0.0005])
    command_cov = np.diag([1.0e-6, 4.0e-7, 2.5e-7])
    command_cov[0, 1] = command_cov[1, 0] = 2.0e-7
    commands = rng.multivariate_normal(mean, command_cov, size=samples)
    sigmas = {"shutoff": 0.01, "resolution": 1e-5, "pointing": 0.02, "autopilot": 2e-5}
    model = model_for(sigmas)

    errors_gates = gates_construction(commands, sigmas, rng)
    errors_uq = model.apply(commands, rng.standard_normal(commands.shape))
    closed_form = gates_second_moment(commands, sigmas)
    expected_uq = model.expected_covariance(mean, command_cov)

    # 1. GMAT-UQ's sampler and the report's construction give the same distribution.
    construction = compare_covariance(errors_gates, closed_form, np.zeros(3))
    sampler = compare_covariance(errors_uq, closed_form, np.zeros(3))
    # 2. GMAT-UQ's expected covariance equals the report's closed form.
    relative = float(np.abs(expected_uq - closed_form).max() / np.abs(closed_form).max())

    # 3. The two published properties, for a fixed (non-random) command.
    command = np.array([[0.03, -0.04, 0.012]])
    magnitude_sq = float(command[0] @ command[0])
    proportional = model_for({"shutoff": 0.02, "resolution": 0.0, "pointing": 0.02, "autopilot": 0.0}).covariance(command[0])
    fixed = model_for({"shutoff": 0.0, "resolution": 3e-5, "pointing": 0.0, "autopilot": 3e-5}).covariance(command[0])
    property_proportional = float(np.abs(proportional - 0.02**2 * magnitude_sq * np.eye(3)).max() / (0.02**2 * magnitude_sq))
    property_fixed = float(np.abs(fixed - (3e-5) ** 2 * np.eye(3)).max() / (3e-5) ** 2)

    checks = {
        "report_construction_consistent_with_closed_form": construction.p_value >= 0.01,
        "gmat_uq_sampler_consistent_with_closed_form": sampler.p_value >= 0.01,
        "gmat_uq_expected_covariance_matches_closed_form": relative < 5e-3,
        "equal_proportional_errors_are_spherical": property_proportional < 1e-12,
        "equal_fixed_errors_are_spherical": property_fixed < 1e-12,
    }
    return {
        "reference": REFERENCE,
        "samples": samples,
        "construction_p_value": construction.p_value,
        "construction_largest_sigma_difference_percent": 100 * construction.max_sigma_deviation,
        "sampler_p_value": sampler.p_value,
        "sampler_largest_sigma_difference_percent": 100 * sampler.max_sigma_deviation,
        "expected_covariance_largest_relative_difference": relative,
        "spherical_property_proportional_error": property_proportional,
        "spherical_property_fixed_error": property_fixed,
        "checks": checks,
        "passed": all(checks.values()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(reproduce(), indent=2, default=float))
