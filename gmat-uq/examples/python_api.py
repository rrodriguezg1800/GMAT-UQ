"""GMAT-UQ from Python: a transfer between circular orbits, analysed three ways.

Run it from anywhere after ``pip install -e gmat-uq``::

    python gmat-uq/examples/python_api.py

No GMAT installation is needed: the dynamics are the built-in two-body model.
To use GMAT instead, replace ``TwoBody(mu)`` with
``GmatDynamics("your_mission.script")`` and set the GMAT_ROOT environment variable.
"""

import numpy as np

from gmat_uq import (
    Maneuver,
    ManeuverErrorModel,
    Mission,
    NavigationErrorModel,
    ParameterTarget,
    StateDispersion,
    TwoBody,
    run_lincov,
    run_monte_carlo,
    run_sigma_point_check,
)
from gmat_uq.consistency import compare_engines


def build_mission():
    mu = 398600.4415
    r1, r2 = 7000.0, 12000.0
    sma = 0.5 * (r1 + r2)
    v1 = np.sqrt(mu / r1)
    v_departure = np.sqrt(mu * (2 / r1 - 1 / sma))
    arrival = np.pi * np.sqrt(sma**3 / mu)  # half a revolution of the transfer orbit

    engine = ManeuverErrorModel(
        fixed_magnitude_sigma=1e-5,  # 1 cm/s
        proportional_magnitude_sigma=0.003,  # 0.3 %
        fixed_pointing_sigma=1e-5,  # 1 cm/s per axis
        proportional_pointing_sigma=0.005,  # 5 mrad per axis
    )
    tracking = NavigationErrorModel.from_sigmas([0.05] * 3 + [5e-5] * 3)  # 50 m, 5 cm/s
    position = ParameterTarget.position()

    mission = Mission(
        initial_state=(r1, 0.0, 0.0, 0.0, v1, 0.0),
        final_time=arrival,
        maneuvers=(
            Maneuver("Burn1", 0.0, dv=(0.0, v_departure - v1, 0.0), error_model=engine),
            Maneuver("TCM1", 0.25 * arrival, error_model=engine, target=position, target_time=arrival, navigation=tracking),
            Maneuver("TCM2", 0.75 * arrival, error_model=engine, target=position, target_time=arrival, navigation=tracking),
        ),
        initial_dispersion=StateDispersion.from_sigmas([0.2] * 3 + [2e-4] * 3),
    )
    return mission, TwoBody(mu)


def main():
    mission, dynamics = build_mission()

    # 1. Linear covariance: dispersions and delta-V statistics from one nominal run.
    lincov = run_lincov(mission, dynamics, seed=1)
    delta_v = lincov.dv_statistics()
    print("Linear covariance")
    print(f"  final position sigma (km): {np.round(lincov.sigmas()[:3], 4)}")
    print(f"  nominal delta-V:  {1e3 * delta_v['nominal_total']:.2f} m/s")
    print(f"  delta-V 99:       {1e3 * delta_v['total']['quantile']['value']:.2f} m/s")

    # 2. Sigma-point check: is the linearization trustworthy on every arc?
    check = run_sigma_point_check(lincov, dynamics)
    print(f"Sigma-point check: largest index {check.max_index:.3f}, linear results {'trusted' if check.linear_ok else 'NOT trusted'}")

    # 3. Closed-loop Monte Carlo with the full dynamics, on four worker processes.
    monte_carlo = run_monte_carlo(mission, dynamics, 1000, seed=1, workers=4, nominal=lincov.nominal)
    quantile = monte_carlo.dv_statistics()["total"]["quantile"]
    print("Monte Carlo (1000 samples)")
    print(f"  final position sigma (km): {np.round(np.sqrt(np.diag(monte_carlo.final_covariance))[:3], 4)}")
    print(f"  delta-V 99:       {1e3 * quantile['value']:.2f} m/s (95% interval {1e3 * quantile['lower']:.2f} to {1e3 * quantile['upper']:.2f})")

    report = compare_engines(lincov, monte_carlo)
    print(f"Engines agree: {report.agrees}")


if __name__ == "__main__":  # required on Windows when worker processes are used
    main()
