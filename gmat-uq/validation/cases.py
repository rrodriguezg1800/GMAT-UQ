# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Reference missions of the GMAT-UQ validation campaign.

Every builder returns ``(mission, dynamics, info)``, where ``info`` is a small
dictionary of facts about the case that the report quotes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from gmat_uq import (
    CR3BP,
    GmatDynamics,
    Maneuver,
    ManeuverErrorModel,
    Mission,
    NavigationErrorModel,
    ParameterTarget,
    StateDispersion,
    TwoBody,
    design_maneuver,
)
from gmat_uq.dynamics.ode import EARTH_MU

HERE = Path(__file__).resolve().parent
CISLUNAR_SCRIPT = HERE / "scripts" / "cislunar.script"
LEO_COVARIANCE_TEMPLATE = HERE / "scripts" / "leo_covariance.script.in"


def leo_covariance_script(state_file, cov_file, diagonal, duration: float) -> str:
    """Text of the GMAT script that propagates a covariance natively (validation tests 2 and 7)."""
    text = LEO_COVARIANCE_TEMPLATE.read_text(encoding="utf-8")
    values = {
        "@STATE_FILE@": str(state_file),
        "@COV_FILE@": str(cov_file),
        "@DIAGONAL@": " ".join(repr(float(value)) for value in diagonal),
        "@DURATION@": repr(float(duration)),
    }
    for token, value in values.items():
        text = text.replace(token, value)
    return text


# -- validation test 3 ----------------------------------------------------------------------------
def two_burn_transfer():
    """Transfer from a 7000 km to a 12000 km circular orbit: two burns and two corrections.

    The first burn starts the transfer and the second one circularises at
    arrival.  Two statistical corrections target the arrival position.  Every
    uncertainty source is active: injection errors, execution errors on all
    four maneuvers and navigation errors at both corrections.
    """
    mu = EARTH_MU
    r1, r2 = 7000.0, 12000.0
    v1, v2 = np.sqrt(mu / r1), np.sqrt(mu / r2)
    sma = 0.5 * (r1 + r2)
    v_departure = np.sqrt(mu * (2.0 / r1 - 1.0 / sma))
    v_arrival = np.sqrt(mu * (2.0 / r2 - 1.0 / sma))
    half_period = np.pi * np.sqrt(sma**3 / mu)
    tilt = np.radians(28.5)
    along = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    radial = np.array([1.0, 0.0, 0.0])
    coast = 600.0
    arrival = coast + half_period
    angle = np.sqrt(mu / r1**3) * coast
    departure_direction = -np.sin(angle) * radial + np.cos(angle) * along
    arrival_direction = -departure_direction  # half a revolution later the velocity is reversed

    engine = ManeuverErrorModel(
        fixed_magnitude_sigma=1e-5,  # 1 cm/s
        proportional_magnitude_sigma=0.003,
        fixed_pointing_sigma=1e-5,
        proportional_pointing_sigma=0.005,
    )
    knowledge = NavigationErrorModel.from_sigmas([0.05] * 3 + [5e-5] * 3)  # 50 m, 5 cm/s
    position = ParameterTarget.position()
    initial_state = (r1, 0.0, 0.0, *(v1 * along))
    mission = Mission(
        initial_state,
        final_time=arrival,
        maneuvers=(
            Maneuver("Burn1", coast, tuple((v_departure - v1) * departure_direction), error_model=engine),
            Maneuver("TCM1", coast + 0.25 * half_period, error_model=engine, target=position, target_time=arrival, navigation=knowledge),
            Maneuver("TCM2", coast + 0.75 * half_period, error_model=engine, target=position, target_time=arrival, navigation=knowledge),
            Maneuver("Burn2", arrival, tuple((v2 - v_arrival) * arrival_direction), error_model=engine),
        ),
        initial_dispersion=StateDispersion.from_local(
            np.diag(np.array([0.2, 0.5, 0.2, 5e-4, 2e-4, 2e-4]) ** 2), np.array(initial_state), "RIC"
        ),
        # The final orbit is circular, where eccentricity is not differentiable, so the dispersion
        # is reported in elements that are smooth there.
        final_target=ParameterTarget(("SMA", "INC", "RAAN"), mu),
    )
    info = {
        "description": "7000 km to 12000 km circular, inclination 28.5 deg, reference two-body dynamics",
        "transfer_time_s": float(half_period),
        "nominal_delta_v_mps": float(1e3 * ((v_departure - v1) + (v2 - v_arrival))),
    }
    return mission, TwoBody(mu), info


# -- validation test 4 ----------------------------------------------------------------------------
def _departure_guess(dynamics: GmatDynamics, perigee_radius: float, flight_time_guess: float):
    """Two-body estimate of a translunar departure: state on the parking orbit, burn and flight time.

    The transfer ellipse lies in the Moon's orbit plane, with its apogee 3%
    beyond the Moon.  The flight time and the Moon's position at arrival
    depend on each other, so they are iterated to a fixed point.
    """
    mu = dynamics.mu
    flight_time = flight_time_guess
    for _ in range(8):
        moon = dynamics.body_state("Luna", flight_time)
        distance = np.linalg.norm(moon[:3])
        apogee = 1.03 * distance
        sma = 0.5 * (perigee_radius + apogee)
        ecc = (apogee - perigee_radius) / (apogee + perigee_radius)
        true_anomaly = np.arccos((sma * (1.0 - ecc**2) / distance - 1.0) / ecc)
        eccentric = 2.0 * np.arctan(np.sqrt((1.0 - ecc) / (1.0 + ecc)) * np.tan(0.5 * true_anomaly))
        flight_time = (eccentric - ecc * np.sin(eccentric)) * np.sqrt(sma**3 / mu)
    moon = dynamics.body_state("Luna", flight_time)
    to_moon = moon[:3] / np.linalg.norm(moon[:3])
    normal = np.cross(moon[:3], moon[3:])
    normal /= np.linalg.norm(normal)

    def rotate(vector, angle):  # about the orbit normal
        return vector * np.cos(angle) + np.cross(normal, vector) * np.sin(angle) + normal * (normal @ vector) * (1 - np.cos(angle))

    perigee_direction = rotate(to_moon, -true_anomaly)
    velocity_direction = np.cross(normal, perigee_direction)
    parking_speed = np.sqrt(mu / perigee_radius)
    perigee_speed = np.sqrt(mu * (2.0 / perigee_radius - 1.0 / sma))
    state = np.concatenate([perigee_radius * perigee_direction, parking_speed * velocity_direction])
    burn = (perigee_speed - parking_speed) * velocity_direction
    return state, burn, float(flight_time), moon, to_moon, np.cross(normal, to_moon)


def lunar_transfer(gmat_root=None, scale: float = 1.0):
    """Translunar injection from a 185 km parking orbit with three corrections to the lunar B-plane.

    The force model (Earth, Moon and Sun point masses) and the integrator come
    from ``scripts/cislunar.script`` and every propagation is done by GMAT.  The
    corrections hold ``BdotT``, ``BdotR`` and the distance to the Moon at a
    fixed arrival time, about seven hours before perilune.
    """
    dynamics = GmatDynamics(str(CISLUNAR_SCRIPT), gmat_root=gmat_root)
    moon_mu = dynamics.body_mu("Luna")
    state, burn_guess, arrival, moon, to_moon, transverse = _departure_guess(dynamics, 6563.14, 4.3 * 86400.0)

    # Aim 30000 km short of the Moon and 20000 km to the side, which gives a flyby well clear
    # of the surface.  The first solve uses a nearer aim point to stay away from the Moon's centre.
    position = ParameterTarget.position()
    burn = burn_guess
    for sideways in (9000.0, -20000.0):
        aim = moon[:3] - 30000.0 * to_moon + sideways * transverse
        solution = design_maneuver(dynamics, state, 0.0, position, arrival, aim, dv_guess=burn)
        if not solution.converged:
            raise RuntimeError("the translunar injection design did not converge")
        burn = solution.dv

    origin = tuple(moon)
    arrival_target = ParameterTarget(("BdotT", "BdotR", "RMAG"), moon_mu, origin)
    upper_stage = ManeuverErrorModel(1e-5 * scale, 5e-4 * scale, 1e-5 * scale, 5e-4 * scale)
    thrusters = ManeuverErrorModel(1e-6 * scale, 5e-3 * scale, 1e-6 * scale, 5e-3 * scale)

    def knowledge(position_sigma, velocity_sigma):
        return NavigationErrorModel.from_sigmas([position_sigma * scale] * 3 + [velocity_sigma * scale] * 3)

    injection = np.diag((np.array([0.5, 2.0, 0.5, 2e-3, 5e-4, 5e-4]) * scale) ** 2)
    mission = Mission(
        tuple(state),
        final_time=arrival,
        maneuvers=(
            Maneuver("TLI", 0.0, tuple(burn), error_model=upper_stage),
            Maneuver("TCM1", 6 * 3600.0, error_model=thrusters, target=arrival_target, target_time=arrival, navigation=knowledge(1.0, 1e-5)),
            Maneuver("TCM2", 48 * 3600.0, error_model=thrusters, target=arrival_target, target_time=arrival, navigation=knowledge(2.0, 1e-5)),
            Maneuver("TCM3", arrival - 18 * 3600.0, error_model=thrusters, target=arrival_target, target_time=arrival, navigation=knowledge(0.5, 5e-6)),
        ),
        initial_dispersion=StateDispersion.from_local(injection, state, "RIC"),
        final_target=ParameterTarget(("BdotT", "BdotR"), moon_mu, origin),
    )
    arrival_state = dynamics.propagate(state + np.concatenate([np.zeros(3), burn]), 0.0, arrival)
    flyby = ParameterTarget(("RadPer", "ECC", "RMAG", "C3Energy"), moon_mu, origin)(arrival_state)
    info = {
        "description": "185 km parking orbit to a lunar flyby; Earth, Moon and Sun point masses propagated by GMAT",
        "gmat_script": CISLUNAR_SCRIPT.name,
        "epoch": "01 Jan 2026 12:00:00.000 UTC",
        "flight_time_days": arrival / 86400.0,
        "tli_delta_v_mps": float(1e3 * np.linalg.norm(burn)),
        "perilune_radius_km": float(flyby[0]),
        "lunar_eccentricity": float(flyby[1]),
        "moon_distance_at_target_km": float(flyby[2]),
        "v_infinity_kmps": float(np.sqrt(flyby[3])),
    }
    return mission, dynamics, info


# -- validation test 5 ----------------------------------------------------------------------------
def _halo_orbit(dynamics: CR3BP, x0: float, z0: float, vy0: float):
    """Differentially correct a halo orbit of the CR3BP, holding ``z0`` fixed.

    The orbit is symmetric about the x-z plane, so it is periodic when it
    crosses that plane perpendicularly: ``vx = vz = 0`` at ``y = 0``.
    """

    def crossing(t, y):
        return y[1]

    crossing.terminal = True
    crossing.direction = 1.0 if vy0 < 0.0 else -1.0  # the return crossing after half a revolution

    def augmented(t, y):
        derivative = np.empty(42)
        derivative[:6] = dynamics.rhs(t, y[:6])
        derivative[6:] = (dynamics.jacobian(t, y[:6]) @ y[6:].reshape(6, 6)).ravel()
        return derivative

    for _ in range(40):
        start = np.concatenate([[x0, 0.0, z0, 0.0, vy0, 0.0], np.eye(6).ravel()])
        solution = solve_ivp(augmented, (0.0, 10.0), start, method="DOP853", rtol=1e-13, atol=1e-13, events=crossing)
        if solution.t_events[0].size == 0:
            raise RuntimeError("the halo orbit never returned to the x-z plane")
        half_period = float(solution.t_events[0][0])
        end = solution.y_events[0][0]
        state, stm = end[:6], end[6:].reshape(6, 6)
        derivative = dynamics.rhs(half_period, state)
        miss = np.array([state[3], state[5]])
        if np.abs(miss).max() < 1e-12:
            return np.array([x0, 0.0, z0, 0.0, vy0, 0.0]), 2.0 * half_period
        # Partials of (vx, vz) at the crossing with respect to (x0, vy0), including the change
        # of the crossing time needed to stay on y = 0.
        partials = np.array([[stm[3, 0], stm[3, 4]], [stm[5, 0], stm[5, 4]]]) - np.outer(
            [derivative[3], derivative[5]], [stm[1, 0], stm[1, 4]]
        ) / state[4]
        correction = np.linalg.solve(partials, -miss)
        x0 += correction[0]
        vy0 += correction[1]
    raise RuntimeError("the halo-orbit correction did not converge")


def near_rectilinear_halo(revolutions: float = 2.0, nodes_per_revolution: int = 8):
    """Coast along an Earth-Moon L2 near-rectilinear halo orbit, starting at apolune.

    The orbit is found by differential correction from a published-style
    starting guess near the 9:2 resonant NRHO.  There are no maneuvers: the
    case isolates the dynamics, so any difference between the linear
    covariance and Monte Carlo is a linearization error.
    """
    dynamics = CR3BP()
    state, period = _halo_orbit(dynamics, 1.0221, -0.1821, -0.1033)
    length_unit_km = 384400.0
    time_unit_s = 375190.0
    velocity_unit = length_unit_km / time_unit_s
    position_sigma = 10.0 / length_unit_km  # 10 km
    velocity_sigma = 1e-4 / velocity_unit  # 10 cm/s
    count = int(round(revolutions * nodes_per_revolution))
    times = tuple(period * index / nodes_per_revolution for index in range(1, count))
    mission = Mission(
        tuple(state),
        final_time=period * count / nodes_per_revolution,
        initial_dispersion=StateDispersion.from_sigmas([position_sigma] * 3 + [velocity_sigma] * 3),
        report_times=times,
    )
    moon = np.array([1.0 - dynamics.mass_ratio, 0.0, 0.0])
    half = dynamics.propagate(state, 0.0, 0.5 * period)
    closure = dynamics.propagate(state, 0.0, period) - state
    info = {
        "description": "Earth-Moon L2 southern near-rectilinear halo orbit in the circular restricted three-body problem",
        "period_days": period * time_unit_s / 86400.0,
        "apolune_radius_km": float(np.linalg.norm(state[:3] - moon) * length_unit_km),
        "perilune_radius_km": float(np.linalg.norm(half[:3] - moon) * length_unit_km),
        "periodicity_error": float(np.abs(closure).max()),
        "initial_position_sigma_km": 10.0,
        "initial_velocity_sigma_mps": 0.1,
        "revolutions": revolutions,
        "length_unit_km": length_unit_km,
    }
    return mission, dynamics, info
