"""Shared fixtures and reference missions for the GMAT-UQ test suite."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from gmat_uq import (
    Maneuver,
    ManeuverErrorModel,
    Mission,
    NavigationErrorModel,
    ParameterTarget,
    StateDispersion,
    TwoBody,
)
from gmat_uq.dynamics.ode import EARTH_MU

DATA = Path(__file__).parent / "data"


def transfer_mission(scale: float = 1.0, navigation: bool = True, execution: bool = True, corrections: bool = True):
    """Two-burn transfer between circular orbits with two position-targeted corrections.

    ``scale`` multiplies every uncertainty source, which lets tests move
    between the linear regime (small scale) and a mildly nonlinear one.
    """
    mu = EARTH_MU
    r1, r2 = 7000.0, 12000.0
    v1 = np.sqrt(mu / r1)
    transfer_sma = 0.5 * (r1 + r2)
    v_transfer = np.sqrt(mu * (2.0 / r1 - 1.0 / transfer_sma))
    half_period = np.pi * np.sqrt(transfer_sma**3 / mu)
    tilt = 0.3
    direction = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    initial_state = (r1, 0.0, 0.0, *(v1 * direction))
    burn_time = 600.0
    arrival = burn_time + half_period
    # The burn is tangential: along the velocity of the circular orbit at the burn time.
    angle = np.sqrt(mu / r1**3) * burn_time
    burn_direction = -np.sin(angle) * np.array([1.0, 0.0, 0.0]) + np.cos(angle) * direction

    gates = ManeuverErrorModel(1e-5 * scale, 0.005 * scale, 2e-5 * scale, 0.003 * scale) if execution else None
    knowledge = NavigationErrorModel.from_sigmas([0.05 * scale] * 3 + [5e-5 * scale] * 3) if navigation else None
    maneuvers = [Maneuver("TOI", burn_time, tuple((v_transfer - v1) * burn_direction), error_model=gates)]
    if corrections:
        for name, fraction in (("TCM1", 0.3), ("TCM2", 0.8)):
            maneuvers.append(
                Maneuver(
                    name,
                    burn_time + fraction * half_period,
                    (0.0, 0.0, 0.0),
                    error_model=gates,
                    target=ParameterTarget.position(),
                    target_time=arrival,
                    navigation=knowledge,
                )
            )
    mission = Mission(
        initial_state,
        final_time=arrival,
        maneuvers=tuple(maneuvers),
        initial_dispersion=StateDispersion.from_sigmas([0.5 * scale] * 3 + [5e-4 * scale] * 3),
    )
    return mission, TwoBody(mu)


@pytest.fixture
def transfer():
    return transfer_mission()


@pytest.fixture(scope="session")
def gmat_root() -> str:
    """Path of a GMAT installation, or skip the test when none is configured."""
    root = os.environ.get("GMAT_ROOT")
    if not root or not (Path(root) / "bin" / "gmatpy").is_dir():
        pytest.skip("GMAT_ROOT is not set to a GMAT installation")
    return root
