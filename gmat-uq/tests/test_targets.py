"""Target functions: orbital elements, B-plane, Jacobians."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from gmat_uq import FunctionTarget, ParameterTarget
from gmat_uq.dynamics.ode import EARTH_MU
from gmat_uq.targets import PARAMETERS, b_plane, cartesian_to_keplerian


def keplerian_to_cartesian(sma, ecc, inc, raan, aop, ta, mu=EARTH_MU):
    """Independent conversion used to build states with known elements (angles in degrees)."""
    inc, raan, aop, ta = np.radians([inc, raan, aop, ta])
    p = sma * (1 - ecc**2)
    r = p / (1 + ecc * np.cos(ta))
    perifocal_r = np.array([r * np.cos(ta), r * np.sin(ta), 0.0])
    perifocal_v = np.sqrt(mu / p) * np.array([-np.sin(ta), ecc + np.cos(ta), 0.0])

    def rot3(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rot1(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    rotation = rot3(raan) @ rot1(inc) @ rot3(aop)
    return np.concatenate([rotation @ perifocal_r, rotation @ perifocal_v])


@pytest.mark.parametrize(
    "elements",
    [
        (8000.0, 0.1, 30.0, 40.0, 50.0, 60.0),
        (26560.0, 0.74, 63.4, 300.0, 270.0, 200.0),
        (7000.0, 0.01, 98.0, 359.0, 1.0, 350.0),
        (-50000.0, 1.6, 20.0, 120.0, 80.0, 310.0),  # hyperbola
    ],
)
def test_keplerian_elements_round_trip(elements):
    state = keplerian_to_cartesian(*elements)
    result = cartesian_to_keplerian(state, EARTH_MU)
    for name, expected in zip(("SMA", "ECC", "INC", "RAAN", "AOP", "TA"), elements):
        if name in ("RAAN", "AOP", "TA"):
            difference = (result[name] - expected + 180.0) % 360.0 - 180.0
            assert abs(difference) < 1e-8
        else:
            assert np.isclose(result[name], expected, rtol=1e-10)


def test_scalar_parameters():
    state = keplerian_to_cartesian(9000.0, 0.2, 45.0, 10.0, 20.0, 30.0)
    target = ParameterTarget(("RMAG", "VMAG", "RadPer", "RadApo", "C3Energy", "x", "vz"), EARTH_MU)
    values = dict(zip(target.labels, target(state)))
    assert np.isclose(values["RMAG"], np.linalg.norm(state[:3]))
    assert np.isclose(values["VMAG"], np.linalg.norm(state[3:]))
    assert np.isclose(values["RadPer"], 9000.0 * 0.8)
    assert np.isclose(values["RadApo"], 9000.0 * 1.2)
    assert np.isclose(values["C3Energy"], -EARTH_MU / 9000.0)
    assert values["X"] == state[0] and values["VZ"] == state[5]
    assert target.labels == ("RMAG", "VMAG", "RadPer", "RadApo", "C3Energy", "X", "VZ")


def test_b_plane_of_an_equatorial_flyby():
    # Hyperbola in the equatorial plane, counter-clockwise: B lies along T, with magnitude b.
    periapsis, v_infinity = 7000.0, 3.0
    sma = -EARTH_MU / v_infinity**2
    ecc = 1 - periapsis / sma
    state = keplerian_to_cartesian(sma, ecc, 0.0, 0.0, 30.0, -100.0)
    b_t, b_r = b_plane(state, EARTH_MU)
    impact_parameter = periapsis * np.sqrt(1 + 2 * EARTH_MU / (periapsis * v_infinity**2))
    assert np.isclose(abs(b_t), impact_parameter, rtol=1e-10)
    assert abs(b_r) < 1e-6
    assert np.isclose(np.hypot(b_t, b_r), abs(sma) * np.sqrt(ecc**2 - 1))


def test_b_plane_is_constant_along_the_hyperbola_and_matches_its_definition():
    sma, ecc = -30000.0, 1.4
    first = keplerian_to_cartesian(sma, ecc, 35.0, 80.0, 110.0, -120.0)
    second = keplerian_to_cartesian(sma, ecc, 35.0, 80.0, 110.0, 40.0)
    assert np.allclose(b_plane(first, EARTH_MU), b_plane(second, EARTH_MU), rtol=1e-10)
    # Independent check: B is the point where the incoming asymptote pierces the plane through
    # the body centre that is perpendicular to it.
    b_t, b_r = b_plane(first, EARTH_MU)
    far = keplerian_to_cartesian(sma, ecc, 35.0, 80.0, 110.0, -np.degrees(np.arccos(-1 / ecc)) + 1e-4)
    s_hat = far[3:] / np.linalg.norm(far[3:])
    t_hat = np.cross(s_hat, [0, 0, 1.0])
    t_hat /= np.linalg.norm(t_hat)
    r_hat = np.cross(s_hat, t_hat)
    miss = far[:3] - (far[:3] @ s_hat) * s_hat
    assert np.isclose(miss @ t_hat, b_t, rtol=1e-4)
    assert np.isclose(miss @ r_hat, b_r, rtol=1e-4)


def test_b_plane_rejects_closed_orbits():
    with pytest.raises(ValueError, match="hyperbolic"):
        b_plane(keplerian_to_cartesian(8000.0, 0.1, 10.0, 0.0, 0.0, 0.0), EARTH_MU)


def test_position_target_jacobian_is_exact():
    target = ParameterTarget.position()
    state = np.array([7000.0, -100.0, 50.0, 1.0, 7.0, 0.5])
    expected = np.hstack([np.eye(3), np.zeros((3, 3))])
    assert np.allclose(target.jacobian(state), expected, atol=1e-9)
    assert target.size == 3


def test_parameter_jacobian_matches_an_analytic_derivative():
    state = keplerian_to_cartesian(9000.0, 0.2, 45.0, 10.0, 20.0, 30.0)
    jacobian = ParameterTarget(("RMAG", "C3Energy"), EARTH_MU).jacobian(state)
    r = np.linalg.norm(state[:3])
    assert np.allclose(jacobian[0], np.concatenate([state[:3] / r, np.zeros(3)]), atol=1e-8)
    expected = np.concatenate([2 * EARTH_MU * state[:3] / r**3, 2 * state[3:]])
    assert np.allclose(jacobian[1], expected, rtol=1e-6)


def test_periodic_angles_wrap_in_residuals_and_jacobians():
    target = ParameterTarget(("RAAN", "INC"), EARTH_MU)
    just_below = keplerian_to_cartesian(8000.0, 0.1, 30.0, 359.9999999, 50.0, 60.0)
    just_above = keplerian_to_cartesian(8000.0, 0.1, 30.0, 0.0000001, 50.0, 60.0)
    residual = target.residual(just_above, target(just_below))
    assert abs(residual[0] - 2e-7) < 1e-9  # not -360
    assert np.all(np.isfinite(target.jacobian(just_below)))
    assert np.abs(target.jacobian(just_below)).max() < 1e3  # no 360-degree jump in the differences


def test_parameter_target_validation():
    with pytest.raises(ValueError, match="unknown parameter"):
        ParameterTarget(("Altitude",), EARTH_MU)
    with pytest.raises(ValueError, match="needs the gravitational parameter"):
        ParameterTarget(("SMA",))
    with pytest.raises(ValueError, match="repeat"):
        ParameterTarget(("X", "x"))
    with pytest.raises(ValueError, match="at least one"):
        ParameterTarget(())
    with pytest.raises(ValueError, match="positive"):
        ParameterTarget(("SMA",), -1.0)
    assert ParameterTarget("RMAG").names == ("RMAG",)
    assert ParameterTarget.b_plane(EARTH_MU).labels == ("BdotT", "BdotR")
    assert set(ParameterTarget.position().describe()["names"]) == {"X", "Y", "Z"}
    assert {"BdotT", "BdotR", "SMA", "X"} <= set(PARAMETERS)


def _radius_and_speed(state):
    return [np.linalg.norm(state[:3]), np.linalg.norm(state[3:])]


def test_function_target():
    target = FunctionTarget(_radius_and_speed, ("radius", "speed"))
    state = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 2.0])
    assert np.allclose(target(state), [5.0, 2.0])
    assert target.size == 2
    assert np.allclose(target.jacobian(state)[0, :3], [0.6, 0.8, 0.0], atol=1e-7)
    assert pickle.loads(pickle.dumps(target))(state)[0] == 5.0
    with pytest.raises(ValueError, match="label"):
        FunctionTarget(_radius_and_speed, ())
    with pytest.raises(ValueError, match="shape"):
        FunctionTarget(_radius_and_speed, ("only_one",))(state)


def test_parameters_relative_to_another_body():
    moon = np.array([380000.0, 20000.0, 5000.0, -0.05, 1.0, 0.1])
    moon_mu = 4902.8
    relative = keplerian_to_cartesian(-9000.0, 1.5, 150.0, 30.0, 60.0, -80.0, moon_mu)
    state = moon + relative
    target = ParameterTarget(("BdotT", "BdotR", "RMAG", "ECC"), moon_mu, tuple(moon))
    direct = ParameterTarget(("BdotT", "BdotR", "RMAG", "ECC"), moon_mu)
    assert np.allclose(target(state), direct(relative), rtol=1e-12)
    assert np.isclose(target(state)[3], 1.5)
    assert np.allclose(target.jacobian(state), direct.jacobian(relative), rtol=1e-5, atol=1e-7)
    assert target.describe()["origin"] == list(moon)
    assert ParameterTarget.b_plane(moon_mu, moon).origin == tuple(moon)
    with pytest.raises(ValueError, match="six finite"):
        ParameterTarget(("RMAG",), origin=(1.0, 2.0, 3.0))
