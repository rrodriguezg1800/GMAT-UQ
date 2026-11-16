"""Reference dynamics: conservation laws, state transition matrices, Jacobians."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from gmat_uq import CR3BP, TwoBody, TwoBodyJ2
from gmat_uq.dynamics import Dynamics, PropagationError, finite_difference_stm
from gmat_uq.dynamics.ode import EARTH_J2, EARTH_MU, EARTH_RADIUS

STATE = np.array([7000.0, 200.0, -300.0, 0.4, 7.2, 1.5])


def _numeric_jacobian(dynamics, state):
    jacobian = np.empty((6, 6))
    for column in range(6):
        step = 1e-6 * max(abs(state[column]), 1.0)
        plus, minus = state.copy(), state.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (dynamics.rhs(0.0, plus) - dynamics.rhs(0.0, minus)) / (2 * step)
    return jacobian


def test_two_body_circular_orbit_closes_after_one_period():
    dynamics = TwoBody()
    radius = 8000.0
    speed = np.sqrt(EARTH_MU / radius)
    state = np.array([radius, 0.0, 0.0, 0.0, speed, 0.0])
    period = 2 * np.pi * np.sqrt(radius**3 / EARTH_MU)
    final = dynamics.propagate(state, 0.0, period)
    assert np.allclose(final[:3], state[:3], atol=1e-6)
    assert np.allclose(final[3:], state[3:], atol=1e-9)


def test_two_body_conserves_energy_and_angular_momentum():
    dynamics = TwoBody()
    final = dynamics.propagate(STATE, 0.0, 20000.0)
    energy = lambda s: 0.5 * s[3:] @ s[3:] - EARTH_MU / np.linalg.norm(s[:3])
    assert np.isclose(energy(final), energy(STATE), rtol=1e-11)
    assert np.allclose(np.cross(final[:3], final[3:]), np.cross(STATE[:3], STATE[3:]), rtol=1e-11)


def test_propagation_is_reversible_and_identity_for_zero_duration():
    dynamics = TwoBodyJ2()
    final = dynamics.propagate(STATE, 100.0, 9000.0)
    back = dynamics.propagate(final, 9000.0, 100.0)
    assert np.allclose(back[:3], STATE[:3], atol=1e-6)
    assert np.allclose(back[3:], STATE[3:], atol=1e-9)
    same, stm = dynamics.propagate_stm(STATE, 5.0, 5.0)
    assert np.array_equal(same, STATE)
    assert np.array_equal(stm, np.eye(6))


@pytest.mark.parametrize("dynamics", [TwoBody(), TwoBodyJ2()])
def test_variational_stm_matches_finite_differences(dynamics):
    final, stm = dynamics.propagate_stm(STATE, 0.0, 6000.0)
    assert np.allclose(final, dynamics.propagate(STATE, 0.0, 6000.0), rtol=1e-11, atol=1e-9)
    reference = finite_difference_stm(dynamics.propagate, STATE, 0.0, 6000.0)
    assert np.abs(stm - reference).max() / np.abs(stm).max() < 1e-6


def test_two_body_stm_is_symplectic():
    _, stm = TwoBody().propagate_stm(STATE, 0.0, 12000.0)
    symplectic = np.block([[np.zeros((3, 3)), np.eye(3)], [-np.eye(3), np.zeros((3, 3))]])
    assert np.allclose(stm.T @ symplectic @ stm, symplectic, atol=1e-6)
    assert np.isclose(np.linalg.det(stm), 1.0, rtol=1e-6)


@pytest.mark.parametrize("dynamics", [TwoBody(), TwoBodyJ2(), CR3BP()])
def test_analytic_jacobian_matches_numeric(dynamics):
    state = np.array([0.82, 0.05, 0.03, 0.01, 0.2, -0.02]) if isinstance(dynamics, CR3BP) else STATE
    analytic = dynamics.jacobian(0.0, state)
    numeric = _numeric_jacobian(dynamics, state)
    scale = np.abs(analytic).max()
    assert np.abs(analytic - numeric).max() / scale < 1e-7


def test_j2_nodal_regression_matches_the_secular_rate():
    dynamics = TwoBodyJ2()
    sma, inclination = 7200.0, np.radians(50.0)
    speed = np.sqrt(EARTH_MU / sma)
    state = np.array([sma, 0.0, 0.0, 0.0, speed * np.cos(inclination), speed * np.sin(inclination)])
    mean_motion = np.sqrt(EARTH_MU / sma**3)
    expected_rate = -1.5 * mean_motion * EARTH_J2 * (EARTH_RADIUS / sma) ** 2 * np.cos(inclination)
    duration = 10 * 2 * np.pi / mean_motion
    final = dynamics.propagate(state, 0.0, duration)

    def node(s):
        h = np.cross(s[:3], s[3:])
        return np.arctan2(h[0], -h[1])

    drift = (node(final) - node(state) + np.pi) % (2 * np.pi) - np.pi
    assert np.isclose(drift, expected_rate * duration, rtol=0.02)


def test_cr3bp_conserves_the_jacobi_constant_and_holds_l4():
    dynamics = CR3BP()
    state = np.array([0.85, 0.0, 0.1, 0.0, 0.18, 0.0])
    final = dynamics.propagate(state, 0.0, 3.0)
    assert np.isclose(dynamics.jacobi_constant(final), dynamics.jacobi_constant(state), rtol=1e-10)
    l4 = np.array([0.5 - dynamics.mass_ratio, np.sqrt(3) / 2, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(dynamics.rhs(0.0, l4), 0.0, atol=1e-12)
    assert dynamics.mu is None


def test_default_stm_uses_finite_differences():
    class Wrapped(Dynamics):
        mu = EARTH_MU

        def propagate(self, state, t0, t1):
            return TwoBody().propagate(state, t0, t1)

    final, stm = Wrapped().propagate_stm(STATE, 0.0, 3000.0)
    _, reference = TwoBody().propagate_stm(STATE, 0.0, 3000.0)
    assert np.abs(stm - reference).max() / np.abs(reference).max() < 1e-6
    assert Wrapped().describe() == {"type": "Wrapped"}


def test_reference_dynamics_are_picklable_and_described():
    for dynamics in (TwoBody(mu=42828.3), TwoBodyJ2(), CR3BP(mass_ratio=0.01)):
        clone = pickle.loads(pickle.dumps(dynamics))
        assert clone == dynamics
        assert clone.describe()["type"] == type(dynamics).__name__


def test_first_positional_argument_is_the_physical_parameter():
    assert TwoBody(42828.3).mu == 42828.3 and TwoBody(42828.3).rtol == 1e-12
    assert TwoBodyJ2(42828.3, 1.96e-3, 3396.2).radius == 3396.2
    assert CR3BP(0.01).mass_ratio == 0.01
    with pytest.raises(TypeError):
        TwoBody(EARTH_MU, 1e-10)  # tolerances are keyword-only
    with pytest.raises(ValueError):
        TwoBody(-1.0)
    with pytest.raises(ValueError):
        TwoBody(rtol=1.0)
    with pytest.raises(ValueError):
        CR3BP(0.9)


def test_failed_integration_raises_propagation_error():
    # A trajectory through the singularity at the centre cannot be integrated.
    with pytest.raises(PropagationError):
        TwoBody(rtol=1e-13).propagate(np.array([1e-3, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0, 1e6)
