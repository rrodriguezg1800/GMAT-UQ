"""Sigma-point check and the nonlinear corrector."""

from __future__ import annotations

import numpy as np
import pytest

from gmat_uq import (
    CR3BP,
    Maneuver,
    Mission,
    ParameterTarget,
    StateDispersion,
    TwoBody,
    compute_nominal,
    run_lincov,
    run_sigma_point_check,
    solve_correction,
)
from gmat_uq.dynamics import Dynamics, PropagationError
from gmat_uq.dynamics.ode import EARTH_MU
from gmat_uq.guidance import linear_correction

from conftest import transfer_mission


class LinearDynamics(Dynamics):
    """Exactly linear dynamics (double integrator with a spring), for null tests."""

    mu = None

    def __init__(self):
        matrix = np.zeros((6, 6))
        matrix[:3, 3:] = np.eye(3)
        matrix[3:, :3] = -1e-6 * np.diag([1.0, 2.0, 3.0])
        self.matrix = matrix

    def _stm(self, dt):
        from scipy.linalg import expm

        return expm(self.matrix * dt)

    def propagate(self, state, t0, t1):
        return self._stm(t1 - t0) @ np.asarray(state, dtype=float)

    def propagate_stm(self, state, t0, t1):
        stm = self._stm(t1 - t0)
        return stm @ np.asarray(state, dtype=float), stm


# -- sigma-point check -----------------------------------------------------------------------------------
def test_linear_dynamics_have_zero_nonlinearity_index():
    mission = Mission(
        (100.0, 0.0, 0.0, 0.0, 0.1, 0.0),
        final_time=2000.0,
        initial_dispersion=StateDispersion.from_sigmas([5.0] * 3 + [0.01] * 3),
        report_times=(700.0,),
    )
    dynamics = LinearDynamics()
    result = run_sigma_point_check(run_lincov(mission, dynamics, dv_samples=0), dynamics)
    assert result.max_index < 1e-9
    assert result.linear_ok and result.flagged_segments == ()
    assert len(result.segments) == 2 and result.propagations == 24


def test_small_dispersions_are_linear_and_the_index_grows_with_them():
    indices = []
    for scale in (0.01, 1.0, 20.0):
        mission, dynamics = transfer_mission(scale=scale)
        result = run_sigma_point_check(run_lincov(mission, dynamics, dv_samples=0), dynamics)
        indices.append(result.max_index)
    assert indices[0] < 1e-3
    assert indices[0] < indices[1] < indices[2]
    # The curvature term scales linearly with the size of the dispersion.
    assert 50.0 < indices[1] / indices[0] < 200.0


def test_zero_covariance_segments_cost_nothing():
    mission = Mission((7000.0, 0.0, 0.0, 0.0, 7.5, 0.0), final_time=1000.0)
    dynamics = TwoBody()
    result = run_sigma_point_check(run_lincov(mission, dynamics, dv_samples=0), dynamics)
    assert result.propagations == 0 and result.max_index == 0.0 and result.linear_ok


def test_rank_deficient_covariance_is_handled():
    # Dispersion along a single direction: eleven of the twelve sigma points coincide with the nominal.
    direction = np.array([1.0, 0.0, 0.0, 0.0, 1e-3, 0.0])
    mission = Mission((7000.0, 0.0, 0.0, 0.0, 7.5, 0.0), final_time=1500.0, initial_dispersion=StateDispersion(np.outer(direction, direction)))
    dynamics = TwoBody()
    result = run_sigma_point_check(run_lincov(mission, dynamics, dv_samples=0), dynamics)
    assert result.propagations == 2
    assert 0.0 <= result.max_index < 0.05


def test_strongly_nonlinear_arc_is_flagged():
    """Validation test 5 (reduced): a long arc near the Moon in the CR3BP is flagged."""
    dynamics = CR3BP()
    state = (0.85, 0.0, 0.17, 0.0, 0.26, 0.0)
    dispersion = StateDispersion.from_sigmas([2e-3] * 3 + [2e-3] * 3)
    mission = Mission(state, final_time=3.0, initial_dispersion=dispersion, report_times=(0.2,))
    result = run_sigma_point_check(run_lincov(mission, dynamics, dv_samples=0), dynamics, threshold=0.1)
    assert not result.linear_ok
    first, second = result.segments
    assert not first.flagged  # the short first arc is still linear
    assert second.flagged and second.index > 0.1
    report = result.to_dict()
    assert report["linear_ok"] is False and report["segments"][1]["flagged"] is True
    with pytest.raises(ValueError):
        run_sigma_point_check(run_lincov(mission, dynamics, dv_samples=0), dynamics, threshold=0.0)


# -- nonlinear corrector ---------------------------------------------------------------------------------
def test_corrector_restores_the_target_with_the_full_dynamics():
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    plan = nominal.plans[1]
    estimate = nominal.states_before[plan.node] + np.array([8.0, -5.0, 3.0, 0.004, 0.002, -0.003])
    solution = solve_correction(dynamics, nominal, plan, estimate)
    assert solution.converged and 2 <= solution.iterations <= 12
    state = estimate.copy()
    state[3:] += solution.dv
    arrival = dynamics.propagate(state, nominal.times[plan.node], nominal.times[plan.target_node])
    assert np.allclose(arrival[:3], plan.target_goal, atol=1e-6)
    # Reusing the nominal partials changes the cost, not the answer.
    newton = solve_correction(dynamics, nominal, plan, estimate, reuse_nominal_partials=False)
    assert newton.converged and newton.iterations <= 6
    assert np.linalg.norm(newton.dv - solution.dv) < 1e-9
    # The linear law alone leaves a second-order miss, which the corrector removes.
    linear_state = estimate.copy()
    linear_state[3:] += linear_correction(plan, nominal, estimate)
    linear_arrival = dynamics.propagate(linear_state, nominal.times[plan.node], nominal.times[plan.target_node])
    assert np.linalg.norm(linear_arrival[:3] - plan.target_goal) > 1e-3


def test_corrector_on_the_nominal_returns_the_nominal_delta_v():
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    plan = nominal.plans[2]
    solution = solve_correction(dynamics, nominal, plan, nominal.states_before[plan.node])
    assert solution.converged
    assert np.linalg.norm(solution.dv - plan.maneuver.dv_vector) < 1e-9


def test_corrector_with_fewer_targets_than_controls():
    target = ParameterTarget(("RMAG", "INC"), EARTH_MU)
    mission = Mission(
        (7000.0, 0.0, 0.0, 0.0, 7.4, 1.0),
        final_time=4000.0,
        maneuvers=(Maneuver("A", 500.0, target=target, target_time=4000.0),),
    )
    dynamics = TwoBody()
    nominal = compute_nominal(mission, dynamics)
    plan = nominal.plans[0]
    estimate = nominal.states_before[plan.node] + np.array([3.0, 2.0, -1.0, 1e-3, -2e-3, 1e-3])
    solution = solve_correction(dynamics, nominal, plan, estimate)
    assert solution.converged
    state = estimate.copy()
    state[3:] += solution.dv
    arrival = dynamics.propagate(state, 500.0, 4000.0)
    assert np.allclose(target.residual(arrival, plan.target_goal), 0.0, atol=1e-6)


def test_corrector_accounts_for_a_later_nominal_burn():
    position = ParameterTarget.position()
    mission = Mission(
        (7000.0, 0.0, 0.0, 0.0, 7.4, 1.0),
        final_time=5000.0,
        maneuvers=(
            Maneuver("TCM", 400.0, target=position, target_time=5000.0),
            Maneuver("DSM", 2000.0, dv=(0.05, 0.02, -0.01)),
        ),
    )
    dynamics = TwoBody()
    nominal = compute_nominal(mission, dynamics)
    plan = nominal.plans[0]
    estimate = nominal.states_before[plan.node] + np.array([2.0, -1.0, 1.0, 1e-3, 1e-3, -1e-3])
    solution = solve_correction(dynamics, nominal, plan, estimate)
    assert solution.converged
    state = estimate.copy()
    state[3:] += solution.dv
    state = dynamics.propagate(state, 400.0, 2000.0)
    state[3:] += np.array([0.05, 0.02, -0.01])
    arrival = dynamics.propagate(state, 2000.0, 5000.0)
    assert np.allclose(arrival[:3], plan.target_goal, atol=1e-6)


def test_corrector_failure_returns_the_linear_guess():
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    plan = nominal.plans[1]
    estimate = nominal.states_before[plan.node] + np.array([50.0, -30.0, 20.0, 0.02, 0.01, -0.02])
    guess = linear_correction(plan, nominal, estimate)
    stopped = solve_correction(dynamics, nominal, plan, estimate, max_iterations=1)
    assert not stopped.converged and stopped.iterations == 1
    assert np.array_equal(stopped.dv, guess)

    class Broken(Dynamics):
        def propagate(self, state, t0, t1):
            raise PropagationError("no")

    failed = solve_correction(Broken(), nominal, plan, estimate)
    assert not failed.converged and np.array_equal(failed.dv, guess)


# -- stand-alone maneuver design ---------------------------------------------------------------------------
def test_design_maneuver_reaches_a_chosen_position():
    from gmat_uq import design_maneuver

    dynamics = TwoBody()
    state = np.array([7000.0, 0.0, 0.0, 0.0, 7.4, 1.0])
    drift = dynamics.propagate(state, 0.0, 3000.0)
    aim = drift[:3] + np.array([40.0, -25.0, 10.0])
    solution = design_maneuver(dynamics, state, 0.0, ParameterTarget.position(), 3000.0, aim)
    assert solution.converged
    burned = state.copy()
    burned[3:] += solution.dv
    assert np.allclose(dynamics.propagate(burned, 0.0, 3000.0)[:3], aim, atol=1e-6)
    # The goal the trajectory already meets needs no burn.
    nothing = design_maneuver(dynamics, state, 0.0, ParameterTarget.position(), 3000.0, drift[:3])
    assert nothing.converged and np.linalg.norm(nothing.dv) < 1e-9
    with pytest.raises(ValueError, match="later than"):
        design_maneuver(dynamics, state, 100.0, ParameterTarget.position(), 100.0, aim)
    with pytest.raises(ValueError, match="goal must hold"):
        design_maneuver(dynamics, state, 0.0, ParameterTarget.position(), 3000.0, [1.0, 2.0])
    with pytest.raises(ValueError, match="one to three"):
        design_maneuver(dynamics, state, 0.0, ParameterTarget(("X", "Y", "Z", "VX")), 3000.0, [0.0] * 4)
