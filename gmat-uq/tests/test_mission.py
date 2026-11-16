"""Mission definition, node grid and nominal trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from gmat_uq import FunctionTarget, Maneuver, Mission, ParameterTarget, ProcessNoise, TwoBody, compute_nominal
from gmat_uq.dynamics.ode import EARTH_MU
from gmat_uq.linalg import velocity_input_matrix

from conftest import transfer_mission

STATE = (7000.0, 0.0, 0.0, 0.0, 7.4, 1.0)


def test_maneuver_validation():
    with pytest.raises(ValueError, match="name"):
        Maneuver("", 10.0)
    with pytest.raises(ValueError, match="three finite"):
        Maneuver("A", 10.0, dv=(1.0, 2.0))
    with pytest.raises(ValueError, match="both target and target_time"):
        Maneuver("A", 10.0, target=ParameterTarget.position())
    with pytest.raises(ValueError, match="both target and target_time"):
        Maneuver("A", 10.0, target_time=100.0)
    with pytest.raises(ValueError, match="later than the maneuver"):
        Maneuver("A", 10.0, target=ParameterTarget.position(), target_time=10.0)
    with pytest.raises(ValueError, match="one to three"):
        Maneuver("A", 10.0, target=ParameterTarget(("X", "Y", "Z", "VX")), target_time=20.0)
    open_loop = Maneuver("A", 10, dv=[0.1, 0, 0])
    assert not open_loop.closed_loop and open_loop.dv == (0.1, 0.0, 0.0) and open_loop.time == 10.0


def test_mission_validation():
    with pytest.raises(ValueError, match="six finite"):
        Mission((1.0, 2.0, 3.0), final_time=10.0)
    with pytest.raises(ValueError, match="later than initial_time"):
        Mission(STATE, final_time=0.0)
    with pytest.raises(ValueError, match="unique"):
        Mission(STATE, final_time=100.0, maneuvers=(Maneuver("A", 10.0), Maneuver("A", 20.0)))
    with pytest.raises(ValueError, match="share the same time"):
        Mission(STATE, final_time=100.0, maneuvers=(Maneuver("A", 10.0), Maneuver("B", 10.0)))
    with pytest.raises(ValueError, match="outside the mission time span"):
        Mission(STATE, final_time=100.0, maneuvers=(Maneuver("A", 200.0),))
    with pytest.raises(ValueError, match="later than final_time"):
        Mission(
            STATE,
            final_time=100.0,
            maneuvers=(Maneuver("A", 10.0, target=ParameterTarget.position(), target_time=150.0),),
        )
    with pytest.raises(ValueError, match="report_times"):
        Mission(STATE, final_time=100.0, report_times=(500.0,))


def test_maneuvers_are_sorted_and_nodes_are_unique():
    mission = Mission(
        STATE,
        final_time=1000.0,
        maneuvers=(
            Maneuver("late", 600.0),
            Maneuver("early", 100.0, target=ParameterTarget.position(), target_time=1000.0),
        ),
        report_times=(100.0, 250.0),
    )
    assert [m.name for m in mission.maneuvers] == ["early", "late"]
    assert np.allclose(mission.node_times(), [0.0, 100.0, 250.0, 600.0, 1000.0])


def test_process_noise_refines_the_node_grid():
    mission = Mission(STATE, final_time=1000.0, maneuvers=(Maneuver("A", 250.0),), process_noise=ProcessNoise((1e-9,) * 3, 100.0))
    nodes = mission.node_times()
    assert np.diff(nodes).max() <= 100.0 + 1e-9
    assert 250.0 in nodes and nodes[0] == 0.0 and nodes[-1] == 1000.0
    assert np.all(np.diff(nodes) > 0)


def test_nominal_trajectory_is_continuous_and_applies_burns():
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    assert nominal.n_nodes == len(mission.node_times())
    assert np.array_equal(nominal.states_before[0], mission.state_vector)
    for node in range(nominal.n_nodes - 1):
        propagated = dynamics.propagate(nominal.states_after[node], nominal.times[node], nominal.times[node + 1])
        assert np.allclose(propagated, nominal.states_before[node + 1], rtol=1e-10, atol=1e-8)
    toi = nominal.plans[0]
    assert toi.maneuver.name == "TOI" and toi.gain is None
    jump = nominal.states_after[toi.node] - nominal.states_before[toi.node]
    assert np.allclose(jump, np.concatenate([np.zeros(3), toi.maneuver.dv_vector]))
    # The transfer arrives at the larger radius, as designed.
    assert np.isclose(np.linalg.norm(nominal.states_before[-1][:3]), 12000.0, rtol=1e-6)


def test_guidance_gain_nulls_the_target_deviation():
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    input_matrix = velocity_input_matrix()
    for plan in nominal.plans[1:]:
        assert plan.gain.shape == (3, 6)
        closed = plan.sensitivity @ (np.eye(6) - input_matrix @ plan.gain)
        assert np.abs(closed).max() < 1e-9 * np.abs(plan.sensitivity).max()
        assert np.allclose(plan.target_goal, nominal.states_before[plan.target_node][:3])
        assert np.allclose(nominal.stm_between(plan.node, plan.target_node)[:3], plan.sensitivity, rtol=1e-9)


def test_under_determined_target_uses_the_minimum_norm_gain():
    target = ParameterTarget(("RMAG",))
    mission = Mission(STATE, final_time=3000.0, maneuvers=(Maneuver("A", 500.0, target=target, target_time=3000.0),))
    nominal = compute_nominal(mission, TwoBody())
    plan = nominal.plans[0]
    control = plan.sensitivity @ velocity_input_matrix()  # 1 x 3
    assert plan.gain.shape == (3, 6)
    # Minimum norm: every commanded correction lies along the single control direction.
    assert np.linalg.matrix_rank(plan.gain) == 1
    assert np.allclose(control @ plan.gain, plan.sensitivity, rtol=1e-9)


def _dependent_components(state):
    return [state[0], 2.0 * state[0]]


def test_uncontrollable_target_is_reported():
    # Two target components that always move together cannot be controlled independently.
    target = FunctionTarget(_dependent_components, ("x", "twice_x"))
    mission = Mission(STATE, final_time=1000.0, maneuvers=(Maneuver("A", 500.0, target=target, target_time=1000.0),))
    with pytest.raises(ValueError, match="cannot control its target"):
        compute_nominal(mission, TwoBody())


def test_final_target_is_evaluated_on_the_nominal():
    mission = Mission(STATE, final_time=2000.0, final_target=ParameterTarget(("RMAG", "SMA"), EARTH_MU))
    nominal = compute_nominal(mission, TwoBody())
    assert nominal.final_target_value.shape == (2,)
    assert nominal.final_target_jacobian.shape == (2, 6)
    assert np.isclose(nominal.final_target_value[0], np.linalg.norm(nominal.states_after[-1][:3]))
    assert nominal.plan_at(0) is None
