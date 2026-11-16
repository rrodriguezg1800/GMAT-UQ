"""Linear covariance engine, random streams and the linear sampler."""

from __future__ import annotations

import numpy as np
import pytest

from gmat_uq import (
    Maneuver,
    ManeuverErrorModel,
    Mission,
    NavigationErrorModel,
    ParameterTarget,
    ProcessNoise,
    StateDispersion,
    TwoBody,
    compute_nominal,
    run_lincov,
    sample_linear,
)
from gmat_uq.dynamics.ode import EARTH_MU
from gmat_uq.linalg import velocity_input_matrix
from gmat_uq.sampling import draw_normals, draw_normals_batch
from gmat_uq.stats import compare_covariance

from conftest import transfer_mission

STATE = (7000.0, 0.0, 0.0, 0.0, 7.4, 1.0)
P0 = np.diag([1.0, 2.0, 0.5, 1e-6, 2e-6, 5e-7])
B = velocity_input_matrix()


# -- random streams -----------------------------------------------------------------------------------
def test_draws_depend_only_on_seed_and_index():
    a = draw_normals(7, 3, n_maneuvers=2)
    b = draw_normals(7, 3, n_maneuvers=2)
    assert np.array_equal(a.initial, b.initial) and np.array_equal(a.execution, b.execution)
    assert not np.array_equal(a.initial, draw_normals(7, 4, 2).initial)
    assert not np.array_equal(a.initial, draw_normals(8, 3, 2).initial)
    assert a.navigation.shape == (2, 6) and a.execution.shape == (2, 3) and a.process.shape == (0, 6)
    with pytest.raises(ValueError):
        draw_normals(7, -1, 2)


def test_batch_draws_equal_single_draws_and_process_noise_does_not_shift_streams():
    batch = draw_normals_batch(5, [0, 9, 2], n_maneuvers=3, n_process_intervals=4)
    for row, index in enumerate([0, 9, 2]):
        single = draw_normals(5, index, 3, 4)
        assert np.array_equal(batch.initial[row], single.initial)
        assert np.array_equal(batch.navigation[row], single.navigation)
        assert np.array_equal(batch.execution[row], single.execution)
        assert np.array_equal(batch.process[row], single.process)
        without = draw_normals(5, index, 3, 0)
        assert np.array_equal(without.navigation, single.navigation)
        assert np.array_equal(without.execution, single.execution)


def test_scattered_indices_and_seed_validation():
    scattered = draw_normals_batch(3, [40, 2, 3, 4, 17, 41], n_maneuvers=2, n_process_intervals=1)
    for row, index in enumerate([40, 2, 3, 4, 17, 41]):
        single = draw_normals(3, index, 2, 1)
        assert np.array_equal(scattered.initial[row], single.initial)
        assert np.array_equal(scattered.execution[row], single.execution)
        assert np.array_equal(scattered.process[row], single.process)
    # Adding a source of uncertainty does not disturb the others.
    assert np.array_equal(draw_normals(3, 9, 0).initial, draw_normals(3, 9, 5, 7).initial)
    empty = draw_normals_batch(3, [], 2)
    assert empty.initial.shape == (0, 6) and empty.navigation.shape == (0, 2, 6)
    for bad_seed in (-1, 2**64):
        with pytest.raises(ValueError, match="seed"):
            draw_normals(bad_seed, 0, 1)
    with pytest.raises(ValueError, match="non-negative"):
        draw_normals_batch(3, [1, -2], 1)
    huge = draw_normals(2**64 - 1, 10**12, 1)  # large seeds and indices are fine
    assert np.all(np.isfinite(huge.initial))


def test_draws_are_standard_normal():
    batch = draw_normals_batch(1, np.arange(4000), n_maneuvers=1)
    values = np.concatenate([batch.initial.ravel(), batch.navigation.ravel(), batch.execution.ravel()])
    assert abs(values.mean()) < 0.01 and abs(values.std() - 1.0) < 0.01
    from scipy import stats as scipy_stats

    assert scipy_stats.kstest(values, "norm").pvalue > 0.001
    # No correlation between the draws of consecutive samples, or between components.
    assert abs(np.corrcoef(batch.initial[:-1, 0], batch.initial[1:, 0])[0, 1]) < 0.06
    assert np.abs(np.corrcoef(batch.initial, rowvar=False) - np.eye(6)).max() < 0.06


# -- covariance recursion -------------------------------------------------------------------------------
def test_coast_only_is_stm_propagation():
    mission = Mission(STATE, final_time=5000.0, initial_dispersion=StateDispersion(P0), report_times=(2000.0,))
    dynamics = TwoBody()
    result = run_lincov(mission, dynamics, dv_samples=0)
    _, stm = dynamics.propagate_stm(np.array(STATE), 0.0, 5000.0)
    assert np.allclose(result.final_covariance, stm @ P0 @ stm.T, rtol=1e-8)
    assert np.allclose(result.covariance_before[0], P0)
    assert len(result.times) == 3 and result.samples is None and result.dv_statistics() is None
    assert result.maneuvers == ()
    assert np.allclose(result.sigmas(0), np.sqrt(np.diag(P0)))


def test_no_uncertainty_gives_zero_covariance_and_nominal_delta_v():
    mission, dynamics = transfer_mission(scale=0.0)
    mission = Mission(mission.initial_state, mission.final_time, mission.maneuvers)  # no initial dispersion either
    result = run_lincov(mission, dynamics, dv_samples=50)
    assert not result.covariance_after.any()
    statistics = result.dv_statistics()
    nominal_dv = np.linalg.norm(mission.maneuvers[0].dv)
    assert np.isclose(statistics["nominal_total"], nominal_dv)
    assert np.allclose(result.samples.total_dv, nominal_dv, rtol=1e-12)


def test_open_loop_burn_adds_its_execution_covariance():
    gates = ManeuverErrorModel(1e-5, 0.01, 2e-5, 0.02)
    dv = (0.05, -0.02, 0.01)
    mission = Mission(STATE, final_time=3000.0, maneuvers=(Maneuver("A", 1000.0, dv, error_model=gates),), initial_dispersion=StateDispersion(P0))
    result = run_lincov(mission, TwoBody(), dv_samples=0)
    node = result.nominal.plans[0].node
    added = result.covariance_after[node] - result.covariance_before[node]
    assert np.allclose(added, B @ gates.covariance(dv) @ B.T, atol=1e-18)
    assert np.allclose(result.maneuvers[0].execution_covariance, gates.covariance(dv))
    assert not result.maneuvers[0].correction_covariance.any() and not result.maneuvers[0].closed_loop


def test_perfect_correction_removes_the_position_dispersion_at_the_target():
    mission, dynamics = transfer_mission(navigation=False, execution=False)
    result = run_lincov(mission, dynamics, dv_samples=0)
    final_position_sigma = result.sigmas()[:3]
    start_position_sigma = result.sigmas(0)[:3]
    assert np.all(final_position_sigma < 1e-6 * start_position_sigma)
    assert result.velocity_sigma_rss() > 0.0  # velocity is not controlled


def test_closed_loop_update_follows_the_documented_formula():
    mission, dynamics = transfer_mission()
    result = run_lincov(mission, dynamics, dv_samples=0)
    for record in result.maneuvers[1:]:
        plan = result.nominal.plan_at(record.node)
        gain, knowledge = plan.gain, plan.maneuver.navigation.covariance
        before = result.covariance_before[record.node]
        correction = gain @ (before + knowledge) @ gain.T
        assert np.allclose(record.correction_covariance, correction, rtol=1e-10)
        closed = np.eye(6) - B @ gain
        execution = plan.maneuver.error_model.expected_covariance(plan.maneuver.dv_vector, correction)
        expected = closed @ before @ closed.T + B @ gain @ knowledge @ gain.T @ B.T + B @ execution @ B.T
        assert np.allclose(result.covariance_after[record.node], expected, rtol=1e-10, atol=1e-20)


def test_navigation_error_alone_sets_the_delivery_accuracy():
    mission, dynamics = transfer_mission(execution=False)
    result = run_lincov(mission, dynamics, dv_samples=0)
    last = result.nominal.plans[-1]
    knowledge = last.maneuver.navigation.covariance
    stm = result.nominal.stm_between(last.node, last.target_node)
    # After the last correction the true deviation at the target is driven by the knowledge
    # error of that correction only: -Phi (B K) e.
    mapping = stm @ B @ last.gain
    expected = mapping @ knowledge @ mapping.T
    assert np.allclose(result.final_covariance[:3, :3], expected[:3, :3], rtol=1e-6, atol=1e-14)


def test_process_noise_adds_q_on_every_interval():
    noise = ProcessNoise((1e-7, 2e-7, 3e-7), max_step=500.0)
    mission = Mission(STATE, final_time=2000.0, initial_dispersion=StateDispersion(P0), process_noise=noise)
    dynamics = TwoBody()
    result = run_lincov(mission, dynamics, dv_samples=0)
    covariance = P0.copy()
    times = result.times
    assert len(times) == 5
    state = np.array(STATE)
    for start, end in zip(times, times[1:]):
        state, stm = dynamics.propagate_stm(state, start, end)
        covariance = stm @ covariance @ stm.T + noise.covariance(end - start)
    assert np.allclose(result.final_covariance, covariance, rtol=1e-9)
    quiet = run_lincov(Mission(STATE, final_time=2000.0, initial_dispersion=StateDispersion(P0)), dynamics, dv_samples=0)
    assert np.all(np.diag(result.final_covariance) > np.diag(quiet.final_covariance))


# -- linear sampler ----------------------------------------------------------------------------------------
def test_linear_samples_reproduce_the_analytic_covariance():
    mission, dynamics = transfer_mission()
    result = run_lincov(mission, dynamics, dv_samples=40_000, seed=3)
    samples = result.samples
    comparison = compare_covariance(samples.final_deviation, result.final_covariance)
    assert comparison.max_sigma_deviation < 0.03
    assert comparison.mean_offset_sigma < 0.03
    for node in range(result.nominal.n_nodes):
        sample_sigma = samples.deviations_before[:, node].std(axis=0)
        assert np.allclose(sample_sigma, result.sigmas(node, after=False), rtol=0.03)
    for index, record in enumerate(result.maneuvers):
        commanded = samples.commanded_dv[:, index] - record.dv_nominal
        if record.closed_loop:
            assert compare_covariance(commanded, record.correction_covariance).max_sigma_deviation < 0.03
        execution = compare_covariance(samples.execution_error[:, index], record.execution_covariance)
        assert execution.max_sigma_deviation < 0.03


def test_linear_samples_with_process_noise_reproduce_the_analytic_covariance():
    noise = ProcessNoise((2e-6, 2e-6, 2e-6), max_step=400.0)
    mission = Mission(STATE, final_time=2000.0, initial_dispersion=StateDispersion(P0 * 1e-4), process_noise=noise)
    result = run_lincov(mission, TwoBody(), dv_samples=30_000, seed=2)
    assert compare_covariance(result.samples.final_deviation, result.final_covariance).max_sigma_deviation < 0.03


def test_sample_linear_is_indexed_and_reproducible():
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    whole = sample_linear(mission, nominal, np.arange(50), seed=9)
    part = sample_linear(mission, nominal, [10, 40], seed=9)
    assert np.array_equal(part.final_deviation, whole.final_deviation[[10, 40]])
    assert np.array_equal(part.commanded_dv, whole.commanded_dv[[10, 40]])
    assert not np.array_equal(whole.final_deviation, sample_linear(mission, nominal, np.arange(50), seed=10).final_deviation)
    assert whole.dv_magnitudes.shape == (50, 3) and whole.total_dv.shape == (50,)


def test_delta_v_statistics_and_target_covariance():
    mission, dynamics = transfer_mission()
    mission = Mission(
        mission.initial_state, mission.final_time, mission.maneuvers, mission.initial_dispersion,
        final_target=ParameterTarget(("RMAG", "INC"), EARTH_MU),
    )
    result = run_lincov(mission, dynamics, dv_samples=20_000, seed=1)
    statistics = result.dv_statistics()
    total = statistics["total"]
    assert total["quantile"]["value"] > total["mean"] > statistics["nominal_total"] > 0.0
    assert total["quantile"]["lower"] <= total["quantile"]["value"] <= total["quantile"]["upper"]
    assert set(statistics["per_maneuver"]) == {"TOI", "TCM1", "TCM2"}
    # A statistical correction has zero nominal delta-V but a positive mean.
    assert statistics["per_maneuver"]["TCM1"]["mean"] > 0.0
    target = result.target_covariance
    jacobian = result.nominal.final_target_jacobian
    assert target.shape == (2, 2)
    assert np.allclose(target, jacobian @ result.final_covariance @ jacobian.T)
    ellipse = result.target_ellipse(3.0)
    assert ellipse.semi_major >= ellipse.semi_minor > 0.0
    no_target = run_lincov(transfer_mission()[0], dynamics, dv_samples=0)
    assert no_target.target_covariance is None and no_target.target_ellipse() is None


def test_isotropic_correction_gives_maxwell_delta_v():
    """An isotropic commanded correction has a Maxwell-distributed magnitude."""
    from scipy import stats as scipy_stats

    mission, dynamics = transfer_mission(execution=False, navigation=False)
    nominal = compute_nominal(mission, dynamics)
    plan = nominal.plans[1]
    # Choose the initial dispersion so that the first correction is isotropic with sigma = 1 m/s:
    # K D K^T = s^2 I for D = s^2 pinv(K) pinv(K)^T propagated back to the initial time.
    sigma = 1e-3
    to_maneuver = nominal.stm_between(0, plan.node)
    back = np.linalg.inv(to_maneuver)
    spread = back @ np.linalg.pinv(plan.gain)
    dispersion = StateDispersion(sigma**2 * spread @ spread.T)
    mission = Mission(mission.initial_state, mission.final_time, mission.maneuvers, dispersion)
    result = run_lincov(mission, dynamics, dv_samples=60_000, seed=4)
    assert np.allclose(result.maneuvers[1].correction_covariance, sigma**2 * np.eye(3), rtol=1e-6, atol=1e-16)
    summary = result.dv_statistics()["per_maneuver"]["TCM1"]
    assert np.isclose(summary["mean"], 2 * sigma * np.sqrt(2 / np.pi), rtol=0.01)
    assert np.isclose(summary["quantile"]["value"], sigma * np.sqrt(scipy_stats.chi2.ppf(0.99, 3)), rtol=0.02)


def test_node_history_is_dropped_for_long_grids(monkeypatch):
    import gmat_uq.lincov as lincov_module

    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    kept = sample_linear(mission, nominal, np.arange(30), seed=1)
    light = sample_linear(mission, nominal, np.arange(30), seed=1, keep_history=False)
    assert light.deviations_before is None
    assert np.array_equal(light.final_deviation, kept.final_deviation)
    assert np.array_equal(light.commanded_dv, kept.commanded_dv)
    monkeypatch.setattr(lincov_module, "_HISTORY_LIMIT", 10)
    result = run_lincov(mission, dynamics, nominal, dv_samples=30, seed=1)
    assert result.samples.deviations_before is None and result.dv_statistics()["total"]["n_samples"] == 30
