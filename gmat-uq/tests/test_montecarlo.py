"""Closed-loop Monte Carlo engine and its agreement with the linear covariance."""

from __future__ import annotations

import numpy as np
import pytest

from gmat_uq import (
    Maneuver,
    Mission,
    ParameterTarget,
    ProcessNoise,
    StateDispersion,
    TwoBody,
    compute_nominal,
    run_lincov,
    run_monte_carlo,
    sample_linear,
)
from gmat_uq.consistency import compare_engines
from gmat_uq.dynamics import Dynamics, PropagationError
from gmat_uq.dynamics.ode import EARTH_MU
from gmat_uq.stats import compare_covariance

from conftest import transfer_mission


def test_linear_guidance_matches_the_linear_sampler_sample_by_sample():
    """With the same draws, the two engines differ only by second-order terms."""
    errors = []
    for scale in (1e-2, 1e-3):
        mission, dynamics = transfer_mission(scale=scale)
        nominal = compute_nominal(mission, dynamics)
        monte_carlo = run_monte_carlo(mission, dynamics, 20, seed=5, guidance="linear", nominal=nominal)
        linear = sample_linear(mission, nominal, np.arange(20), seed=5)
        assert np.array_equal(monte_carlo.indices, linear.indices)
        corrections = linear.commanded_dv[:, 1:]
        assert np.abs(monte_carlo.commanded_dv[:, 1:] - corrections).max() < 1e-2 * np.abs(corrections).max()
        assert np.array_equal(monte_carlo.commanded_dv[:, 0], linear.commanded_dv[:, 0])  # open-loop burn
        difference = np.abs(monte_carlo.final_deviation - linear.final_deviation).max()
        size = np.abs(linear.final_deviation).max()
        errors.append(difference / size)
    assert errors[0] < 1e-3
    # Ten times smaller dispersions: the relative mismatch shrinks about ten times (quadratic terms).
    assert 5.0 < errors[0] / errors[1] < 20.0


def test_results_are_identical_for_any_number_of_workers():
    """Validation test 6: the same seed gives bit-identical results on 1 and 3 workers."""
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    serial = run_monte_carlo(mission, dynamics, 18, seed=11, workers=1, nominal=nominal)
    parallel = run_monte_carlo(mission, dynamics, 18, seed=11, workers=3, nominal=nominal, chunk_size=2)
    assert parallel.workers == 3 and serial.workers == 1
    for name in ("indices", "final_deviation", "commanded_dv", "execution_error", "deviations_before", "converged", "iterations", "failed"):
        assert np.array_equal(getattr(serial, name), getattr(parallel, name)), name


def test_samples_do_not_depend_on_the_batch_they_are_run_in():
    mission, dynamics = transfer_mission()
    nominal = compute_nominal(mission, dynamics)
    whole = run_monte_carlo(mission, dynamics, 12, seed=2, nominal=nominal)
    tail = run_monte_carlo(mission, dynamics, 4, seed=2, nominal=nominal, first_index=8)
    assert np.array_equal(tail.indices, np.arange(8, 12))
    assert np.array_equal(tail.final_deviation, whole.final_deviation[8:])
    other_seed = run_monte_carlo(mission, dynamics, 4, seed=3, nominal=nominal)
    assert not np.array_equal(other_seed.final_deviation, whole.final_deviation[:4])


def test_perfect_knowledge_and_execution_deliver_the_target_exactly():
    mission, dynamics = transfer_mission(navigation=False, execution=False)
    result = run_monte_carlo(mission, dynamics, 15, seed=1)
    assert result.n_failed == 0 and result.n_unconverged == 0
    assert np.abs(result.final_deviation[:, :3]).max() < 1e-6  # km
    assert result.iterations[:, 1:].min() >= 1 and result.iterations[:, 0].max() == 0
    # The same case with the linear law leaves the second-order miss.
    linear = run_monte_carlo(mission, dynamics, 15, seed=1, guidance="linear")
    assert np.abs(linear.final_deviation[:, :3]).max() > 1e-5


def test_linear_covariance_agrees_with_monte_carlo():
    """Validation test 3: burns, corrections, navigation and execution errors on a two-burn transfer."""
    mission, dynamics = transfer_mission()
    mission = Mission(
        mission.initial_state, mission.final_time, mission.maneuvers, mission.initial_dispersion,
        final_target=ParameterTarget(("RMAG", "INC"), EARTH_MU),
    )
    lincov = run_lincov(mission, dynamics, dv_samples=40_000, seed=21)
    monte_carlo = run_monte_carlo(mission, dynamics, 600, seed=21, nominal=lincov.nominal, workers=3)
    assert monte_carlo.n_failed == 0 and monte_carlo.n_unconverged == 0
    report = compare_engines(lincov, monte_carlo)
    # One-sigma dispersions within 5% in every state component (statistical tolerance included).
    for ratio, lower, upper in report.component_sigma_ratios:
        assert lower < 1.05 and upper > 0.95, (ratio, lower, upper)
        assert abs(ratio - 1.0) < 0.1
    # In six dimensions, 600 samples spread the extreme directional ratios by about +/- 10% by
    # chance alone, so the covariance is judged with the statistical test.
    assert report.final_state.p_value > 0.01
    assert report.final_state.max_sigma_deviation < 0.2
    assert report.target.max_sigma_deviation < 0.1
    assert report.dv_inside_interval
    assert report.dv_relative_difference < 0.02
    assert report.agrees
    assert report.to_dict()["delta_v_quantile"]["linear_inside_monte_carlo_interval"] is True


def test_process_noise_is_consistent_between_the_engines():
    noise = ProcessNoise((3e-6, 3e-6, 3e-6), max_step=500.0)
    mission = Mission(
        (7000.0, 0.0, 0.0, 0.0, 7.4, 1.0),
        final_time=2000.0,
        initial_dispersion=StateDispersion.from_sigmas([1e-3] * 3 + [1e-6] * 3),
        process_noise=noise,
    )
    dynamics = TwoBody()
    lincov = run_lincov(mission, dynamics, dv_samples=0)
    monte_carlo = run_monte_carlo(mission, dynamics, 1500, seed=6, nominal=lincov.nominal)
    assert compare_covariance(monte_carlo.final_deviation, lincov.final_covariance).max_sigma_deviation < 0.1


class _FailsFarFromNominal(Dynamics):
    """Two-body dynamics that refuses states displaced along +x, to exercise failure handling."""

    mu = EARTH_MU

    def __init__(self, limit):
        self.limit = limit
        self.inner = TwoBody()

    def propagate(self, state, t0, t1):
        if t0 == 0.0 and state[0] > self.limit:
            raise PropagationError("simulated failure")
        return self.inner.propagate(state, t0, t1)

    def propagate_stm(self, state, t0, t1):
        return self.inner.propagate_stm(state, t0, t1)


def test_failed_samples_are_flagged_and_excluded():
    mission = Mission(
        (7000.0, 0.0, 0.0, 0.0, 7.4, 1.0),
        final_time=1500.0,
        maneuvers=(Maneuver("A", 500.0, dv=(0.01, 0.0, 0.0)),),
        initial_dispersion=StateDispersion.from_sigmas([1.0] * 3 + [1e-4] * 3),
        final_target=ParameterTarget(("RMAG",)),
    )
    result = run_monte_carlo(mission, _FailsFarFromNominal(7000.5), 200, seed=4)
    assert 0 < result.n_failed < 200
    assert np.isnan(result.final_deviation[result.failed]).all()
    assert np.isfinite(result.final_deviation[result.valid]).all()
    assert result.dv_magnitudes.shape == (200 - result.n_failed, 1)
    assert result.final_covariance.shape == (6, 6) and np.isfinite(result.final_covariance).all()
    assert result.target_covariance.shape == (1, 1)
    assert np.isnan(result.target_deviation[result.failed]).all()
    # Roughly 31% of the samples start beyond +0.5 sigma.
    assert 0.2 < result.n_failed / 200 < 0.42


def test_open_loop_mission_statistics():
    mission, dynamics = transfer_mission(corrections=False)
    lincov = run_lincov(mission, dynamics, dv_samples=0)
    result = run_monte_carlo(mission, dynamics, 400, seed=8, nominal=lincov.nominal)
    assert result.converged.all() and not result.iterations.any()
    statistics = result.dv_statistics()
    assert np.isclose(statistics["total"]["mean"], statistics["nominal_total"])  # open loop: always the nominal command
    assert compare_covariance(result.final_deviation, lincov.final_covariance).max_sigma_deviation < 0.2
    assert result.covariance_before(1).shape == (6, 6)
    assert result.target_covariance is None and result.n_samples == 400


def test_argument_validation():
    mission, dynamics = transfer_mission()
    with pytest.raises(ValueError, match="n_samples"):
        run_monte_carlo(mission, dynamics, 0)
    with pytest.raises(ValueError, match="workers"):
        run_monte_carlo(mission, dynamics, 5, workers=0)
    with pytest.raises(ValueError, match="guidance"):
        run_monte_carlo(mission, dynamics, 5, guidance="magic")
    with pytest.raises(ValueError, match="first_index"):
        run_monte_carlo(mission, dynamics, 5, first_index=-1)


def test_progress_callback_and_worker_cap():
    mission, dynamics = transfer_mission(corrections=False)
    calls = []
    result = run_monte_carlo(mission, dynamics, 3, workers=8, progress=lambda done, total: calls.append((done, total)))
    assert result.workers == 3  # never more workers than samples
    assert calls[-1] == (3, 3)
