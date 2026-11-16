"""Consider parameters: uncertain model parameters that are neither estimated nor corrected."""

from __future__ import annotations

import numpy as np
import pytest

from gmat_uq import (
    ConsiderParameter,
    Maneuver,
    Mission,
    StateDispersion,
    TwoBody,
    TwoBodyJ2,
    compute_nominal,
    run_lincov,
    run_monte_carlo,
    run_sigma_point_check,
    sample_linear,
)
from gmat_uq.config import ConfigError, build_from_dict
from gmat_uq.dynamics.ode import EARTH_MU
from gmat_uq.report import build_report
from gmat_uq.stats import compare_covariance

from conftest import transfer_mission

STATE = (7000.0, 0.0, 0.0, 0.0, 7.4, 1.0)
MU_SIGMA = 0.5  # km^3/s^2: deliberately large, to make the effect visible


def test_consider_parameter_validation():
    assert ConsiderParameter(" mu ", 1).name == "mu" and ConsiderParameter("mu", 1).sigma == 1.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sigma"):
            ConsiderParameter("Cd", bad)
    with pytest.raises(ValueError, match="name"):
        ConsiderParameter("  ", 0.1)
    with pytest.raises(ValueError, match="unique"):
        Mission(STATE, final_time=10.0, consider=(ConsiderParameter("mu", 1.0), ConsiderParameter("mu", 2.0)))
    with pytest.raises(ValueError, match="ConsiderParameter"):
        Mission(STATE, final_time=10.0, consider=("mu",))


def test_reference_models_expose_their_physical_parameters():
    model = TwoBodyJ2()
    assert model.parameter_value("mu") == EARTH_MU and model.parameter_value("j2") == model.j2
    with pytest.raises(ValueError, match="no parameter 'rtol'"):
        model.parameter_value("rtol")  # integrator settings are not physical parameters
    with pytest.raises(ValueError, match="no parameter 'Cd'"):
        model.propagate(np.array(STATE), 0.0, 10.0, parameters={"Cd": 2.2})
    # A propagation with a parameter equals a propagation of the model built with that value.
    changed = model.propagate(np.array(STATE), 0.0, 3000.0, parameters={"mu": EARTH_MU + 10.0})
    assert np.array_equal(changed, TwoBodyJ2(EARTH_MU + 10.0).propagate(np.array(STATE), 0.0, 3000.0))
    _, stm = model.propagate_stm(np.array(STATE), 0.0, 3000.0, parameters={"mu": EARTH_MU + 10.0})
    assert np.array_equal(stm, TwoBodyJ2(EARTH_MU + 10.0).propagate_stm(np.array(STATE), 0.0, 3000.0)[1])
    assert np.array_equal(model.propagate(np.array(STATE), 0.0, 3000.0, parameters={}), model.propagate(np.array(STATE), 0.0, 3000.0))


def test_unknown_consider_parameter_is_reported():
    mission = Mission(STATE, final_time=100.0, consider=(ConsiderParameter("Cd", 0.1),))
    with pytest.raises(ValueError, match="no parameter 'Cd'"):
        compute_nominal(mission, TwoBody())


def test_sensitivity_of_a_coast_matches_an_independent_difference():
    dynamics = TwoBody()
    mission = Mission(STATE, final_time=5000.0, consider=(ConsiderParameter("mu", MU_SIGMA),), report_times=(2000.0,))
    nominal = compute_nominal(mission, dynamics)
    assert nominal.consider_names == ("mu",) and np.allclose(nominal.consider_nominal, [EARTH_MU])
    # The chained sensitivity over both intervals equals d(final state)/d(mu) over the whole coast.
    total = nominal.stms[1] @ nominal.consider_sensitivities[0][:, 0] + nominal.consider_sensitivities[1][:, 0]
    step = 1e-2
    reference = (
        TwoBody(EARTH_MU + step).propagate(np.array(STATE), 0.0, 5000.0)
        - TwoBody(EARTH_MU - step).propagate(np.array(STATE), 0.0, 5000.0)
    ) / (2 * step)
    assert np.allclose(total, reference, rtol=1e-5, atol=1e-9)
    transition = nominal.augmented_transition(0)
    assert transition.shape == (7, 7) and transition[6, 6] == 1.0 and not transition[6, :6].any()


def test_consider_only_covariance_is_the_sensitivity_outer_product():
    dynamics = TwoBody()
    mission = Mission(STATE, final_time=5000.0, consider=(ConsiderParameter("mu", MU_SIGMA),))
    result = run_lincov(mission, dynamics, dv_samples=0)
    sensitivity = result.nominal.consider_sensitivities[0][:, 0]
    assert np.allclose(result.final_covariance, MU_SIGMA**2 * np.outer(sensitivity, sensitivity), rtol=1e-12)
    assert result.augmented_after.shape == (2, 7, 7)
    assert np.isclose(result.augmented_after[-1][6, 6], MU_SIGMA**2)
    assert not result.covariance_before[0].any()  # the initial state itself is exact


def test_linear_samples_and_monte_carlo_agree_with_the_consider_covariance():
    mission, dynamics = transfer_mission()
    mission = Mission(
        mission.initial_state, mission.final_time, mission.maneuvers, mission.initial_dispersion,
        consider=(ConsiderParameter("mu", MU_SIGMA),),
    )
    result = run_lincov(mission, dynamics, dv_samples=30_000, seed=4)
    assert compare_covariance(result.samples.final_deviation, result.final_covariance).max_sigma_deviation < 0.03
    monte_carlo = run_monte_carlo(mission, dynamics, 400, seed=4, nominal=result.nominal, workers=2)
    assert monte_carlo.n_failed == 0 and monte_carlo.n_unconverged == 0
    assert monte_carlo.consider_deviation.shape == (400, 1)
    # Same draws in both engines.
    linear = sample_linear(mission, result.nominal, np.arange(400), seed=4)
    assert np.allclose(np.std(monte_carlo.consider_deviation), MU_SIGMA, rtol=0.1)
    comparison = compare_covariance(monte_carlo.final_deviation, result.final_covariance)
    assert comparison.p_value > 0.001 and comparison.max_sigma_deviation < 0.25
    assert np.abs(monte_carlo.final_deviation - linear.final_deviation).max() < 0.05 * np.abs(linear.final_deviation).max()


def test_the_parameter_error_is_not_corrected_away():
    """With perfect knowledge and execution, the target is still missed because of the parameter."""
    mission, dynamics = transfer_mission(navigation=False, execution=False)
    exact = Mission(mission.initial_state, mission.final_time, mission.maneuvers, mission.initial_dispersion)
    uncertain = Mission(
        mission.initial_state, mission.final_time, mission.maneuvers, mission.initial_dispersion,
        consider=(ConsiderParameter("mu", MU_SIGMA),),
    )
    perfect = run_lincov(exact, dynamics, dv_samples=0)
    degraded = run_lincov(uncertain, dynamics, dv_samples=0)
    assert perfect.position_sigma_rss() < 1e-6
    assert degraded.position_sigma_rss() > 1e-3
    samples = run_monte_carlo(uncertain, dynamics, 60, seed=2)
    assert np.abs(samples.final_deviation[:, :3]).max() > 1e-3


def test_sigma_point_check_includes_the_parameters():
    mission = Mission(STATE, final_time=5000.0, consider=(ConsiderParameter("mu", MU_SIGMA),), report_times=(2500.0,))
    dynamics = TwoBody()
    check = run_sigma_point_check(run_lincov(mission, dynamics, dv_samples=0), dynamics)
    # Seven dimensions; the six state columns of the factor are zero, so only the parameter costs.
    assert check.propagations == 2 * len(check.segments)
    assert check.linear_ok and check.max_index < 0.01
    large = Mission(STATE, final_time=5000.0, consider=(ConsiderParameter("mu", 20000.0),), report_times=(2500.0,))
    assert run_sigma_point_check(run_lincov(large, dynamics, dv_samples=0), dynamics).max_index > check.max_index


def test_configuration_and_report_carry_the_consider_parameters():
    data = {
        "dynamics": {"type": "two_body"},
        "mission": {"initial_state": list(STATE), "final_time": 3000.0},
        "consider": [{"name": "mu", "sigma": 0.5}],
    }
    config = build_from_dict(data)
    assert config.mission.consider == (ConsiderParameter("mu", 0.5),)
    result = run_lincov(config.mission, config.dynamics, dv_samples=0)
    report = build_report(config.mission, config.dynamics, lincov=result)
    assert report["mission"]["consider"] == [{"name": "mu", "sigma": 0.5}]
    assert report["lincov"]["consider_nominal"] == {"mu": EARTH_MU}
    for bad, message in (
        ([{"name": "Cd", "sigma": 0.1}], "no parameter 'Cd'"),
        ([{"name": "mu"}], "missing 'sigma'"),
        ([{"name": "mu", "sigma": -1}], "sigma"),
        ([{"name": "mu", "sigma": 1, "extra": 2}], "unknown key"),
        ({"name": "mu", "sigma": 1}, "array of tables"),
    ):
        with pytest.raises(ConfigError, match=message):
            build_from_dict({**data, "consider": bad})
