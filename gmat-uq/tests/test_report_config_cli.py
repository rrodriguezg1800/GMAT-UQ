"""Reports, configuration files and the command-line interface."""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pytest

from gmat_uq import CR3BP, TwoBody, TwoBodyJ2, run_lincov, run_monte_carlo, run_sigma_point_check
from gmat_uq.analysis import run_analysis
from gmat_uq.cli import main
from gmat_uq.config import ConfigError, build_from_dict, load_config
from gmat_uq.consistency import compare_engines
from gmat_uq.models import local_frame_rotation
from gmat_uq.report import build_report, format_summary, write_report

from conftest import transfer_mission

CONFIG = """
[dynamics]
type = "two_body"
mu = 398600.4415

[mission]
initial_state = [7000.0, 0.0, 0.0, 0.0, 7.2, 2.2]
final_time = 5200.0
final_target = ["RMAG", "INC"]

[mission.initial_dispersion]
sigmas = [0.5, 0.5, 0.5, 5e-4, 5e-4, 5e-4]
frame = "RIC"

[error_models.Engine]
fixed_magnitude_sigma = 1e-5
proportional_magnitude_sigma = 0.005
fixed_pointing_sigma = 2e-5
proportional_pointing_sigma = 0.003

[navigation.Tracking]
sigmas = [0.05, 0.05, 0.05, 5e-5, 5e-5, 5e-5]
frame = "RIC"

[[maneuver]]
name = "TOI"
time = 600.0
dv = [-0.56, 0.71, 0.22]
error_model = "Engine"

[[maneuver]]
name = "TCM1"
time = 2000.0
error_model = "Engine"
navigation = "Tracking"
target = ["X", "Y", "Z"]
target_time = 5200.0

[analysis]
methods = ["lincov", "sigma_point", "monte_carlo"]
samples = 60
seed = 5
dv_samples = 4000
report = "out/report.json"
"""


def _write(tmp_path, text, name="mission.toml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


# -- report --------------------------------------------------------------------------------------------
def test_report_is_valid_json_and_complete(tmp_path):
    mission, dynamics = transfer_mission()
    lincov = run_lincov(mission, dynamics, dv_samples=5000, seed=1)
    sigma = run_sigma_point_check(lincov, dynamics)
    monte_carlo = run_monte_carlo(mission, dynamics, 40, seed=1, nominal=lincov.nominal)
    consistency = compare_engines(lincov, monte_carlo)
    report = build_report(mission, dynamics, lincov, sigma, monte_carlo, consistency)
    text = json.dumps(report, allow_nan=False)  # strict JSON: no NaN, no Infinity literals
    assert json.loads(text)["gmat_uq_version"]
    assert report["dynamics"] == {"type": "TwoBody", "mu": dynamics.mu}
    assert report["sample_size_guidance"]["wilks_minimum_samples"] == 299
    assert len(report["lincov"]["nodes"]) == lincov.nominal.n_nodes
    assert [m["name"] for m in report["lincov"]["maneuvers"]] == ["TOI", "TCM1", "TCM2"]
    assert report["mission"]["maneuvers"][1]["target"]["names"] == ["X", "Y", "Z"]
    assert report["monte_carlo"]["n_samples"] == 40 and report["monte_carlo"]["failed_samples"] == 0
    # 40 samples cannot bound the 99th percentile from above: the report says so explicitly.
    assert report["monte_carlo"]["delta_v"]["total"]["quantile"]["upper"] == "Infinity"
    assert report["consistency"]["final_state"]["n_samples"] == 40
    path = write_report(report, tmp_path / "nested" / "report.json")
    assert json.loads(path.read_text(encoding="utf-8")) == report

    summary = format_summary(report)
    for phrase in ("Linear covariance", "Sigma-point check", "Monte Carlo (40 samples", "delta-V 99", "verdict"):
        assert phrase in summary


def test_report_with_a_single_engine():
    mission, dynamics = transfer_mission(corrections=False)
    lincov = run_lincov(mission, dynamics, dv_samples=0)
    report = build_report(mission, dynamics, lincov=lincov)
    assert "monte_carlo" not in report and "sigma_point" not in report
    assert report["lincov"]["delta_v"] is None
    assert "Linear covariance" in format_summary(report)
    assert "Monte Carlo" not in format_summary(report)


# -- configuration ---------------------------------------------------------------------------------------
def test_configuration_builds_the_mission(tmp_path):
    config = load_config(_write(tmp_path, CONFIG))
    assert isinstance(config.dynamics, TwoBody)
    mission = config.mission
    assert [m.name for m in mission.maneuvers] == ["TOI", "TCM1"]
    assert mission.maneuvers[1].closed_loop and mission.maneuvers[1].target.names == ("X", "Y", "Z")
    assert mission.maneuvers[0].error_model.proportional_magnitude_sigma == 0.005
    assert mission.final_target.names == ("RMAG", "INC") and mission.final_target.mu == 398600.4415
    assert config.samples == 60 and config.seed == 5 and config.methods == ("lincov", "sigma_point", "monte_carlo")
    assert config.report == str(tmp_path / "out" / "report.json")
    assert config.source == str(tmp_path / "mission.toml")
    # Equal sigmas are unchanged by the rotation from the local frame.
    assert np.allclose(np.diag(mission.initial_dispersion.covariance), [0.25] * 3 + [2.5e-7] * 3)


def test_local_frame_navigation_is_oriented_with_the_state_at_the_maneuver(tmp_path):
    text = CONFIG.replace("sigmas = [0.05, 0.05, 0.05, 5e-5, 5e-5, 5e-5]", "sigmas = [0.3, 0.02, 0.01, 5e-5, 5e-5, 5e-5]")
    config = load_config(_write(tmp_path, text))
    maneuver = config.mission.maneuvers[1]
    # Recompute the nominal state arriving at the correction and check the radial sigma lies along it.
    state = np.array(config.mission.initial_state)
    state = config.dynamics.propagate(state, 0.0, 600.0)
    state[3:] += [-0.56, 0.71, 0.22]
    state = config.dynamics.propagate(state, 600.0, 2000.0)
    radial = local_frame_rotation(state, "RIC")[:, 0]
    assert np.isclose(radial @ maneuver.navigation.covariance[:3, :3] @ radial, 0.3**2, rtol=1e-9)


def test_json_configuration_and_other_dynamics(tmp_path):
    data = {
        "dynamics": {"type": "two_body_j2"},
        "mission": {"initial_state": [7000.0, 0, 0, 0, 7.4, 1.0], "final_time": 1000.0,
                    "initial_dispersion": {"covariance": np.diag([1.0] * 3 + [1e-6] * 3).tolist()},
                    "process_noise": {"accel_sigma": [1e-9, 1e-9, 1e-9], "max_step": 250.0}},
        "analysis": {"methods": "lincov"},
    }
    path = tmp_path / "mission.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    config = load_config(path)
    assert isinstance(config.dynamics, TwoBodyJ2) and config.methods == ("lincov",)
    assert config.mission.process_noise.max_step == 250.0
    assert isinstance(build_from_dict({"dynamics": {"type": "cr3bp"}, "mission": {"initial_state": [0.8, 0, 0, 0, 0.2, 0], "final_time": 1.0}}).dynamics, CR3BP)


def test_covariance_file_is_resolved_next_to_the_configuration(tmp_path):
    (tmp_path / "injection.txt").write_text("\n".join(" ".join(str(v) for v in row) for row in np.diag([4.0, 1.0, 1.0, 1e-6, 1e-6, 1e-6])), encoding="utf-8")
    text = CONFIG.replace('sigmas = [0.5, 0.5, 0.5, 5e-4, 5e-4, 5e-4]\nframe = "RIC"', 'covariance_file = "injection.txt"')
    config = load_config(_write(tmp_path, text))
    assert np.isclose(config.mission.initial_dispersion.covariance[0, 0], 4.0)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda t: t.replace('type = "two_body"', 'type = "warp"'), "unknown dynamics type"),
        (lambda t: t.replace("final_time = 5200.0\n", ""), "missing 'final_time'"),
        (lambda t: t.replace('error_model = "Engine"\n\n[[maneuver]]', 'error_model = "Missing"\n\n[[maneuver]]'), "unknown error model"),
        (lambda t: t.replace('navigation = "Tracking"', 'navigation = "Nope"'), "unknown navigation model"),
        (lambda t: t.replace('target = ["X", "Y", "Z"]', 'target = ["Altitude"]'), "unknown parameter"),
        (lambda t: t.replace("samples = 60", "samples = 60\nturbo = true"), r"unknown key\(s\) in \[analysis\]: turbo"),
        (lambda t: t.replace('methods = ["lincov", "sigma_point", "monte_carlo"]', 'methods = ["lincov", "guess"]'), "unknown method"),
        (lambda t: t.replace("samples = 60", "samples = 0"), "at least 1"),
        (lambda t: t.replace("fixed_magnitude_sigma = 1e-5", "fixed_magnitude_sigma = -1"), "non-negative"),
        (lambda t: t.replace("sigmas = [0.5, 0.5, 0.5, 5e-4, 5e-4, 5e-4]", "sigmas = [0.5, 0.5]\ncovariance = [[1.0]]"), "exactly one of"),
        (lambda t: t.replace("target_time = 5200.0", "target_time = 9000.0"), "later than final_time"),
        (lambda t: t.replace("initial_state = [7000.0, 0.0, 0.0, 0.0, 7.2, 2.2]\n", ""), "missing 'initial_state'"),
        (lambda t: t + "\n[extra]\nvalue = 1\n", r"unknown key\(s\) in the configuration"),
        (lambda t: t.replace("[dynamics]", "[dynamics"), "cannot parse"),
    ],
)
def test_configuration_errors_are_explained(tmp_path, edit, message):
    with pytest.raises(ConfigError, match=message):
        load_config(_write(tmp_path, edit(CONFIG)))


def test_missing_configuration_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml")


# -- analysis driver and CLI -------------------------------------------------------------------------------
def test_run_analysis_honours_the_method_list(tmp_path):
    config = load_config(_write(tmp_path, CONFIG))
    config.methods = ("monte_carlo",)
    result = run_analysis(config)
    assert result.lincov is None and result.sigma_point is None and result.consistency is None
    assert set(result.report) >= {"mission", "monte_carlo"} and "lincov" not in result.report
    config.methods = ("sigma_point",)
    result = run_analysis(config)
    assert result.sigma_point is not None and result.monte_carlo is None
    assert "sigma_point" in result.report and "lincov" not in result.report


def test_cli_runs_an_analysis_and_writes_the_report(tmp_path, capsys):
    path = _write(tmp_path, CONFIG)
    code = main(["run", str(path)])
    output = capsys.readouterr()
    assert code == 0
    assert "Report written to" in output.out and "Linear covariance" in output.out
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["monte_carlo"]["n_samples"] == 60
    assert report["consistency"]["agrees"] is True
    assert report["sigma_point"]["linear_ok"] is True
    assert report["lincov"]["target"]["labels"] == ["RMAG", "INC"]


def test_cli_overrides_and_quiet_mode(tmp_path, capsys):
    path = _write(tmp_path, CONFIG)
    other = tmp_path / "other.json"
    code = main(["run", str(path), "--methods", "lincov", "--report", str(other), "--seed", "9", "--quiet"])
    output = capsys.readouterr()
    assert code == 0 and output.out == ""
    report = json.loads(other.read_text(encoding="utf-8"))
    assert "monte_carlo" not in report and report["lincov"]["seed"] == 9
    code = main(["run", str(path), "--methods", "monte_carlo", "--samples", "12", "--workers", "2", "--guidance", "linear", "--quiet"])
    assert code == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["monte_carlo"]["n_samples"] == 12 and report["monte_carlo"]["workers"] == 2
    assert report["monte_carlo"]["guidance"] == "linear"


def test_cli_reports_errors_with_exit_code_one(tmp_path, capsys):
    assert main(["run", str(tmp_path / "absent.toml")]) == 1
    assert "configuration file not found" in capsys.readouterr().err
    bad = _write(tmp_path, CONFIG.replace('type = "two_body"', 'type = "warp"'))
    assert main(["run", str(bad)]) == 1
    assert "unknown dynamics type" in capsys.readouterr().err
    assert main(["run", str(_write(tmp_path, CONFIG, "ok.toml")), "--samples", "0"]) == 1


def test_cli_exit_code_two_flags_untrusted_linear_results(tmp_path, capsys):
    text = """
    [dynamics]
    type = "cr3bp"

    [mission]
    initial_state = [0.85, 0.0, 0.17, 0.0, 0.26, 0.0]
    final_time = 3.0

    [mission.initial_dispersion]
    sigmas = [2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3]

    [analysis]
    methods = ["lincov", "sigma_point"]
    """
    assert main(["run", str(_write(tmp_path, text))]) == 2
    assert "NONLINEAR ARCS FOUND" in capsys.readouterr().out


def test_cli_wilks_and_version(capsys):
    assert main(["wilks"]) == 0
    assert capsys.readouterr().out.strip() == "299"
    assert main(["wilks", "--probability", "0.95", "--confidence", "0.95"]) == 0
    assert capsys.readouterr().out.strip() == "59"
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "gmat-uq" in capsys.readouterr().out


def test_body_relative_targets_need_gmat_dynamics(tmp_path):
    text = CONFIG.replace('target = ["X", "Y", "Z"]', 'target = { parameters = ["BdotT", "BdotR"], body = "Luna" }')
    with pytest.raises(ConfigError, match="needs dynamics of type 'gmat'"):
        load_config(_write(tmp_path, text))
    table = CONFIG.replace('target = ["X", "Y", "Z"]', 'target = { parameters = ["X", "Y", "Z"] }')
    assert load_config(_write(tmp_path, table)).mission.maneuvers[1].target.names == ("X", "Y", "Z")
    unknown = CONFIG.replace('target = ["X", "Y", "Z"]', 'target = { parameters = ["X"], planet = "Mars" }')
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, unknown))
    no_time = CONFIG.replace("target_time = 5200.0\n", "")
    with pytest.raises(ConfigError, match="needs a target_time"):
        load_config(_write(tmp_path, no_time))


def test_report_when_every_sample_fails():
    from gmat_uq import Mission, StateDispersion
    from gmat_uq.dynamics import Dynamics, PropagationError

    class Refuses(Dynamics):
        mu = 398600.4415

        def propagate(self, state, t0, t1):
            raise PropagationError("always")

        def propagate_stm(self, state, t0, t1):
            return TwoBody().propagate_stm(state, t0, t1)

    mission = Mission((7000.0, 0.0, 0.0, 0.0, 7.4, 1.0), final_time=600.0, initial_dispersion=StateDispersion.from_sigmas([1.0] * 3 + [1e-3] * 3))
    dynamics = Refuses()
    monte_carlo = run_monte_carlo(mission, dynamics, 5, seed=1)
    assert monte_carlo.n_failed == 5
    report = build_report(mission, dynamics, monte_carlo=monte_carlo)
    assert report["monte_carlo"]["failed_samples"] == 5 and report["monte_carlo"]["final_sigmas"] is None
    assert "fewer than two samples" in format_summary(report)
    json.dumps(report, allow_nan=False)


def test_two_body_example_is_the_validation_case():
    """The shipped example is generated from validation test 3 and must stay in step with it."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "validation"))
    try:
        import cases
    finally:
        sys.path.pop(0)
    config = load_config(root / "examples" / "two_burn_transfer.toml")
    mission, dynamics, _ = cases.two_burn_transfer()
    example = run_lincov(config.mission, config.dynamics, dv_samples=0)
    reference = run_lincov(mission, dynamics, dv_samples=0)
    assert np.allclose(example.final_covariance, reference.final_covariance, rtol=1e-9, atol=1e-18)
    assert [m.name for m in config.mission.maneuvers] == [m.name for m in mission.maneuvers]


def test_values_of_the_wrong_kind_are_configuration_errors(tmp_path, capsys):
    for edit in (
        lambda t: t.replace("time = 600.0", 'time = "soon"'),
        lambda t: t.replace("final_time = 5200.0", "final_time = [1, 2]"),
        lambda t: t.replace("dv = [-0.56, 0.71, 0.22]", "dv = 3"),
        lambda t: t.replace("fixed_magnitude_sigma = 1e-5", 'fixed_magnitude_sigma = "big"'),
        lambda t: t.replace("[navigation.Tracking]", "[navigation]\nTracking = 7\n[navigation.Other]"),
    ):
        path = _write(tmp_path, edit(CONFIG))
        with pytest.raises(ConfigError):
            load_config(path)
    assert main(["run", str(_write(tmp_path, CONFIG.replace("time = 600.0", 'time = "soon"')))]) == 1
    assert "error" in capsys.readouterr().err
