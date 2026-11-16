"""Integration with a real GMAT installation.

Tests marked ``gmat`` need the ``GMAT_ROOT`` environment variable to point at a
GMAT installation and are skipped otherwise.  The first tests in this file
need no installation.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from gmat_uq import (
    GmatDynamics,
    GmatNotFoundError,
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
    run_monte_carlo,
    run_sigma_point_check,
)
from gmat_uq.cli import main
from gmat_uq.dynamics import finite_difference_stm
from gmat_uq.dynamics.gmat import find_gmat_root, resources_only_script

import gmat_scripts as scripts

needs_gmat = pytest.mark.gmat


# -- no installation needed --------------------------------------------------------------------------
def test_resources_only_script_drops_outputs_and_the_mission_sequence():
    text = scripts.HIGH_FIDELITY.format(error_control="None", **scripts.LEO_STATE)
    text += scripts.REPORTS.format(state_file="state.txt", cov_file="cov.txt")
    text += "Create XYPlot Plot1, Plot2; % two plots\nPlot1.XVariable = Sat.ElapsedSecs;\nGMAT Plot2.Grid = On;\n"
    text += "Create Variable Sat2Count;\nSat2Count = 3;\n"
    text += scripts.NATIVE_COVARIANCE_SEQUENCE.format(diagonal="1 1 1 1 1 1", duration=60)
    cleaned = resources_only_script(text)
    lowered = cleaned.lower()
    for forbidden in ("reportfile", "statereport", "covreport", "xyplot", "plot1", "plot2", "propagate", "orbiterrorcovariance"):
        assert forbidden not in lowered, forbidden
    for kept in ("Create Spacecraft Sat;", "Create ForceModel FM;", "Create Propagator Prop;", "Prop.FM = FM;", "Create Variable Sat2Count;", "Sat2Count = 3;"):
        assert kept in cleaned, kept
    assert cleaned.rstrip().endswith("BeginMissionSequence;")
    assert cleaned.count("BeginMissionSequence") == 1


def test_resources_only_script_respects_comments_and_quotes():
    text = "\n".join(
        [
            "% Create ReportFile Commented;",
            "Create Spacecraft Sat;  % the spacecraft",
            "Sat.Id = '100% real';",
            "create reportfile Rep;",
            "rep.Filename = 'a.txt';",
            "Create String note;",
            "note = 'BeginMissionSequence is not here';",
            "BeginMissionSequence",
            "Propagate Prop(Sat);",
        ]
    )
    cleaned = resources_only_script(text)
    assert "Sat.Id = '100% real';" in cleaned
    assert "% Create ReportFile Commented;" in cleaned  # comments are left alone
    assert "create reportfile" not in cleaned and "rep.Filename" not in cleaned
    assert "note = 'BeginMissionSequence is not here';" in cleaned
    assert "Propagate" not in cleaned


def test_missing_gmat_is_reported_clearly(tmp_path, monkeypatch):
    monkeypatch.delenv("GMAT_ROOT", raising=False)
    with pytest.raises(GmatNotFoundError, match="GMAT_ROOT"):
        find_gmat_root()
    with pytest.raises(GmatNotFoundError, match="does not look like a GMAT installation"):
        find_gmat_root(tmp_path)
    script = scripts.write_script(tmp_path, "a.script", "BeginMissionSequence;\n")
    with pytest.raises(GmatNotFoundError):
        GmatDynamics(str(script))
    with pytest.raises(FileNotFoundError):
        GmatDynamics(str(tmp_path / "absent.script"), gmat_root=str(tmp_path))
    with pytest.raises(ValueError, match="stm_method"):
        GmatDynamics(str(script), stm_method="magic")


# -- with a GMAT installation ---------------------------------------------------------------------------
@pytest.fixture
def point_mass(gmat_root, tmp_path):
    script = scripts.write_script(tmp_path, "point_mass.script", scripts.POINT_MASS.format(**scripts.LEO_STATE))
    return GmatDynamics(str(script), gmat_root=gmat_root)


@pytest.fixture
def high_fidelity(gmat_root, tmp_path):
    text = scripts.HIGH_FIDELITY.format(error_control="RSSStep", **scripts.LEO_STATE) + "BeginMissionSequence;\n"
    return GmatDynamics(str(scripts.write_script(tmp_path, "high_fidelity.script", text)), gmat_root=gmat_root)


@needs_gmat
def test_script_is_loaded_and_described(point_mass):
    description = point_mass.describe()
    assert description["central_body"] == "Earth" and description["stm"] == "variational"
    assert description["step_control"] == "adaptive"
    assert np.isclose(point_mass.mu, 398600.4415)
    assert np.allclose(point_mass.initial_state(), [7000, 0, 1300, 0, 7.35, 1.0])
    assert 31041.0 < point_mass.epoch_a1mjd < 31043.0  # 01 Jan 2026 in A.1 modified Julian days
    assert point_mass.uses_variational_stm and point_mass.central_body == "Earth"


@needs_gmat
def test_gmat_point_mass_propagation_matches_the_reference_two_body(point_mass):
    state = point_mass.initial_state()
    reference = TwoBody(point_mass.mu)
    for duration in (600.0, 5400.0, 10800.0):
        assert np.allclose(point_mass.propagate(state, 0.0, duration), reference.propagate(state, 0.0, duration), atol=2e-7, rtol=0.0)
    # Start time other than zero, backwards propagation and a zero-length interval.
    forward = point_mass.propagate(state, 1000.0, 4000.0)
    assert np.allclose(point_mass.propagate(forward, 4000.0, 1000.0), state, atol=2e-7)
    assert np.array_equal(point_mass.propagate(state, 50.0, 50.0), state)
    same, identity = point_mass.propagate_stm(state, 50.0, 50.0)
    assert np.array_equal(identity, np.eye(6))


@needs_gmat
def test_gmat_variational_stm_matches_finite_differences_and_the_reference(point_mass, high_fidelity):
    state = point_mass.initial_state()
    final, stm = point_mass.propagate_stm(state, 0.0, 5400.0)
    _, reference = TwoBody(point_mass.mu).propagate_stm(state, 0.0, 5400.0)
    assert np.abs(stm - reference).max() / np.abs(reference).max() < 1e-8
    assert np.array_equal(final, point_mass.propagate(state, 0.0, 5400.0))  # same trajectory with or without the STM

    state = high_fidelity.initial_state()
    _, stm = high_fidelity.propagate_stm(state, 0.0, 5400.0)
    numeric = finite_difference_stm(high_fidelity.propagate, state, 0.0, 5400.0)
    assert np.abs(stm - numeric).max() / np.abs(stm).max() < 1e-6


@needs_gmat
def test_gmat_propagation_does_not_depend_on_call_history(high_fidelity):
    state = high_fidelity.initial_state()
    first = high_fidelity.propagate(state, 0.0, 7200.0)
    high_fidelity.propagate(state * 1.01, 300.0, 9000.0)
    high_fidelity.propagate_stm(state * 0.99, 0.0, 1000.0)
    assert np.array_equal(high_fidelity.propagate(state, 0.0, 7200.0), first)


@needs_gmat
def test_finite_difference_stm_mode(gmat_root, tmp_path):
    script = scripts.write_script(tmp_path, "fd.script", scripts.POINT_MASS.format(**scripts.LEO_STATE))
    dynamics = GmatDynamics(str(script), gmat_root=gmat_root, stm_method="finite_difference")
    assert not dynamics.uses_variational_stm and dynamics.describe()["stm"] == "finite_difference"
    state = dynamics.initial_state()
    _, stm = dynamics.propagate_stm(state, 0.0, 3000.0)
    _, reference = TwoBody(dynamics.mu).propagate_stm(state, 0.0, 3000.0)
    assert np.abs(stm - reference).max() / np.abs(reference).max() < 1e-6


@needs_gmat
def test_linear_covariance_reproduces_gmat_native_covariance_propagation(gmat_root, tmp_path):
    """Validation test 2: with no burns, GMAT-UQ equals ``Propagate Prop(Sat, 'Covariance')``."""
    state_file, cov_file = tmp_path / "native_state.txt", tmp_path / "native_cov.txt"
    diagonal = [1.0e-2, 1.5e-2, 1.0e-2, 2.5e-7, 2.5e-7, 4.0e-7]
    duration = 21600.0
    text = scripts.HIGH_FIDELITY.format(error_control="None", **scripts.LEO_STATE)
    text += scripts.REPORTS.format(state_file=state_file, cov_file=cov_file)
    text += scripts.NATIVE_COVARIANCE_SEQUENCE.format(diagonal=" ".join(repr(v) for v in diagonal), duration=duration)
    script = scripts.write_script(tmp_path, "native_covariance.script", text)

    scripts.run_gmat_console(gmat_root, script)  # GMAT's own answer
    native_state = np.loadtxt(state_file)
    native_covariance = np.loadtxt(cov_file).reshape(6, 6)
    reports_before = {path.name: path.read_bytes() for path in (state_file, cov_file)}

    dynamics = GmatDynamics(str(script), gmat_root=gmat_root)
    assert dynamics.describe()["step_control"] == "fixed 20 s"
    mission = Mission(tuple(dynamics.initial_state()), final_time=duration, initial_dispersion=StateDispersion(np.diag(diagonal)))
    result = run_lincov(mission, dynamics, dv_samples=0)

    assert np.allclose(result.nominal.states_after[-1], native_state, rtol=0.0, atol=1e-7)
    native_sigma = np.sqrt(np.diag(native_covariance))
    # Position and velocity sigmas equal to 8 significant digits (the acceptance criterion).
    assert np.abs(result.sigmas() / native_sigma - 1.0).max() < 1e-8
    assert np.abs(result.final_covariance - native_covariance).max() < 1e-8 * np.abs(native_covariance).max()

    # Analysing a script must never touch the files that script writes.
    for path in (state_file, cov_file):
        assert path.read_bytes() == reports_before[path.name]
    assert not list(tmp_path.glob("gmat_uq_tmp_*"))


@needs_gmat
def test_process_noise_reproduces_gmat_state_noise_compensation(gmat_root, tmp_path):
    """The process-noise model equals GMAT's ``ProcessNoiseModel`` (SNC) on the inertial axes."""
    state_file, cov_file = tmp_path / "state.txt", tmp_path / "cov.txt"
    diagonal = [1.0e-2, 1.5e-2, 1.0e-2, 2.5e-7, 2.5e-7, 4.0e-7]
    accel_sigma = (1.0e-6, 2.0e-6, 3.0e-6)
    duration = 3600.0
    text = scripts.HIGH_FIDELITY.format(error_control="None", **scripts.LEO_STATE)
    text += (
        "Create ProcessNoiseModel SNC;\nSNC.Type = StateNoiseCompensation;\nSNC.CoordinateSystem = EarthMJ2000Eq;\n"
        f"SNC.AccelNoiseSigma = [{' '.join(repr(v) for v in accel_sigma)}];\nSat.ProcessNoiseModel = SNC;\n"
    )
    text += scripts.REPORTS.format(state_file=state_file, cov_file=cov_file)
    text += scripts.NATIVE_COVARIANCE_SEQUENCE.format(diagonal=" ".join(repr(v) for v in diagonal), duration=duration)
    script = scripts.write_script(tmp_path, "snc.script", text)
    scripts.run_gmat_console(gmat_root, script)
    native_sigma = np.sqrt(np.diag(np.loadtxt(cov_file).reshape(6, 6)))

    dynamics = GmatDynamics(str(script), gmat_root=gmat_root)
    initial = StateDispersion(np.diag(diagonal))
    state = tuple(dynamics.initial_state())
    # GMAT adds the noise once per integration step (20 s in this script).
    noisy = Mission(state, final_time=duration, initial_dispersion=initial, process_noise=ProcessNoise(accel_sigma, max_step=20.0))
    result = run_lincov(noisy, dynamics, dv_samples=0)
    assert np.abs(result.sigmas() / native_sigma - 1.0).max() < 1e-9
    # The noise matters in this case: leaving it out changes the sigmas by about 2%.
    quiet = run_lincov(Mission(state, final_time=duration, initial_dispersion=initial), dynamics, dv_samples=0)
    assert np.abs(quiet.sigmas() / native_sigma - 1.0).max() > 5e-3
    # A coarser noise grid stays close.
    coarse = Mission(state, final_time=duration, initial_dispersion=initial, process_noise=ProcessNoise(accel_sigma, max_step=60.0))
    assert np.abs(run_lincov(coarse, dynamics, dv_samples=0).sigmas() / native_sigma - 1.0).max() < 1e-3


@needs_gmat
def test_body_states_and_body_relative_targets(gmat_root, tmp_path, high_fidelity):
    moon = high_fidelity.body_state("Luna", 0.0)
    assert 350000.0 < np.linalg.norm(moon[:3]) < 410000.0 and 0.9 < np.linalg.norm(moon[3:]) < 1.1
    assert not high_fidelity.body_state("Earth", 0.0).any()  # the central body is the origin
    later = high_fidelity.body_state("Luna", 60.0)
    assert np.allclose(later[:3] - moon[:3], 60.0 * moon[3:], rtol=0.01)  # it moves with its velocity
    assert np.isclose(high_fidelity.body_mu("Luna"), 4902.8, rtol=1e-4)
    with pytest.raises(ValueError, match="not a celestial body"):
        high_fidelity.body_state("Vulcan", 0.0)

    # Parameters relative to the Moon equal what GMAT reports in a Moon-centred coordinate system.
    report = tmp_path / "moon.txt"
    flyby = moon + np.array([-20000.0, 9000.0, 3000.0, 0.9, 0.3, -0.1])
    text = scripts.SPACECRAFT.format(**dict(zip(("x", "y", "z", "vx", "vy", "vz"), (repr(float(v)) for v in flyby))))
    text += (
        "Create CoordinateSystem MoonMJ2000Eq;\nMoonMJ2000Eq.Origin = Luna;\nMoonMJ2000Eq.Axes = MJ2000Eq;\n"
        f"Create ReportFile Rep;\nRep.Filename = '{report}';\nRep.WriteHeaders = False;\nRep.Precision = 16;\n"
        "Rep.ColumnWidth = 26;\nRep.WriteReport = False;\nBeginMissionSequence;\n"
        "Report Rep Sat.MoonMJ2000Eq.BdotT Sat.MoonMJ2000Eq.BdotR Sat.Luna.RMAG Sat.Luna.ECC;\n"
    )
    scripts.run_gmat_console(gmat_root, scripts.write_script(tmp_path, "moon.script", text))
    expected = np.loadtxt(report)
    target = ParameterTarget(("BdotT", "BdotR", "RMAG", "ECC"), high_fidelity.body_mu("Luna"), tuple(moon))
    assert np.allclose(target(flyby), expected, rtol=1e-9)


@needs_gmat
def test_target_parameters_match_gmat(gmat_root, tmp_path):
    report = tmp_path / "parameters.txt"
    script = scripts.write_script(tmp_path, "parameters.script", scripts.PARAMETER_REPORT.format(report_file=report, **scripts.FLYBY_STATE))
    scripts.run_gmat_console(gmat_root, script)
    expected = np.loadtxt(report)
    state = np.array([scripts.FLYBY_STATE[k] for k in ("x", "y", "z", "vx", "vy", "vz")], dtype=float)
    mine = ParameterTarget(scripts.PARAMETER_NAMES, 398600.4415)(state)
    assert np.allclose(mine, expected, rtol=1e-12, atol=0.0)


def _gmat_mission(dynamics):
    state = dynamics.initial_state()
    gates = ManeuverErrorModel(1e-6, 0.002, 1e-6, 0.002)
    knowledge = NavigationErrorModel.from_sigmas([0.02] * 3 + [2e-5] * 3)
    position = ParameterTarget.position()
    return Mission(
        tuple(state),
        final_time=5400.0,
        maneuvers=(
            Maneuver("Burn", 900.0, dv=(0.0, 0.05, 0.01), error_model=gates),
            Maneuver("TCM", 2400.0, error_model=gates, target=position, target_time=5400.0, navigation=knowledge),
        ),
        initial_dispersion=StateDispersion.from_sigmas([0.1] * 3 + [1e-4] * 3),
        final_target=ParameterTarget(("RMAG", "INC"), dynamics.mu),
    )


@needs_gmat
def test_all_engines_run_on_gmat_dynamics_and_agree(high_fidelity):
    mission = _gmat_mission(high_fidelity)
    lincov = run_lincov(mission, high_fidelity, dv_samples=5000, seed=3)
    sigma = run_sigma_point_check(lincov, high_fidelity)
    assert sigma.linear_ok and sigma.max_index < 0.01
    monte_carlo = run_monte_carlo(mission, high_fidelity, 60, seed=3, nominal=lincov.nominal)
    assert monte_carlo.n_failed == 0 and monte_carlo.n_unconverged == 0
    # The correction really lands on the target with GMAT's full force model.
    perfect = Mission(mission.initial_state, mission.final_time, tuple(
        Maneuver(m.name, m.time, m.dv, target=m.target, target_time=m.target_time) for m in mission.maneuvers
    ), mission.initial_dispersion)
    exact = run_monte_carlo(perfect, high_fidelity, 5, seed=3)
    assert np.abs(exact.final_deviation[:, :3]).max() < 1e-6
    sample_sigma = monte_carlo.final_deviation.std(axis=0, ddof=1)
    assert np.allclose(sample_sigma, lincov.sigmas(), rtol=0.35)  # 60 samples: +/- 2.6 sigma of sampling noise


@needs_gmat
def test_gmat_monte_carlo_is_identical_across_worker_counts(high_fidelity):
    """Validation test 6 with GMAT in the loop: each worker process loads its own GMAT."""
    mission = _gmat_mission(high_fidelity)
    nominal = compute_nominal(mission, high_fidelity)
    serial = run_monte_carlo(mission, high_fidelity, 8, seed=7, workers=1, nominal=nominal)
    parallel = run_monte_carlo(mission, high_fidelity, 8, seed=7, workers=2, nominal=nominal, chunk_size=2)
    assert parallel.workers == 2
    for name in ("final_deviation", "commanded_dv", "execution_error", "deviations_before", "iterations", "converged"):
        assert np.array_equal(getattr(serial, name), getattr(parallel, name)), name


@needs_gmat
def test_pickled_dynamics_reconnect_and_two_scripts_can_alternate(gmat_root, tmp_path, point_mass):
    state = point_mass.initial_state()
    reference = point_mass.propagate(state, 0.0, 3000.0)
    clone = pickle.loads(pickle.dumps(point_mass))
    assert np.array_equal(clone.propagate(state, 0.0, 3000.0), reference)

    text = scripts.HIGH_FIDELITY.format(error_control="RSSStep", **scripts.LEO_STATE) + "BeginMissionSequence;\n"
    other = GmatDynamics(str(scripts.write_script(tmp_path, "other.script", text)), gmat_root=gmat_root)
    different = other.propagate(state, 0.0, 3000.0)
    assert np.abs(different - reference).max() > 1e-3  # the gravity field and third bodies matter
    # Going back to the first object reloads its script transparently.
    assert np.array_equal(point_mass.propagate(state, 0.0, 3000.0), reference)
    assert np.array_equal(other.propagate(state, 0.0, 3000.0), different)


@needs_gmat
def test_script_problems_are_reported(gmat_root, tmp_path):
    script = scripts.write_script(tmp_path, "names.script", scripts.POINT_MASS.format(**scripts.LEO_STATE))
    with pytest.raises(ValueError, match="does not define Spacecraft 'Probe'"):
        GmatDynamics(str(script), spacecraft="Probe", gmat_root=gmat_root)
    with pytest.raises(ValueError, match="does not define"):
        GmatDynamics(str(script), propagator="Nope", gmat_root=gmat_root)
    moon_frame = scripts.POINT_MASS.format(**scripts.LEO_STATE).replace(
        "Sat.CoordinateSystem = EarthMJ2000Eq;",
        "Sat.CoordinateSystem = MoonFrame;",
    ).replace(
        "Create Spacecraft Sat;",
        "Create CoordinateSystem MoonFrame;\nMoonFrame.Origin = Luna;\nMoonFrame.Axes = MJ2000Eq;\nCreate Spacecraft Sat;",
    )
    with pytest.raises(ValueError, match="centred on the force model's central body"):
        GmatDynamics(str(scripts.write_script(tmp_path, "moon.script", moon_frame)), gmat_root=gmat_root)


@needs_gmat
def test_cli_with_a_gmat_script(gmat_root, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("GMAT_ROOT", gmat_root)
    text = scripts.HIGH_FIDELITY.format(error_control="RSSStep", **scripts.LEO_STATE) + "BeginMissionSequence;\n"
    scripts.write_script(tmp_path, "mission.script", text)
    config = tmp_path / "mission.toml"
    config.write_text(
        """
[dynamics]
type = "gmat"
script = "mission.script"

[mission]
final_time = 5400.0
final_target = ["RMAG", "INC"]

[mission.initial_dispersion]
sigmas = [0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4]
frame = "RIC"

[error_models.Engine]
proportional_magnitude_sigma = 0.002
proportional_pointing_sigma = 0.002

[navigation.Tracking]
sigmas = [0.02, 0.02, 0.02, 2e-5, 2e-5, 2e-5]

[[maneuver]]
name = "Burn"
time = 900.0
dv = [0.0, 0.05, 0.01]
error_model = "Engine"

[[maneuver]]
name = "TCM"
time = 2400.0
error_model = "Engine"
navigation = "Tracking"
target = ["X", "Y", "Z"]
target_time = 5400.0

[analysis]
samples = 24
seed = 1
workers = 2
dv_samples = 4000
report = "report.json"
""",
        encoding="utf-8",
    )
    assert main(["run", str(config)]) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["dynamics"]["type"] == "GmatDynamics" and report["dynamics"]["central_body"] == "Earth"
    assert report["mission"]["initial_state"] == [7000.0, 0.0, 1300.0, 0.0, 7.35, 1.0]  # taken from the script
    assert report["monte_carlo"]["n_samples"] == 24 and report["monte_carlo"]["workers"] == 2
    assert report["monte_carlo"]["failed_samples"] == 0 and report["sigma_point"]["linear_ok"] is True
    assert "engines agree" in capsys.readouterr().out


@needs_gmat
def test_lunar_flyby_example(gmat_root, monkeypatch):
    """The shipped lunar example loads and gives the dispersions of validation test 4."""
    from pathlib import Path

    from gmat_uq.config import load_config

    monkeypatch.setenv("GMAT_ROOT", gmat_root)
    config = load_config(Path(__file__).resolve().parent.parent / "examples" / "lunar_flyby" / "lunar_flyby.toml")
    assert [m.name for m in config.mission.maneuvers] == ["TLI", "TCM1", "TCM2", "TCM3"]
    assert config.mission.maneuvers[1].target.names == ("BdotT", "BdotR", "RMAG")
    assert config.mission.maneuvers[1].target.origin is not None  # relative to the Moon
    lincov = run_lincov(config.mission, config.dynamics, dv_samples=2000, seed=1)
    b_plane_sigma = np.sqrt(np.diag(lincov.target_covariance))
    assert np.allclose(b_plane_sigma, [0.6326, 0.6051], rtol=5e-3)  # km, as in docs/validation.md
    assert run_sigma_point_check(lincov, config.dynamics).linear_ok
    assert 3155.0 < 1e3 * lincov.dv_statistics()["total"]["quantile"]["value"] < 3170.0


@needs_gmat
def test_drag_coefficient_as_a_consider_parameter(gmat_root, tmp_path):
    """An uncertain Cd, with GMAT's drag model, gives the same dispersion in both engines."""
    from gmat_uq import ConsiderParameter

    script = scripts.write_script(tmp_path, "drag.script", scripts.DRAG)
    dynamics = GmatDynamics(str(script), gmat_root=gmat_root)
    assert dynamics.parameter_value("Cd") == 2.2 and dynamics.parameter_value("Cr") == 1.8
    with pytest.raises(ValueError, match="no numeric field"):
        dynamics.parameter_value("NoSuchField")
    with pytest.raises(ValueError, match="orbit state"):
        dynamics.parameter_value("X")
    state = dynamics.initial_state()
    # Setting a parameter for one propagation leaves the nominal untouched.
    reference = dynamics.propagate(state, 0.0, 10800.0)
    changed = dynamics.propagate(state, 0.0, 10800.0, parameters={"Cd": 2.42})
    assert np.linalg.norm(changed[:3] - reference[:3]) > 0.01
    assert np.array_equal(dynamics.propagate(state, 0.0, 10800.0), reference)
    _, stm_with = dynamics.propagate_stm(state, 0.0, 10800.0, parameters={"Cd": 2.42})
    final, _ = dynamics.propagate_stm(state, 0.0, 10800.0)
    assert np.array_equal(final, reference)

    mission = Mission(
        tuple(state),
        final_time=21600.0,
        initial_dispersion=StateDispersion.from_sigmas([1e-3] * 3 + [1e-6] * 3),  # 1 m, 1 mm/s
        consider=(ConsiderParameter("Cd", 0.22),),  # 10%
        report_times=(10800.0,),
    )
    lincov = run_lincov(mission, dynamics, dv_samples=0)
    quiet = run_lincov(Mission(mission.initial_state, mission.final_time, initial_dispersion=mission.initial_dispersion), dynamics, dv_samples=0)
    # The parameter adds, in quadrature, exactly the position change that +/- one sigma of Cd causes.
    high = dynamics.propagate(state, 0.0, 21600.0, parameters={"Cd": 2.42})
    low = dynamics.propagate(state, 0.0, 21600.0, parameters={"Cd": 1.98})
    effect = 0.5 * np.linalg.norm(high[:3] - low[:3])
    assert effect > quiet.position_sigma_rss()  # drag dominates
    assert np.isclose(lincov.position_sigma_rss() ** 2 - quiet.position_sigma_rss() ** 2, effect**2, rtol=1e-3)  # chained over two arcs vs one
    monte_carlo = run_monte_carlo(mission, dynamics, 200, seed=5, nominal=lincov.nominal, workers=2)
    assert monte_carlo.n_failed == 0
    ratio = np.sqrt(np.diag(monte_carlo.final_covariance))[:3] / lincov.sigmas()[:3]
    assert np.all(np.abs(ratio - 1.0) < 0.2)  # 200 samples: about +/- 10% of sampling noise per sigma
    assert run_sigma_point_check(lincov, dynamics).linear_ok


@needs_gmat
def test_navigation_covariance_from_a_gmat_estimator_run(gmat_root, tmp_path):
    """Run GMAT's own batch-estimation sample, report the covariance, and use it as navigation knowledge."""
    from pathlib import Path

    from gmat_uq import NavigationErrorModel

    sample = Path(gmat_root) / "samples" / "Navigation" / "Ex_Simulate_and_Process_GpsPosVec_data.script"
    if not sample.exists():
        pytest.skip("this GMAT installation has no GpsPosVec navigation sample")
    knowledge_file = tmp_path / "knowledge.txt"
    report = (
        "Create ReportFile UQCovReport;\n"
        f"UQCovReport.Filename = '{knowledge_file.as_posix()}';\n"
        "UQCovReport.WriteHeaders = False;\nUQCovReport.Precision = 16;\nUQCovReport.WriteReport = False;\n\n"
    )
    text = sample.read_text(encoding="utf-8", errors="replace").replace("BeginMissionSequence", report + "BeginMissionSequence", 1)
    text = text.rstrip() + "\nReport UQCovReport EstSat.EarthMJ2000Eq.OrbitErrorCovariance;\n"
    script = scripts.write_script(tmp_path, "estimate.script", text)
    scripts.run_gmat_console(gmat_root, script)

    knowledge = NavigationErrorModel.from_file(knowledge_file)
    sigmas = np.sqrt(np.diag(knowledge.covariance))
    assert np.all(sigmas[:3] < 1e-2) and np.all(sigmas[:3] > 1e-6)  # metre-level GPS solution, in km

    dynamics = GmatDynamics(str(scripts.write_script(tmp_path, "pm.script", scripts.POINT_MASS.format(**scripts.LEO_STATE))), gmat_root=gmat_root)
    mission = Mission(
        tuple(dynamics.initial_state()),
        final_time=5400.0,
        maneuvers=(Maneuver("TCM", 1800.0, target=ParameterTarget.position(), target_time=5400.0, navigation=knowledge),),
        initial_dispersion=StateDispersion.from_sigmas([1.0] * 3 + [1e-3] * 3),
    )
    result = run_lincov(mission, dynamics, dv_samples=0)
    # With perfect execution, the delivery error at the target comes from the knowledge error only.
    assert 0.0 < result.position_sigma_rss() < 1.0


def test_resources_only_script_drops_the_plugin_resources():
    text = (
        "Create Spacecraft Sat;\nCreate ManeuverErrorModel Engine;\nEngine.FixedMagnitudeSigma = 1e-5;\n"
        "Create StateDispersion Injection, Spare;\nInjection.Sigmas = [1 1 1 1e-3 1e-3 1e-3];\n"
        "Create NavigationErrorModel Nav;\nCreate UncertainManeuver TCM1;\nTCM1.Target = {X, Y, Z};\n"
        "Create UncertaintyAnalysis UQ;\nUQ.Spacecraft = Sat;\nBeginMissionSequence;\nRunUncertaintyAnalysis UQ;\n"
    )
    cleaned = resources_only_script(text)
    assert "Create Spacecraft Sat;" in cleaned
    for gone in ("Engine", "Injection", "Spare", "Nav", "TCM1", "UQ", "RunUncertaintyAnalysis"):
        assert gone not in cleaned, gone


@needs_gmat
def test_times_from_another_epoch_are_shifted_to_the_script_epoch(gmat_root, tmp_path):
    from gmat_uq.config import build_from_dict

    script = scripts.write_script(tmp_path, "pm.script", scripts.POINT_MASS.format(**scripts.LEO_STATE))
    dynamics = GmatDynamics(str(script), gmat_root=gmat_root)
    later = dynamics.epoch_a1mjd + 0.5  # twelve hours after the script epoch
    data = {
        "dynamics": {"type": "gmat", "script": str(script), "gmat_root": gmat_root},
        "mission": {"epoch_a1mjd": later, "initial_state": [7000.0, 0, 0, 0, 7.5, 0.5], "final_time": 3000.0, "report_times": [1000.0]},
        "maneuver": [{"name": "TCM", "time": 600.0, "target": ["X", "Y", "Z"], "target_time": 3000.0}],
    }
    config = build_from_dict(data, tmp_path)
    mission = config.mission
    assert np.isclose(mission.initial_time, 43200.0) and np.isclose(mission.final_time, 46200.0)
    assert np.isclose(mission.maneuvers[0].time, 43800.0) and np.isclose(mission.maneuvers[0].target_time, 46200.0)
    assert np.allclose(mission.report_times, [44200.0])
    assert run_lincov(mission, config.dynamics, dv_samples=0).nominal.times[0] == mission.initial_time
