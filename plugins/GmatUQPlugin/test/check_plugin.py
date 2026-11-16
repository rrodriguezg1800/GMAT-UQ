"""End-to-end test of the GMAT-UQ plugin.

1. Installs the plugin library into a GMAT installation (copies it into
   ``plugins/`` and adds it to ``bin/gmat_startup_file.txt``).
2. Runs ``uq_plugin_test.script`` with GMAT's console application: the
   analysis is written in GMAT's script language and run by the
   RunUncertaintyAnalysis command.
3. Runs the same analysis directly with the gmat_uq Python package and checks
   that the numbers GMAT reports are the same.
4. Checks that script errors are reported by GMAT with a clear message.

Usage: python check_plugin.py <GMAT root> <plugin library file>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def install(gmat_root: Path, library: Path) -> None:
    target = gmat_root / "plugins" / library.name
    shutil.copyfile(library, target)
    startup = gmat_root / "bin" / "gmat_startup_file.txt"
    lines = startup.read_text(encoding="utf-8", errors="replace").splitlines()
    entry = "PLUGIN                  = ../plugins/" + library.stem
    if entry not in lines:
        index = max(i for i, line in enumerate(lines) if line.startswith("PLUGIN")) + 1
        lines.insert(index, entry)
        startup.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # The API reads its own copy of the startup file; rebuild it from the edited one.
    api = gmat_root / "bin" / "api_startup_file.txt"
    if api.exists():
        api.unlink()
    print(f"installed {library.name} into {gmat_root}")


def console(gmat_root: Path) -> Path:
    for name in ("GmatConsole.exe", "GmatConsole"):
        path = gmat_root / "bin" / name
        if path.exists():
            return path
    raise SystemExit("GmatConsole not found")


def run_gmat(gmat_root: Path, script: Path) -> subprocess.CompletedProcess:
    binary = console(gmat_root)
    return subprocess.run([str(binary), "--run", str(script)], capture_output=True, text=True, timeout=1800, cwd=str(binary.parent))


def check_configuration(gmat_root: Path, script: Path, failures: list[str]) -> tuple[float, np.ndarray]:
    """Check what the plugin passed to the engine: the spacecraft state and epoch 600 s into the mission."""
    import tomllib

    from gmat_uq import GmatDynamics

    config = tomllib.loads((gmat_root / "output" / "UQ_config.toml").read_text(encoding="utf-8"))
    dynamics = GmatDynamics(str(script), gmat_root=str(gmat_root))
    start = (config["mission"]["epoch_a1mjd"] - dynamics.epoch_a1mjd) * 86400.0
    state = np.array(config["mission"]["initial_state"])
    if abs(start - 600.0) > 1e-5:
        failures.append(f"the analysis starts {start} s into the mission instead of 600 s")
    propagated = dynamics.propagate(dynamics.initial_state(), 0.0, start)
    if np.abs(propagated - state).max() > 1e-6:
        failures.append(f"the plugin passed the state {state}, GMAT propagates to {propagated}")
    return start, state


def expected_results(gmat_root: Path, script: Path, start: float, state: np.ndarray) -> dict:
    """The same analysis as the script, set up by hand with the Python API."""
    from gmat_uq import (
        ConsiderParameter,
        GmatDynamics,
        Maneuver,
        ManeuverErrorModel,
        Mission,
        NavigationErrorModel,
        ParameterTarget,
        StateDispersion,
        run_lincov,
        run_monte_carlo,
        run_sigma_point_check,
    )
    from gmat_uq.consistency import compare_engines

    dynamics = GmatDynamics(str(script), gmat_root=str(gmat_root))
    engine =ManeuverErrorModel(1e-5, 0.003, 1e-5, 0.005)
    mission = Mission(
        tuple(state),
        initial_time=start,
        final_time=start + 5400.0,
        maneuvers=(
            Maneuver("Raise", start + 300.0, (0.0, 0.02, 0.01), error_model=engine),
            Maneuver(
                "TCM", start + 2400.0, error_model=engine, target=ParameterTarget.position(), target_time=start + 5400.0,
                navigation=NavigationErrorModel.from_sigmas([0.02] * 3 + [2e-5] * 3),
            ),
        ),
        initial_dispersion=StateDispersion.from_local(np.diag(np.array([0.1, 0.3, 0.1, 1e-4, 5e-5, 5e-5]) ** 2), state, "RIC"),
        final_target=ParameterTarget(("RMAG", "INC"), dynamics.mu),
        consider=(ConsiderParameter("Cd", 0.22),),
    )
    lincov = run_lincov(mission, dynamics, dv_samples=5000, seed=7)
    sigma = run_sigma_point_check(lincov, dynamics, 0.1)
    monte_carlo = run_monte_carlo(mission, dynamics, 40, seed=7, workers=2, nominal=lincov.nominal)
    statistics = lincov.dv_statistics()
    return {
        "DV99": statistics["total"]["quantile"]["value"],
        "DVMean": statistics["total"]["mean"],
        "MCDV99": monte_carlo.dv_statistics()["total"]["quantile"]["value"],
        "PosSigma": lincov.position_sigma_rss(),
        "VelSigma": lincov.velocity_sigma_rss(),
        "Trusted": 1.0 if sigma.linear_ok else 0.0,
        "Agree": 1.0 if compare_engines(lincov, monte_carlo).agrees else 0.0,
        "Index": sigma.max_index,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    gmat_root = Path(argv[1]).resolve()
    library = Path(argv[2]).resolve()
    install(gmat_root, library)
    failures = []

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        results_file = folder / "results.txt"
        report_file = folder / "uq_report.json"
        text = (HERE / "uq_plugin_test.script").read_text(encoding="utf-8")
        text = text.replace("@REPORT_JSON@", report_file.as_posix()).replace("@RESULTS_TXT@", results_file.as_posix())
        script = folder / "uq_plugin_test.script"
        script.write_text(text, encoding="utf-8")

        completed = run_gmat(gmat_root, script)
        print(completed.stdout[-4000:])
        if completed.returncode != 0 or not results_file.exists():
            print(completed.stderr[-2000:])
            raise SystemExit("GMAT did not run the plugin test script")

        names = ["DV99", "DVMean", "MCDV99", "PosSigma", "VelSigma", "Trusted", "Agree", "Index"]
        reported = dict(zip(names, np.loadtxt(results_file).ravel()))
        report = json.loads(report_file.read_text(encoding="utf-8"))
        print("GMAT reported:", reported)
        start, state = check_configuration(gmat_root, script, failures)
        expected = expected_results(gmat_root, script, start, state)
        print("Python API:   ", expected)
        for name in names:
            if not np.isclose(reported[name], expected[name], rtol=1e-9, atol=1e-12):
                failures.append(f"{name}: GMAT {reported[name]!r}, Python {expected[name]!r}")
        if report["mission"]["consider"] != [{"name": "Cd", "sigma": 0.22}]:
            failures.append("the consider parameter did not reach the engine")
        if [m["name"] for m in report["mission"]["maneuvers"]] != ["Raise", "TCM"]:
            failures.append("the maneuvers did not reach the engine")
        if not 0.0 < reported["PosSigma"] < 50.0 or reported["DV99"] <= np.linalg.norm([0.0, 0.02, 0.01]):
            failures.append("implausible results")

        # A script error is reported by GMAT, with the reason.
        broken = folder / "broken.script"
        broken.write_text(text.replace("UQ.Spacecraft = Sat;\n", ""), encoding="utf-8")
        completed = run_gmat(gmat_root, broken)
        output = completed.stdout + completed.stderr
        if completed.returncode == 0 or "UQ.Spacecraft is not set" not in output:
            failures.append("a missing UQ.Spacecraft was not reported")
        unknown = folder / "unknown.script"
        unknown.write_text(text.replace("UQ.Methods = {LinCov, SigmaPoint, MonteCarlo};", "UQ.Methods = {LinCov, Guess};"), encoding="utf-8")
        completed = run_gmat(gmat_root, unknown)
        if completed.returncode == 0 or "unknown method 'Guess'" not in completed.stdout + completed.stderr:
            failures.append("an unknown method was not reported")

    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("GMAT-UQ plugin: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
