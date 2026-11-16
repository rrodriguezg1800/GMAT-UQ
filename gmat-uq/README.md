# GMAT-UQ

Linear covariance and navigation-aware Monte Carlo analysis for
[GMAT](https://github.com/nasa/GMAT) missions.

GMAT-UQ answers two questions a mission design review asks and GMAT alone does
not answer:

- **How much delta-V margin does the trajectory need?** It reports delta-V 99
  with a confidence interval.
- **How accurately is the spacecraft delivered?** It reports the dispersion
  covariance along the trajectory and at the target, for example as an ellipse
  in the lunar B-plane.

The inputs are the three error sources that drive both answers: the dispersion
of the initial state, the execution error of every maneuver (Gates model) and
the navigation error at every correction maneuver.

GMAT does the propagation. GMAT-UQ reads the spacecraft, force model and
propagator from an ordinary GMAT script and drives them through the GMAT Python
API. A built-in two-body model lets everything run without GMAT too.

## The three engines

| Engine | What it gives | Cost |
| --- | --- | --- |
| Linear covariance | Dispersion covariance through burns and corrections; delta-V statistics | one nominal propagation |
| Sigma-point check | A flag on every arc where the linear answer cannot be trusted | 12 propagations per arc |
| Closed-loop Monte Carlo | The same statistics with the full nonlinear dynamics; each sample re-targets its corrections from an imperfect estimate | thousands of propagations, in parallel |

When the linear covariance and Monte Carlo both run, GMAT-UQ compares them and
says whether they agree. [docs/theory.md](docs/theory.md) has the equations and
the assumptions.

## Status

Version 1.0.0. Tested against GMAT R2026a on Windows with Python 3.13.

| Part of the design | State |
| --- | --- |
| Error models: initial dispersion, Gates execution error, navigation error, process noise | done |
| Linear covariance engine, with delta-V statistics | done |
| Sigma-point check | done |
| Closed-loop Monte Carlo, parallel and reproducible | done |
| Statistics and consistency report, JSON output, command line | done |
| GMAT in the loop through the GMAT Python API | done |
| Validation tests 1 to 7 | done, all pass ([docs/validation.md](docs/validation.md)) |
| Validation test 8 (reproduce a published case) | **not done** |
| C++ plugin with script-level resources (`Create ManeuverErrorModel ...`) | **not done**; a TOML file plays that role |
| Linux and macOS | not tested; continuous integration runs the GMAT-free tests on Linux |

The C++ plugin of the original design is the main gap. It needs GMAT's build
toolchain, and no untested C++ is shipped here. The configuration file mirrors
the resources the plugin would add, so moving to it later changes the syntax,
not the models.

## Install

Python 3.11 or newer, NumPy and SciPy.

```
pip install -e gmat-uq            # from the repository root
pip install -e "gmat-uq[test]"    # with pytest
```

To use GMAT, install [GMAT R2025a or R2026a](https://sourceforge.net/projects/gmat/files/GMAT/)
and point `GMAT_ROOT` at its top-level folder, the one that contains `bin`:

```
set GMAT_ROOT=C:\path\to\GMAT        (Windows)
export GMAT_ROOT=/path/to/GMAT       (Linux, macOS)
```

Your Python version must be one GMAT ships an API for (3.9 to 3.14 in R2026a;
GMAT-UQ itself needs 3.11 or newer). Only R2026a has been tested.

## Quick start without GMAT

```
gmat-uq run gmat-uq/examples/two_burn_transfer.toml
```

This analyses a transfer between two circular orbits with two burns and two
corrections, and prints:

```
Linear covariance
  final position sigma (x, y, z) [km]   : ...
  nominal delta-V [m/s]                  : 1751.1
  delta-V 99 [m/s]                      : 1777.5
Sigma-point check
  verdict                                : linear results trusted
Monte Carlo (1000 samples, nonlinear guidance, 4 worker(s), ...)
  delta-V 99 [m/s]                      : ... (95% interval ... to ...)
Linear covariance against Monte Carlo
  verdict                                : engines agree
```

The full results go to a JSON report.

## Quick start with GMAT

```
gmat-uq run gmat-uq/examples/lunar_flyby/lunar_flyby.toml
```

This is a translunar injection with three corrections that hold the lunar
B-plane. GMAT propagates it with the force model in
[`examples/lunar_flyby/cislunar.script`](examples/lunar_flyby/cislunar.script).
The part of the configuration that describes the uncertainty is short:

```toml
[dynamics]
type = "gmat"
script = "cislunar.script"          # your GMAT script: spacecraft, force model, propagator

[mission.initial_dispersion]
sigmas = [0.5, 2.0, 0.5, 2e-3, 5e-4, 5e-4]    # km and km/s: radial, in-track, cross-track
frame = "RIC"

[error_models.Thrusters]            # Gates model
fixed_magnitude_sigma = 1e-6        # km/s
proportional_magnitude_sigma = 0.005
fixed_pointing_sigma = 1e-6         # km/s per axis
proportional_pointing_sigma = 0.005 # rad per axis

[navigation.Tracking1]
sigmas = [1.0, 1.0, 1.0, 1e-5, 1e-5, 1e-5]

[[maneuver]]
name = "TCM1"
time = 21600.0                      # seconds from the script's epoch
error_model = "Thrusters"
navigation = "Tracking1"
target = { parameters = ["BdotT", "BdotR", "RMAG"], body = "Luna" }
target_time = 333691.4888800655
```

[docs/configuration.md](docs/configuration.md) is the full reference.

GMAT-UQ loads only the resources of your script. It never runs the mission
sequence and never loads `ReportFile`, `EphemerisFile` or plot resources, so
the files your script writes are not touched. The one thing it may add to a
GMAT installation is `bin/api_startup_file.txt`, the file GMAT's own
`api/BuildApiStartupFile.py` creates, if it is not there yet.

## Python API

```python
from gmat_uq import (GmatDynamics, Mission, Maneuver, ManeuverErrorModel, NavigationErrorModel,
                     ParameterTarget, StateDispersion, run_lincov, run_sigma_point_check, run_monte_carlo)

dynamics = GmatDynamics("mission.script")            # or TwoBody(398600.4415)
engine = ManeuverErrorModel(proportional_magnitude_sigma=0.005, proportional_pointing_sigma=0.005)
tracking = NavigationErrorModel.from_sigmas([1.0] * 3 + [1e-5] * 3)

mission = Mission(
    initial_state=dynamics.initial_state(),
    final_time=86400.0,
    maneuvers=(
        Maneuver("Burn", 3600.0, dv=(0.0, 0.05, 0.0), error_model=engine),
        Maneuver("TCM", 20000.0, error_model=engine, navigation=tracking,
                 target=ParameterTarget.position(), target_time=86400.0),
    ),
    initial_dispersion=StateDispersion.from_sigmas([0.1] * 3 + [1e-4] * 3),
)

lincov = run_lincov(mission, dynamics)
print(lincov.sigmas())                                   # final 1-sigma dispersion
print(lincov.dv_statistics()["total"]["quantile"])       # delta-V 99 and its interval
print(run_sigma_point_check(lincov, dynamics).linear_ok)
monte_carlo = run_monte_carlo(mission, dynamics, 2000, seed=1)   # add workers=6 to run in parallel
```

[examples/python_api.py](examples/python_api.py) is a complete script. On
Windows, a script that uses more than one worker needs the usual
`if __name__ == "__main__":` guard.

## Validation

Every figure below was measured with GMAT R2026a; the full report, with the
environment and every measurement, is [docs/validation.md](docs/validation.md).

| # | Test | Result |
| --- | --- | --- |
| 1 | Gates sampler against its analytic covariance, 100,000 draws | consistent; sigmas within 0.5% |
| 2 | No burns, against GMAT's native `Propagate Prop(Sat, 'Covariance')` | sigmas equal to 11 significant digits, with and without process noise |
| 3 | Linear covariance against Monte Carlo, two-burn transfer, 4,000 samples | sigmas within 1.3%; linear delta-V 99 inside the Monte Carlo interval |
| 4 | Linear covariance against Monte Carlo, lunar transfer with GMAT in the loop, 2,000 samples | sigmas within 0.9%; B-plane sigmas 0.63 km and 0.61 km linear, 0.64 km and 0.59 km Monte Carlo; linear delta-V 99 inside the interval |
| 5 | Near-rectilinear halo orbit, two revolutions | the sigma-point check flags the perilune arcs, where the engines differ by orders of magnitude |
| 6 | Same seed on 1, 4 and 16 worker processes, with and without GMAT | bit-identical |
| 7 | GMAT's files and a script's outputs before and after an analysis | byte-identical |
| 8 | A published case | not run |

On the lunar case the linear analysis, sigma-point check included, takes about
one second. A 2,000-sample Monte Carlo on one core takes about 500 s
(extrapolated from 200 timed samples), or about 2 minutes on six cores.

The GMAT parameters GMAT-UQ computes itself (`BdotT`, `BdotR`, the Keplerian
elements, `C3Energy` and others) match GMAT's own reports to 12 digits.

## Limitations

- Maneuvers are impulsive and happen at fixed times. Targets are evaluated at
  fixed times. Stopping conditions such as "at periapsis" are not modelled.
- The navigation error at each correction is an input. No navigation filter
  runs in the loop, and the error is taken as independent of the dispersion.
- Only the six orbit-state components are dispersed; uncertain drag or
  reflectivity coefficients are not.
- The spacecraft in the GMAT script must use an MJ2000Eq coordinate system
  centred on the force model's central body.
- A target must be differentiable at the nominal (not `ECC` on a circular
  orbit, for example).
- GMAT keeps one configuration per process, so a Python process can drive one
  GMAT script at a time. Monte Carlo workers each load their own.

## Development

```
cd gmat-uq
python -m pytest                              # 139 tests; 18 more run when GMAT_ROOT is set
python validation/run_validation.py           # the validation campaign; writes docs/validation.md
python validation/make_examples.py            # regenerates the example configurations
```

```
src/gmat_uq/
  models.py        error models (Gates, state dispersion, navigation, process noise)
  dynamics/        propagators: gmat.py (GMAT API), ode.py (two-body, J2, CR3BP)
  targets.py       target functions with GMAT's parameter names
  mission.py       mission definition, nominal trajectory, linear guidance law
  lincov.py        linear covariance engine and linear sampler
  sigma.py         sigma-point check
  guidance.py      nonlinear corrector
  montecarlo.py    closed-loop Monte Carlo engine
  sampling.py      reproducible random streams
  stats.py         quantile intervals, covariance comparison, ellipses
  consistency.py   linear covariance against Monte Carlo
  config.py, cli.py, report.py, analysis.py
tests/             unit and integration tests
validation/        validation cases and campaign
examples/          ready-to-run configurations
docs/              theory, configuration reference, validation report
```

## Relation to GMAT

GMAT-UQ is an independent project. It is not part of GMAT and has not been
reviewed or endorsed by NASA or the GMAT development team. It builds on two
existing pieces and does not replace either: GMAT's own covariance propagation
(used as the reference in test 2) and the idea of process-isolated parallel
runs used by third-party tools such as `gmat-sweep`.

## License

Apache License, Version 2.0, the license of GMAT. See
[`License.txt`](../License.txt) in the repository root.
