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

The inputs are the error sources that drive both answers: the dispersion of
the initial state, the execution error of every maneuver (Gates model), the
navigation error at every correction maneuver (which can come straight from a
GMAT estimation run), process noise, and uncertain model parameters such as a
drag or reflectivity coefficient.

GMAT does the propagation. GMAT-UQ reads the spacecraft, force model and
propagator from an ordinary GMAT script and drives them through the GMAT Python
API. A built-in two-body model lets everything run without GMAT too.

There are two ways to use it: a configuration file (or the Python API), or the
GMAT plugin, which puts the analysis in GMAT's own script language:

```matlab
Propagate Prop(Sat) {Sat.ElapsedSecs = 600};
RunUncertaintyAnalysis UQ;
Report Results UQ.DeltaV99 UQ.FinalPositionSigma UQ.LinearTrusted;
```

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

Version 1.0.0. Every part of the original design is implemented, and all
eight validation tests pass ([docs/validation.md](docs/validation.md)).

| Tested with | How |
| --- | --- |
| GMAT R2026a, Windows, Python 3.13 | full test suite and validation campaign, locally |
| GMAT R2026a, Windows and Ubuntu, Python 3.12 | full test suite, plugin built and run inside GMAT, by continuous integration |
| GMAT R2025a, Windows, Python 3.12 | full test suite, locally |
| Python 3.11 to 3.13, Windows and Ubuntu, no GMAT | GMAT-free tests, by continuous integration |

Not tested: macOS, and the plugin with GMAT R2025a (it is built against the
R2026a headers; a plugin must match its GMAT release).

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

Your Python version must be one GMAT ships an API for (look in GMAT's
`bin/gmatpy` folder: 3.9 to 3.14 in R2026a, up to 3.12 in R2025a). GMAT-UQ
itself needs 3.11 or newer.

For the plugin, see [docs/plugin.md](docs/plugin.md).

## Documentation

- [Tutorial 1](docs/tutorials/01-first-analysis.md): a first analysis, without GMAT.
- [Tutorial 2](docs/tutorials/02-your-gmat-script.md): your own GMAT script, an uncertain Cd, and the navigation error of a GMAT estimation run.
- [Tutorial 3](docs/tutorials/03-gmat-plugin.md): the analysis inside a GMAT script, with the plugin.
- [Theory](docs/theory.md): the models, the equations and their assumptions.
- [Configuration reference](docs/configuration.md) and [plugin reference](docs/plugin.md).
- [Validation report](docs/validation.md).

## Quick start without GMAT

```
gmat-uq run gmat-uq/examples/two_burn_transfer.toml
```

This analyses a transfer between two circular orbits with two burns and two
corrections, and prints:

```
Linear covariance
  final position sigma (x, y, z) [km]   : 0.09085, 0.07245, 0.07063
  nominal delta-V [m/s]                  : 1751.1
  delta-V 99 [m/s]                      : 1777.72
Sigma-point check
  verdict                                : linear results trusted
Monte Carlo (1000 samples, nonlinear guidance, 4 worker(s), 8.2 s)
  delta-V 99 [m/s]                      : 1776.89 (95% interval 1776.09 to 1779.03)
Linear covariance against Monte Carlo
  verdict                                : engines agree
```

The full results go to a JSON report. [Tutorial 1](docs/tutorials/01-first-analysis.md)
walks through this output.

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
| 8 | The published Gates (1963) execution-error model: the report's construction and formula | reproduced; the report's two stated properties hold to rounding |

Test 8 of the original design asked for a complete published mission. The
public papers reviewed rely on navigation filters whose outputs GMAT-UQ takes
as inputs, so none could be reproduced number for number; that remains open.

On the lunar case the linear analysis, sigma-point check included, takes about
one second. A 2,000-sample Monte Carlo on one core takes about 500 s
(extrapolated from 200 timed samples), or about 2 minutes on six cores.

The GMAT parameters GMAT-UQ computes itself (`BdotT`, `BdotR`, the Keplerian
elements, `C3Energy` and others) match GMAT's own reports to 12 digits.

The plugin test runs an analysis written in GMAT's script language with
GMAT's console application and checks that GMAT reports the same numbers, to
every digit, as the same analysis set up directly in Python. Test coverage of
the Python package is 96% (with GMAT installed).

## Limitations

- Maneuvers are impulsive and happen at fixed times. Targets are evaluated at
  fixed times. Stopping conditions such as "at periapsis" are not modelled.
- The navigation error at each correction is an input. No navigation filter
  runs in the loop, and the error is taken as independent of the dispersion.
- Consider parameters are constant over the mission; time-varying parameter
  errors (Gauss-Markov processes) are not modelled.
- The plugin delegates the numerics to the Python package, so it needs Python
  and `gmat-uq` at run time, and each run reloads GMAT through its API.
- The spacecraft in the GMAT script must use an MJ2000Eq coordinate system
  centred on the force model's central body.
- A target must be differentiable at the nominal (not `ECC` on a circular
  orbit, for example).
- GMAT keeps one configuration per process, so a Python process can drive one
  GMAT script at a time. Monte Carlo workers each load their own.

## Development

```
cd gmat-uq
python -m pytest                              # 157 tests; 18 more run when GMAT_ROOT is set
python validation/run_validation.py           # the validation campaign; writes docs/validation.md
python validation/make_examples.py            # regenerates the example configurations
```

```
src/gmat_uq/
  models.py        error models (Gates, state dispersion, navigation, process noise, consider parameters)
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
docs/              tutorials, theory, configuration and plugin references, validation report
../plugins/GmatUQPlugin/   the GMAT plugin (C++), its build files and its end-to-end test
```

The plugin is built and tested by `.github/workflows/gmat-uq-plugin.yml`.

## Relation to GMAT

GMAT-UQ is an independent project. It is not part of GMAT and has not been
reviewed or endorsed by NASA or the GMAT development team. It builds on two
existing pieces and does not replace either: GMAT's own covariance propagation
(used as the reference in test 2) and the idea of process-isolated parallel
runs used by third-party tools such as `gmat-sweep`.

## License

Apache License, Version 2.0, the license of GMAT. See
[`License.txt`](../License.txt) in the repository root.
