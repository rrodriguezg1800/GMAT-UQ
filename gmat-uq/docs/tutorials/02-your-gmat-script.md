# Tutorial 2: analysing a mission defined in a GMAT script

GMAT-UQ can take the spacecraft, force model and propagator from your own
GMAT script, and let GMAT do every propagation. This tutorial runs the lunar
example, then adds the two things a real analysis usually needs: an uncertain
model parameter and a navigation error that comes from a GMAT estimation run.

You need GMAT R2026a or R2025a, and a Python version for which your GMAT ships
its API (look in GMAT's `bin/gmatpy` folder: `_py312` means Python 3.12).

## 1. Point GMAT-UQ at GMAT

```
set GMAT_ROOT=C:\path\to\GMAT          (Windows)
export GMAT_ROOT=/path/to/GMAT         (Linux, macOS)
```

`GMAT_ROOT` is the folder that contains `bin`. You can also give it in the
configuration (`gmat_root` under `[dynamics]`).

## 2. What GMAT-UQ needs from your script

Open [`examples/lunar_flyby/cislunar.script`](../../examples/lunar_flyby/cislunar.script).
GMAT-UQ uses three resources from it:

- the `Spacecraft` (default name `Sat`), for the epoch and, unless the
  configuration gives one, the initial state;
- the `Propagator` (default name `Prop`) and its `ForceModel`.

The spacecraft must use an MJ2000Eq coordinate system centred on the force
model's central body (`EarthMJ2000Eq` here). GMAT-UQ never runs the mission
sequence and never loads `ReportFile`, `EphemerisFile` or plot resources, so
analysing a script never touches the files it writes.

## 3. Run the lunar example

[`lunar_flyby.toml`](../../examples/lunar_flyby/lunar_flyby.toml) describes a
translunar injection from a 185 km parking orbit and three corrections that
hold the lunar B-plane (`BdotT`, `BdotR`) and the distance to the Moon:

```toml
[[maneuver]]
name = "TCM1"
time = 21600.0
error_model = "Thrusters"
navigation = "Tracking1"
target = { parameters = ["BdotT", "BdotR", "RMAG"], body = "Luna" }
target_time = 333691.4888800655
```

`body = "Luna"` evaluates the target relative to the Moon, with GMAT's own
ephemeris and gravitational parameter.

```
cd gmat-uq/examples/lunar_flyby
gmat-uq run lunar_flyby.toml --samples 300 --workers 6
```

```
Linear covariance
  final position sigma (x, y, z) [km]   : 10.86, 34.27, 20.71
  nominal delta-V [m/s]                  : 3140.45
  delta-V 99 [m/s]                      : 3161.97
  final target sigmas                    : BdotT=0.6326, BdotR=0.6051

Sigma-point check
  verdict                                : linear results trusted

Monte Carlo (300 samples, nonlinear guidance, 6 worker(s), 19.3 s)
  delta-V 99 [m/s]                      : 3162.4 (95% interval 3159.71 to Infinity)

Linear covariance against Monte Carlo
  verdict                                : engines agree
```

The flyby is delivered to about 0.6 km in each B-plane axis, while the
spacecraft's position at the target time is dispersed by tens of kilometres
along the flight path: the corrections control the B-plane, not the timing.
The upper end of the Monte Carlo interval is infinite because 300 samples
cannot bound a 99th percentile from above with 95% confidence (299 is the
minimum, and an interval needs more); the validation report uses 2000.

## 4. Add an uncertain model parameter

Drag coefficients, reflectivity coefficients and areas are rarely known well.
A *consider* parameter is uncertain, constant over the mission, and not
corrected by the flight software. With GMAT dynamics, any numeric field of the
script's spacecraft can be one:

```toml
[[consider]]
name = "Cd"
sigma = 0.22          # 10% of 2.2
```

A six-hour coast in a 300 km orbit (the `DRAG` script in
`tests/gmat_scripts.py`, which uses GMAT's exponential atmosphere), with 1 m
and 1 mm/s of initial dispersion:

```
Linear covariance
  final position sigma (x, y, z) [km]   : 0.2008, 0.09757, 0.05963
Monte Carlo (300 samples, nonlinear guidance, 6 worker(s), 11.3 s)
  final position sigma (x, y, z) [km]   : 0.2141, 0.1041, 0.06362
  verdict                                : engines agree
```

Without the `Cd` uncertainty the same coast ends with about 0.11 km of
position dispersion (RSS); with it, 0.23 km. The test suite checks that the
difference is exactly what ±1 sigma of `Cd` does to a GMAT propagation.

## 5. Use the navigation error of a real estimation run

The knowledge error at a correction is an input. If you estimate the orbit
with GMAT, report the estimated covariance and give the file to GMAT-UQ.
In the estimation script, after `RunEstimator`:

```matlab
Create ReportFile KnowledgeReport;
KnowledgeReport.Filename = 'knowledge.txt';
KnowledgeReport.WriteHeaders = False;
KnowledgeReport.Precision = 16;
KnowledgeReport.WriteReport = False;

BeginMissionSequence;
RunEstimator BLS;
Report KnowledgeReport EstSat.EarthMJ2000Eq.OrbitErrorCovariance;
```

The covariance is at the estimation epoch. To use it at a maneuver, propagate
it there first, with GMAT's covariance propagation, and report it then:

```matlab
Propagate Prop(EstSat, 'Covariance') {EstSat.ElapsedDays = 2};
Report KnowledgeReport EstSat.EarthMJ2000Eq.OrbitErrorCovariance;
```

(GMAT propagates covariance only with fixed steps: `ErrorControl = None` on
the force model.) Then, in the GMAT-UQ configuration:

```toml
[navigation.FromEstimator]
covariance_file = "knowledge.txt"     # the last matrix in the file; "block" chooses another
```

The test suite runs GMAT's own `Ex_Simulate_and_Process_GpsPosVec_data`
sample this way and uses its covariance in an analysis.

## 6. When the linear answer cannot be trusted

If the sigma-point check flags an arc, the summary says
`NONLINEAR ARCS FOUND - prefer Monte Carlo` and `gmat-uq` exits with code 2.
The report lists the flagged arcs. Typical causes are close planetary flybys
and long arcs in sensitive orbits; in the validation report, a
near-rectilinear halo orbit is flagged at its perilune passages. Use the Monte
Carlo results there, with enough samples.

Next: [Tutorial 3](03-gmat-plugin.md) writes the same kind of analysis inside
a GMAT script.
