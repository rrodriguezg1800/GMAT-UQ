# The GMAT plugin

The GMAT-UQ plugin lets you write an uncertainty analysis in GMAT's own script
language and run it from the mission sequence:

```matlab
BeginMissionSequence;
Propagate Prop(Sat) {Sat.ElapsedSecs = 600};
RunUncertaintyAnalysis UQ;
Report Results UQ.DeltaV99 UQ.FinalPositionSigma UQ.LinearTrusted;
```

When `RunUncertaintyAnalysis` executes, it takes the spacecraft's epoch and
state *at that moment*, runs the GMAT-UQ engines with the force model and
propagator of the same script, and stores the results in GMAT parameters
(`UQ.DeltaV99` and others) that the rest of the sequence can report or assign.

## How it works

The plugin is a GMAT plugin library (C++) that adds five resources, one
command and eight parameters. It does no numerical work itself. The command
writes the analysis to a GMAT-UQ configuration file and runs the GMAT-UQ
Python package, which drives GMAT through its API to propagate. So the
plugin needs, at run time:

- the `gmat-uq` Python package installed (`pip install gmat-uq/` from this repository);
- a Python version for which GMAT ships its API (`bin/gmatpy/_py3xx`);
- the `python` command (`python3` on Linux and macOS) on the path, or
  `UQ.PythonExecutable` set to the interpreter to use.

The files the command writes go to GMAT's output folder:
`<analysis>_config.toml` (the configuration it built),
`<analysis>_log.txt` (the engine's output), `<analysis>_results.txt` and the
JSON report (`<analysis>.json`, or `UQ.ReportFile`).

## Getting the library

**Built by continuous integration.** Every push builds the plugin against the
official GMAT R2026a release on Linux and Windows and tests it inside GMAT.
The libraries are kept as artifacts of the "GMAT-UQ with GMAT" workflow
(`libGmatUQ-windows-R2026a`, `libGmatUQ-linux-R2026a`).

**Building it yourself, against an installed GMAT.** You need CMake, a C++17
compiler (Visual Studio 2022 on Windows), the GMAT source tree checked out at
the tag of your GMAT release, and the CSPICE headers from NAIF:

```
git checkout R2026a -- src/base src/gmatutil        # in a clone of GMAT, or use a separate checkout
cmake -S plugins/GmatUQPlugin -B build-uq -DCMAKE_BUILD_TYPE=Release ^
      -DGMAT_SOURCE_DIR=<GMAT source at R2026a> -DGMAT_INSTALL_DIR=<GMAT install> ^
      -DCSPICE_INCLUDE_DIR=<cspice/include>
cmake --build build-uq --config Release
```

The headers must come from the same release as the installed GMAT. GMAT's
`main` branch already has header changes after R2026a that alter the layout
of its classes, and a plugin built against them would not work with the
R2026a binaries. On Windows the build creates the import libraries the
release does not ship (`tools/make_import_lib.py`, run from a Visual Studio
developer prompt).

**Building it with GMAT.** Configure GMAT's own build with
`-DPLUGIN_GMATUQ=ON`.

## Installing

1. Copy `libGmatUQ.dll` (Windows) or `libGmatUQ.so` (Linux) into GMAT's `plugins` folder.
2. Add this line to `bin/gmat_startup_file.txt`, next to the other `PLUGIN` lines:

   ```
   PLUGIN                  = ../plugins/libGmatUQ
   ```
3. If `bin/api_startup_file.txt` exists, delete it; GMAT-UQ rebuilds it.

## Script reference

Times are seconds after the moment `RunUncertaintyAnalysis` executes. Lengths
are km, velocities km/s, pointing angles rad.

### ManeuverErrorModel

The Gates execution-error model.

| Field | Unit | Default |
| --- | --- | --- |
| `FixedMagnitudeSigma` | km/s | 0 |
| `ProportionalMagnitudeSigma` | fraction of the delta-V | 0 |
| `FixedPointingSigma` | km/s per axis | 0 |
| `ProportionalPointingSigma` | rad per axis | 0 |

### StateDispersion and NavigationErrorModel

`StateDispersion` is the dispersion of the state when the analysis starts;
`NavigationErrorModel` is the knowledge error at a correction.

| Field | Meaning |
| --- | --- |
| `Sigmas` | six values (km, km/s), or two (position, velocity) |
| `CovarianceFile` | a 6x6 matrix as a GMAT `ReportFile` writes `Sat.EarthMJ2000Eq.OrbitErrorCovariance` |
| `Frame` | `MJ2000Eq` (default), `RIC` or `VNB` |

Give either `Sigmas` or `CovarianceFile`.

### UncertainManeuver

| Field | Meaning |
| --- | --- |
| `Time` | when the maneuver happens |
| `DeltaV` | nominal delta-V, MJ2000Eq axes; leave it out for a purely statistical correction |
| `ErrorModel` | a `ManeuverErrorModel` |
| `Target` | makes the maneuver a correction, for example `{BdotT, BdotR}` or `{X, Y, Z}` |
| `TargetBody` | evaluate the target relative to this body, for example `Luna` |
| `TargetTime` | when the target is evaluated |
| `Navigation` | a `NavigationErrorModel` (corrections only) |

### UncertaintyAnalysis

| Field | Default | Meaning |
| --- | --- | --- |
| `Spacecraft`, `Propagator` | required | the resources to analyse with |
| `Duration` | required | length of the analysis |
| `InitialDispersion` | none | a `StateDispersion` |
| `Maneuvers` | none | `UncertainManeuver` resources |
| `ConsiderParameters`, `ConsiderSigmas` | none | uncertain Spacecraft fields and their sigmas, for example `{Cd}` and `[0.22]` |
| `FinalTarget`, `FinalTargetBody` | none | also report the final dispersion in these parameters |
| `Methods` | all | any of `LinCov`, `SigmaPoint`, `MonteCarlo` |
| `Samples`, `Seed`, `Workers` | 1000, 0, 1 | Monte Carlo settings |
| `Guidance` | `Nonlinear` | or `Linear` |
| `DeltaVSamples` | 20000 | samples of the linear system for delta-V statistics |
| `NonlinearityThreshold` | 0.1 | sigma-point flag level |
| `Probability`, `Confidence` | 0.99, 0.95 | delta-V quantile and its interval |
| `ReportFile` | output folder | path of the JSON report |
| `PythonExecutable` | `python` / `python3` | interpreter that has `gmat-uq` |

### RunUncertaintyAnalysis

```matlab
RunUncertaintyAnalysis UQ;
```

The resources are checked when the mission starts; an error names the field
at fault, for example `UQ.Spacecraft is not set`. If the engine fails, GMAT
stops with the last lines of its log.

### Result parameters

| Parameter | Unit | Meaning |
| --- | --- | --- |
| `UQ.DeltaV99` | km/s | delta-V 99th percentile, linear covariance |
| `UQ.DeltaVMean` | km/s | mean total delta-V, linear covariance |
| `UQ.MonteCarloDeltaV99` | km/s | delta-V 99th percentile, Monte Carlo |
| `UQ.FinalPositionSigma` | km | final position dispersion, RSS of the three 1-sigma axes |
| `UQ.FinalVelocitySigma` | km/s | the same for velocity |
| `UQ.NonlinearityIndex` | | largest sigma-point index |
| `UQ.LinearTrusted` | | 1 if no arc is flagged, 0 otherwise |
| `UQ.EnginesAgree` | | 1 if linear covariance and Monte Carlo agree, 0 otherwise |

A parameter is -1 when its engine did not run.

## Testing

`plugins/GmatUQPlugin/test/check_plugin.py <GMAT root> <plugin library>`
installs the plugin, runs `uq_plugin_test.script` with GMAT's console
application, runs the same analysis directly from Python and checks that the
numbers GMAT reports are the same. It also checks that a missing field and an
unknown method are reported. Continuous integration runs it on Linux and
Windows.

## Limits

- The spacecraft must use an MJ2000Eq coordinate system centred on the force
  model's central body.
- Burns are given as delta-V vectors; `ImpulsiveBurn` resources are not read.
- Each run starts a Python process and loads GMAT again through its API,
  which adds a few seconds.
