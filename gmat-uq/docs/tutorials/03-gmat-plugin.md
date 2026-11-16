# Tutorial 3: an uncertainty analysis inside a GMAT script

With the GMAT-UQ plugin installed, the analysis is part of the mission
sequence: GMAT flies the mission up to a point, analyses the uncertainty from
the state at that point, and the results are GMAT parameters you can report.

## 1. Install the plugin

Follow [the plugin page](../plugin.md#installing): copy `libGmatUQ.dll` (or
`libGmatUQ.so`) into GMAT's `plugins` folder, add

```
PLUGIN                  = ../plugins/libGmatUQ
```

to `bin/gmat_startup_file.txt`, and install the `gmat-uq` Python package for
a Python version your GMAT supports.

## 2. Describe the uncertainty with resources

This is [`plugins/GmatUQPlugin/test/uq_plugin_test.script`](../../../plugins/GmatUQPlugin/test/uq_plugin_test.script),
the script the continuous-integration tests run. After the usual spacecraft,
force model and propagator:

```matlab
Create StateDispersion Injection;
Injection.Sigmas = [0.1 0.3 0.1 1e-4 5e-5 5e-5];
Injection.Frame = RIC;

Create ManeuverErrorModel Engine;
Engine.FixedMagnitudeSigma = 1e-5;
Engine.ProportionalMagnitudeSigma = 0.003;
Engine.FixedPointingSigma = 1e-5;
Engine.ProportionalPointingSigma = 0.005;

Create NavigationErrorModel Tracking;
Tracking.Sigmas = [0.02 0.02 0.02 2e-5 2e-5 2e-5];

Create UncertainManeuver Raise;
Raise.Time = 300;
Raise.DeltaV = [0.0 0.02 0.01];
Raise.ErrorModel = Engine;

Create UncertainManeuver TCM;
TCM.Time = 2400;
TCM.ErrorModel = Engine;
TCM.Navigation = Tracking;
TCM.Target = {X, Y, Z};
TCM.TargetTime = 5400;

Create UncertaintyAnalysis UQ;
UQ.Spacecraft = Sat;
UQ.Propagator = Prop;
UQ.Duration = 5400;
UQ.InitialDispersion = Injection;
UQ.Maneuvers = {Raise, TCM};
UQ.ConsiderParameters = {Cd};
UQ.ConsiderSigmas = [0.22];
UQ.Methods = {LinCov, SigmaPoint, MonteCarlo};
UQ.Samples = 40;
UQ.Seed = 7;
UQ.Workers = 2;
```

Times are seconds after the moment the analysis runs.

## 3. Run the analysis from the mission sequence

```matlab
Create Variable DV99;

BeginMissionSequence;
Propagate Prop(Sat) {Sat.ElapsedSecs = 600};
RunUncertaintyAnalysis UQ;
DV99 = UQ.DeltaV99;
Report Results UQ.DeltaV99 UQ.FinalPositionSigma UQ.LinearTrusted UQ.EnginesAgree;
```

GMAT prints, in its message window or console:

```
RunUncertaintyAnalysis UQ: running the GMAT-UQ engines
RunUncertaintyAnalysis UQ: delta-V 99 0.024194 km/s, final position sigma 0.334641 km, linear results trusted. Report: ...
```

and the report holds `0.02419415691503253 0.3346414296804706 1 1`: delta-V 99
of 24.2 m/s against 22.4 m/s nominal, 0.33 km of final position dispersion
(RSS), linear results trusted and confirmed by Monte Carlo.

These are the same numbers, to every digit, as the same analysis set up by
hand with the Python API from the state GMAT reached at 600 s; the plugin
test checks exactly that.

## 4. What happened

`RunUncertaintyAnalysis` read the spacecraft's epoch and state after the
600 s coast, wrote the analysis to `UQ_config.toml` in GMAT's output folder,
and ran `python -m gmat_uq run UQ_config.toml`. GMAT-UQ loaded the force
model and propagator of this same script through GMAT's API and ran the three
engines. The command then read the results back into the `UQ.*` parameters.
The full report is `UQ.json` in the output folder (or `UQ.ReportFile`), and
the engine's output is in `UQ_log.txt`.

## 5. Mistakes are reported where they are made

```
UQ.Spacecraft is not set
RunUncertaintyAnalysis: unknown method 'Guess' (use LinCov, SigmaPoint or MonteCarlo)
UQ: ConsiderParameters and ConsiderSigmas need the same number of entries
```

These are checked when the mission starts, before any propagation.
