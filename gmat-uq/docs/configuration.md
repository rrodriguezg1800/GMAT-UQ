# Configuration file reference

`gmat-uq run mission.toml` reads one TOML file (JSON with the same structure
also works). Relative paths in the file are resolved against the file's folder.
Lengths are in km, velocities in km/s, times in seconds from the mission epoch,
angles in degrees.

The tables correspond to the resources of the GMAT-UQ design:

| Design resource | Configuration table |
| --- | --- |
| `StateDispersion` | `[mission.initial_dispersion]` |
| `ManeuverErrorModel` | `[error_models.<name>]` |
| `NavigationErrorModel` | `[navigation.<name>]` |
| `UncertaintyAnalysis` | `[analysis]` |

## `[dynamics]`

| Key | Meaning |
| --- | --- |
| `type` | `"gmat"`, `"two_body"`, `"two_body_j2"` or `"cr3bp"` |

For `type = "gmat"`:

| Key | Default | Meaning |
| --- | --- | --- |
| `script` | required | GMAT script that creates the spacecraft, force model and propagator |
| `spacecraft` | `"Sat"` | name of the `Spacecraft` resource |
| `propagator` | `"Prop"` | name of the `Propagator` resource |
| `gmat_root` | `GMAT_ROOT` environment variable | top-level GMAT folder |
| `stm_method` | `"auto"` | `"variational"`, `"finite_difference"` or `"auto"` |

GMAT-UQ loads the resources of the script and ignores its mission sequence.
Output resources (`ReportFile`, `EphemerisFile`, plots) are not loaded, so the
files your script writes are never touched. The spacecraft must use an
MJ2000Eq coordinate system centred on the force model's central body. Time
zero is the spacecraft epoch.

For the built-in reference dynamics: `mu` (`two_body`, `two_body_j2`), `j2` and
`radius` (`two_body_j2`), `mass_ratio` (`cr3bp`). The defaults are Earth values
and the Earth-Moon mass ratio.

## `[mission]`

| Key | Default | Meaning |
| --- | --- | --- |
| `initial_state` | the script's spacecraft state (GMAT only) | `[x, y, z, vx, vy, vz]` |
| `initial_time` | `0.0` | start of the analysis |
| `final_time` | required | end of the analysis |
| `report_times` | none | extra times at which the dispersion is reported |
| `final_target` | none | target in which the final dispersion is also reported |
| `epoch_a1mjd` | the script's epoch | GMAT dynamics only: every time in the file is counted from this A.1 modified Julian date instead of the script's epoch (the GMAT plugin uses it to start an analysis in the middle of a mission) |

### `[mission.initial_dispersion]`

Exactly one of:

- `sigmas = [6 values]`, or `[position, velocity]`;
- `covariance = [[6x6]]`;
- `covariance_file = "file.txt"`: a 6x6 matrix as text, either six rows of six
  numbers or 36 numbers on one row. This is what a GMAT `ReportFile` writes for
  `Sat.EarthMJ2000Eq.OrbitErrorCovariance`, for example after `RunEstimator`.
  Lines starting with `#` or `%` are comments. A file with several matrices
  (a report written at several times) gives the last one, or the one chosen
  with `block` (0 for the first, -1 for the last).

`frame` is `"inertial"` (default), `"RIC"` (radial, in-track, cross-track) or
`"VNB"` (velocity, normal, bi-normal), built from the initial state.

### `[mission.process_noise]`

| Key | Default | Meaning |
| --- | --- | --- |
| `accel_sigma` | required | three values, km/s^(3/2), on the inertial axes |
| `max_step` | `600.0` | longest interval over which the noise formula is applied |

## `[error_models.<name>]`

The Gates model. Every key defaults to zero.

| Key | Unit |
| --- | --- |
| `fixed_magnitude_sigma` | km/s |
| `proportional_magnitude_sigma` | fraction of the commanded delta-V |
| `fixed_pointing_sigma` | km/s per axis |
| `proportional_pointing_sigma` | rad per axis |

## `[navigation.<name>]`

Knowledge error of the estimate used to design a correction. Same keys as
`[mission.initial_dispersion]`. A local `frame` is built from the nominal state
arriving at the maneuver that uses the model.

## `[[maneuver]]`

One table per maneuver, in any order.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | required | unique label |
| `time` | required | maneuver time |
| `dv` | `[0, 0, 0]` | nominal delta-V, inertial axes |
| `error_model` | none | name of an `[error_models.<name>]` table |
| `target` | none | makes the maneuver a closed-loop correction |
| `target_time` | required with `target` | when the target is evaluated |
| `navigation` | perfect knowledge | name of a `[navigation.<name>]` table |

## Targets

A target is a list of one to three parameter names:

```toml
target = ["X", "Y", "Z"]
```

or a table that also names a body, to evaluate the parameters relative to it
(GMAT dynamics only; the body's ephemeris and gravitational parameter come from
GMAT):

```toml
target = { parameters = ["BdotT", "BdotR", "RMAG"], body = "Luna" }
```

Available parameters, with GMAT's names and conventions:

`X`, `Y`, `Z`, `VX`, `VY`, `VZ`, `RMAG`, `VMAG`, `SMA`, `ECC`, `INC`, `RAAN`,
`AOP`, `TA`, `RadPer`, `RadApo`, `C3Energy`, `BdotT`, `BdotR`.

`final_target` uses the same syntax and may have any number of components.
A target must be differentiable at the nominal: do not target `ECC` on a
circular orbit or `RAAN` on an equatorial one.

## `[[consider]]`

Uncertain model parameters that are neither estimated nor corrected: the true
value is drawn once per Monte Carlo sample and kept for the whole mission,
while corrections are designed with the nominal value.

| Key | Meaning |
| --- | --- |
| `name` | with GMAT dynamics, a numeric Spacecraft field of the script (`Cd`, `Cr`, `DragArea`, `SRPArea`, `DryMass`...); with the reference models, `mu`, `j2`, `radius` or `mass_ratio` |
| `sigma` | 1-sigma error, in the parameter's own unit |

The nominal value comes from the script or the model.

## `[analysis]`

| Key | Default | Meaning |
| --- | --- | --- |
| `methods` | all three | any of `"lincov"`, `"sigma_point"`, `"monte_carlo"` |
| `samples` | `1000` | Monte Carlo samples |
| `seed` | `0` | master random seed |
| `workers` | `1` | worker processes for Monte Carlo |
| `guidance` | `"nonlinear"` | `"nonlinear"` or `"linear"` |
| `dv_samples` | `20000` | samples of the linear system for delta-V statistics |
| `threshold` | `0.1` | nonlinearity index above which an arc is flagged |
| `probability` | `0.99` | delta-V quantile |
| `confidence` | `0.95` | confidence of its interval |
| `report` | none | path of the JSON report |

## Command line

```
gmat-uq run mission.toml [--report FILE] [--methods lincov sigma_point monte_carlo]
                         [--samples N] [--workers N] [--seed N] [--guidance nonlinear|linear] [--quiet]
                         [--values FILE]
gmat-uq wilks [--probability 0.99] [--confidence 0.95]
gmat-uq --version
```

`--values` also writes the headline results as `key = value` lines (km and
km/s): `delta_v_99`, `delta_v_mean`, `monte_carlo_delta_v_99`,
`final_position_sigma`, `final_velocity_sigma`, `nonlinearity_index`,
`linear_trusted`, `engines_agree`. The GMAT plugin reads this file.

Exit codes: `0` success; `1` invalid input; `2` the analysis ran, but the
sigma-point check flagged an arc or the two engines disagree, so the linear
results should not be trusted.
