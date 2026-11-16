# Tutorial 1: a first analysis, without GMAT

You will analyse a transfer between two circular orbits, read the results, and
see how one error source changes them. This tutorial needs only Python and the
`gmat-uq` package; the dynamics are GMAT-UQ's built-in two-body model.

## 1. Install

From the root of the repository:

```
pip install -e gmat-uq
```

## 2. The mission

Open [`examples/two_burn_transfer.toml`](../../examples/two_burn_transfer.toml).
It describes:

- a spacecraft on a 7000 km circular orbit, inclined 28.5 degrees;
- `Burn1` at 600 s, which starts a transfer to 12000 km;
- two statistical corrections, `TCM1` and `TCM2`, that aim at the arrival
  position; they have no nominal delta-V, they only remove dispersions;
- `Burn2` at arrival, which circularises the orbit.

And the three error sources:

```toml
[mission.initial_dispersion]
sigmas = [0.2, 0.5, 0.2, 5e-4, 2e-4, 2e-4]    # km and km/s, radial, in-track, cross-track
frame = "RIC"

[error_models.Engine]                          # Gates model, used by all four burns
fixed_magnitude_sigma = 1e-05            # km/s
proportional_magnitude_sigma = 0.003     # fraction
fixed_pointing_sigma = 1e-05             # km/s per axis
proportional_pointing_sigma = 0.005      # rad per axis

[navigation.Tracking]                          # knowledge error at each correction
sigmas = [0.05, 0.05, 0.05, 5e-05, 5e-05, 5e-05]    # 50 m, 5 cm/s
```

## 3. Run it

```
cd gmat-uq/examples
gmat-uq run two_burn_transfer.toml
```

It prints this summary. The seed is fixed in the file, so you get the same
numbers, whatever the number of worker processes; only the run time differs.

```
Linear covariance
  final position sigma (x, y, z) [km]   : 0.09085, 0.07245, 0.07063
  final velocity sigma (x, y, z) [m/s]  : 3.672, 4.616, 4.826
  nominal delta-V [m/s]                  : 1751.1
  mean total delta-V [m/s]               : 1760.15
  delta-V 99 [m/s]                      : 1777.72
  final target sigmas                    : SMA=13.71, INC=0.03896, RAAN=0.06166

Sigma-point check
  largest nonlinearity index             : 0.05042 (threshold 0.1)
  verdict                                : linear results trusted

Monte Carlo (1000 samples, nonlinear guidance, 4 worker(s), 8.2 s)
  final position sigma (x, y, z) [km]   : 0.09315, 0.07105, 0.06976
  final velocity sigma (x, y, z) [m/s]  : 3.596, 4.75, 5.077
  mean total delta-V [m/s]               : 1760.08
  delta-V 99 [m/s]                      : 1776.89 (95% interval 1776.09 to 1779.03)
  failed / unconverged samples           : 0 / 0

Linear covariance against Monte Carlo
  sigma ratio over all directions        : 0.941 to 1.068
  linear delta-V 99                      : inside the Monte Carlo interval (difference 0.05%)
  verdict                                : engines agree
```

## 4. Read it

- **Delta-V budget.** The trajectory needs 1751.1 m/s nominally. With the
  errors, 99% of missions need at most 1777.7 m/s: the margin to carry is
  about 27 m/s, 1.5%.
- **Delivery accuracy.** The position at the end is known to about 90 m
  (1-sigma per axis). The velocity dispersion is larger, about 4 to 5 m/s,
  because the circularisation burn at the end has its own execution error,
  which nothing corrects; that is also why the final semi-major axis
  dispersion is 13.7 km.
- **Can the linear answer be trusted?** Yes, twice over: the sigma-point
  check finds no nonlinear arc, and Monte Carlo, which re-targets every
  correction with the full dynamics, agrees.
- **Is 1000 samples enough?** `gmat-uq wilks` prints 299: the smallest sample
  that bounds the 99th percentile with 95% confidence. Monte Carlo reports an
  interval, and the linear value falls inside it.

The full results, including the dispersion at every maneuver, are in
`two_burn_transfer_report.json`.

## 5. Change one thing

Make the navigation four times worse (200 m, 20 cm/s) and run only the fast
engines:

```toml
[navigation.Tracking]
sigmas = [0.2, 0.2, 0.2, 2e-4, 2e-4, 2e-4]
```

```
gmat-uq run two_burn_transfer.toml --methods lincov sigma_point
```

```
  final position sigma (x, y, z) [km]   : 0.3601, 0.2865, 0.2792
  delta-V 99 [m/s]                      : 1778.89
```

The delivery error grows about four times, because after the last correction
it is set by the knowledge error. The delta-V 99 barely moves (1.2 m/s):
the corrections are driven by the injection and burn errors, not by the
navigation. This is the kind of trade the linear engine answers in a second.

## 6. The same from Python

[`examples/python_api.py`](../../examples/python_api.py) builds a similar
mission in code and runs the three engines:

```
python examples/python_api.py
```

Next: [Tutorial 2](02-your-gmat-script.md) analyses a mission defined in a
GMAT script.
