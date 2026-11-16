# Theory

This page states the models GMAT-UQ implements, with the assumptions behind each.
Units are kilometres, kilometres per second, seconds and radians, as in GMAT.

## The problem

A trajectory is designed for a nominal state history. In flight, three things
make the true state differ from it:

- the initial state is not the nominal one (for example launch injection errors);
- every maneuver is executed with an error;
- correction maneuvers are designed from an orbit estimate, not from the truth.

GMAT-UQ computes the statistics of the resulting deviation, and of the delta-V
spent correcting it.

## Mission model

A mission is an initial state, a final time and a list of impulsive maneuvers at
fixed times. The engines stop at *nodes*: the initial and final times, every
maneuver time, every target time and every requested report time.

A maneuver is either **open loop** or **closed loop**.

- An open-loop maneuver commands its nominal delta-V whatever the state is.
- A closed-loop maneuver (a *correction*) has a target, a vector function ψ of
  the state at a later, fixed target time. It commands the delta-V that returns
  ψ to its nominal value. A target has one to three components.

Both kinds can have a nominal delta-V of zero (a purely statistical correction)
or non-zero (a deterministic burn).

## Maneuver execution error: the Gates model

For a commanded delta-V **u** with magnitude m and unit direction **û**, the
execution error has a component along **û** and a component on each of the two
axes perpendicular to it:

```
sigma_along = sqrt(s1^2 + (s2 m)^2)        s1: fixed magnitude error      [km/s]
                                           s2: proportional magnitude error [-]
sigma_cross = sqrt(s3^2 + (s4 m)^2)        s3: fixed pointing error        [km/s per axis]
                                           s4: proportional pointing error [rad per axis]

P_exec(u) = sigma_along^2  û û^T  +  sigma_cross^2 (I - û û^T)
```

The pointing sigmas are **per axis**. Some references quote a total pointing
error over both axes; divide it by √2 before using it here.

A commanded delta-V of exactly zero has no direction. The fixed errors are then
applied isotropically with the same total variance, s1² + 2 s3².

In the linear covariance engine the command itself is random, **u** ~ N(**ū**, S).
The expected execution covariance is

```
E[P_exec] = s1^2 E[û û^T] + s2^2 (S + ū ū^T) + s3^2 (I - E[û û^T]) + s4^2 (tr(S + ū ū^T) I - (S + ū ū^T))
```

The proportional terms are exact. The direction dyad E[û û^T] has no closed
form and is evaluated with a deterministic quasi-Monte Carlo rule (8192 Sobol
points); it reduces to **û û**^T when S is zero.

## Navigation error

At a correction the flight software knows an estimate, x̂ = x + e, with
e ~ N(0, P_nav). GMAT-UQ takes P_nav as an input per maneuver. It can be given
in the inertial frame or in a local orbit frame (RIC or VNB) at the maneuver.

**Assumption.** e is independent of the true deviation. This is the usual
simplification when the orbit determination is *specified* rather than
simulated. A full treatment carries the cross-correlation in an augmented
state; that is future work.

## Linear covariance engine

Let D be the covariance of the true deviation from the nominal, Φᵢ the state
transition matrix from node i to node i+1 on the nominal, and B the 6×3 matrix
that maps a velocity change into the state.

**Coast.**

```
D_before[i+1] = Φ_i D_after[i] Φ_i^T + Q_i
```

Q is optional process noise (below).

**Open-loop maneuver.**

```
D_after = D_before + B P_exec(ū) B^T
```

**Closed-loop maneuver.** Let M be the sensitivity of the target to the state
leaving the maneuver, M = (∂ψ/∂x) Φ(target time, maneuver time), and
M_v = M B its part with respect to the delta-V. The linear guidance law is the
minimum-norm correction that nulls the predicted target deviation:

```
Δu = -K (δx + e),        K = pinv(M_v) M
```

so that

```
S       = K (D_before + P_nav) K^T                                  commanded correction
D_after = (I - B K) D_before (I - B K)^T + B K P_nav K^T B^T + B E[P_exec] B^T
```

Later maneuvers between the correction and its target are taken at their
nominal values when K is computed.

**Target-space dispersion.** For a final target with Jacobian J, the dispersion
is J D J^T. For a two-component target (the B-plane) it is reported as 1-sigma
and 3-sigma ellipses.

### Delta-V statistics

Delta-V is a magnitude, so its distribution is not Gaussian, and the
magnitudes of successive corrections are correlated. GMAT-UQ therefore samples
the *linear system itself*: each sample draws the initial error, the navigation
errors and the execution errors, and pushes them through the same matrices.
A sample costs a few matrix products and no propagation. The execution error of
a sample uses the Gates model on that sample's own commanded delta-V, not the
averaged covariance.

Delta-V 99 is the 99th percentile of the total commanded delta-V. It is
reported with a distribution-free confidence interval from order statistics:
for a sorted sample, the interval between ranks l and u covers the true
quantile with probability F(u−1) − F(l−1), where F is the Binomial(n, p)
distribution function. The smallest sample whose maximum bounds the p-quantile
with confidence β is the Wilks number, the smallest N with 1 − pᴺ ≥ β: 299 for
p = 0.99 and β = 0.95.

### Process noise

Optional white acceleration noise with spectral density σ² per axis adds, over
an interval Δt,

```
Q = σ^2 [ Δt^3/3 I   Δt^2/2 I ]
        [ Δt^2/2 I   Δt     I ]
```

This is the state-noise-compensation model of GMAT's `ProcessNoiseModel`, with
the sigmas given on the inertial axes. The formula is kinematic, so the node
grid is refined to keep every interval shorter than `max_step`.

## Sigma-point check

A linear covariance is only as good as the linearization. For every coast arc
the check places twelve sigma points at ±√6 times the columns of a factor of
D_after, propagates them with the full dynamics, and compares their mean and
covariance with the linear prediction, in coordinates whitened by the linear
covariance. Two numbers come out:

- the **mean shift**, in units of the linear 1-sigma dispersion;
- the **sigma deviation**, the largest relative difference between sigma-point
  and linear standard deviations over all directions.

The larger of the two is the arc's nonlinearity index. Both are exactly zero for
linear dynamics. An arc whose index exceeds the threshold (0.1 by default) is
flagged. The check looks at the dynamics of each arc, not at the guidance, and
it uses the linear covariance at the start of the arc; it is a warning light,
not a substitute for Monte Carlo.

## Closed-loop Monte Carlo engine

Each sample carries the true state through the full dynamics:

1. Draw the initial error and add it to the nominal initial state.
2. Propagate the truth to the next node.
3. At a correction, draw a knowledge error and build the estimate.
4. Design the burn from the estimate (see below).
5. Draw an execution error from the Gates model for that commanded burn and
   apply the imperfect burn to the truth.
6. Repeat to the final time; record deviations, commanded delta-V and the
   final target.

Step 4 has two modes. **Nonlinear guidance** (the default) solves for the
delta-V that restores the target with the full dynamics, by a differential
corrector started at the linear solution. It first reuses the partials of the
nominal trajectory (a chord method, which needs no state transition matrix per
sample) and switches to a full Newton iteration if a step fails to halve the
previous update. **Linear guidance** applies Δu = −K(δx + e) directly; it is
what the linear covariance engine assumes, and it is useful to separate
guidance nonlinearity from dynamics nonlinearity.

A sample that cannot be propagated is flagged as failed and excluded from the
statistics; a correction that does not converge falls back to the linear
solution and is counted.

### Reproducibility

The draws of a sample depend on the master seed and the sample index only, so
a sample has the same numbers whatever the number of worker processes, their
scheduling or the batch it runs in.

The streams come from Philox, a counter-based generator. The block of numbers
that belongs to sample i is reached by advancing the counter to i times the
block size, with no need to generate what comes before it. Each source of
uncertainty (initial state, navigation, execution, process noise) has its own
key, so switching one source on or off leaves the others unchanged. Uniform
numbers become normal ones through the inverse normal distribution function,
which uses exactly one uniform per normal and keeps the blocks aligned.

The linear sampler reads the same streams, so the two engines can be compared
sample by sample: in the linear regime they differ by second-order terms only.

## Consistency report

When both engines run, GMAT-UQ compares the Monte Carlo sample covariance with
the linear covariance in whitened coordinates. It reports the smallest and
largest ratio of sample to linear standard deviation over all directions, the
likelihood-ratio test of equal covariance (with Bartlett's correction) and the
per-component ratios with chi-square confidence intervals.

The engines are declared to **disagree on dispersions** only when the
difference is both large (a directional ratio further than 10% from one) and
statistically significant (test rejected at the 1% level). A small sample shows
large ratios by chance, and a very large sample detects differences too small
to matter; requiring both avoids the two false alarms. They **disagree on
delta-V** when the linear delta-V 99 is outside the Monte Carlo confidence
interval and further than 2% from the Monte Carlo estimate.

## Limits of the method

- Maneuvers are impulsive and at fixed times. Stopping conditions such as
  "at periapsis" are not modelled.
- Targets are evaluated at fixed times (fixed-time-of-arrival guidance).
- A target must be differentiable at the nominal. The eccentricity of a
  circular orbit, for example, is not.
- Statistics are Gaussian in the linear engine. Strongly nonlinear arcs need
  Monte Carlo; the sigma-point check says where.
- The navigation error is an input, not the output of a filter in the loop.
- Only the six orbit-state components are dispersed. Uncertain parameters such
  as drag or reflectivity coefficients are not yet supported.
