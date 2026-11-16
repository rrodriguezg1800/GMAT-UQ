# GMAT + GMAT-UQ

This repository is a fork of [nasa/GMAT](https://github.com/nasa/GMAT), NASA's
General Mission Analysis Tool. It adds **GMAT-UQ**, uncertainty analysis for
GMAT missions:

- [`gmat-uq/`](gmat-uq/): the GMAT-UQ engines, a Python package that drives
  GMAT through its API.
- [`plugins/GmatUQPlugin/`](plugins/GmatUQPlugin/): a GMAT plugin that adds
  the analysis to GMAT's script language (`RunUncertaintyAnalysis`,
  `UQ.DeltaV99`, ...).

GMAT-UQ answers two questions that GMAT does not answer on its own:

- **How much delta-V margin does the trajectory need?** Delta-V 99, with a
  confidence interval, given injection errors, maneuver execution errors,
  navigation errors and uncertain model parameters such as Cd.
- **How accurately is the spacecraft delivered?** Dispersion covariance along
  the trajectory and dispersion ellipses at the target, for example in the
  lunar B-plane.

It does so with a linear covariance engine, a sigma-point check that says when
the linear answer can be trusted, and a closed-loop Monte Carlo engine that
runs in parallel. GMAT itself does the propagation.

Start with [`gmat-uq/README.md`](gmat-uq/README.md). The validation results are
in [`gmat-uq/docs/validation.md`](gmat-uq/docs/validation.md).

## What is and is not changed

- New: `gmat-uq/`, `plugins/GmatUQPlugin/`, this file, and the workflows
  `.github/workflows/gmat-uq.yml` and `.github/workflows/gmat-uq-plugin.yml`.
- Changed: `plugins/CMakeLists.txt` gains the option `PLUGIN_GMATUQ`, off by
  default, to build the plugin with GMAT.
- Everything else is the upstream GMAT source, unmodified.

## Status

This is an independent, unofficial project. It is not part of GMAT, and it has
not been reviewed or endorsed by NASA or by the GMAT development team.

## GMAT

For GMAT itself, see [`README.txt`](README.txt) and
[`application/README.txt`](application/README.txt), or the upstream project:

- https://github.com/nasa/GMAT (secondary repository)
- https://sourceforge.net/projects/gmat/ (primary repository and releases)

GMAT and GMAT-UQ are both licensed under the Apache License, Version 2.0; see
[`License.txt`](License.txt).
