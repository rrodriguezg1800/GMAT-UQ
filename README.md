# GMAT + GMAT-UQ

This repository is a fork of [nasa/GMAT](https://github.com/nasa/GMAT), NASA's
General Mission Analysis Tool. It adds one thing: **GMAT-UQ**, a tool for
uncertainty analysis of GMAT missions, in the [`gmat-uq/`](gmat-uq/) folder.

GMAT-UQ answers two questions that GMAT does not answer on its own:

- **How much delta-V margin does the trajectory need?** Delta-V 99, with a
  confidence interval, given injection errors, maneuver execution errors and
  navigation errors.
- **How accurately is the spacecraft delivered?** Dispersion covariance along
  the trajectory and dispersion ellipses at the target, for example in the
  lunar B-plane.

It does so with a linear covariance engine, a sigma-point check that says when
the linear answer can be trusted, and a closed-loop Monte Carlo engine that
runs in parallel. GMAT itself does the propagation, through its Python API.

Start with [`gmat-uq/README.md`](gmat-uq/README.md). The validation results are
in [`gmat-uq/docs/validation.md`](gmat-uq/docs/validation.md).

## What is and is not changed

- Everything outside `gmat-uq/`, this file and `.github/workflows/gmat-uq.yml`
  is the upstream GMAT source, unmodified.
- GMAT-UQ is a separate Python package. It is not compiled into GMAT and needs
  no changes to a GMAT installation.

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
