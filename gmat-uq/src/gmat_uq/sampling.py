# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Reproducible random streams shared by the linear sampler and the Monte Carlo engine.

The draws of a sample depend on the master seed and the sample index only.
A sample therefore gets the same numbers no matter how many worker processes
run, in which order, in which batch, or which engine consumes them.  Because
the linear sampler and the Monte Carlo engine read the same draws, their
results can be compared sample by sample.

The streams come from Philox, a counter-based generator: the block of numbers
that belongs to sample ``i`` is found by advancing the counter to ``i`` times
the block size, without generating anything before it.  Each source of
uncertainty (initial state, navigation, execution, process noise) has its own
key, so switching one source on or off leaves the draws of the others
unchanged.  Uniform numbers are mapped to normal ones with the inverse normal
distribution function, which uses exactly one uniform per normal and so keeps
the blocks aligned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

__all__ = ["SampleNormals", "draw_normals", "draw_normals_batch"]

_INITIAL, _NAVIGATION, _EXECUTION, _PROCESS = 0, 1, 2, 3
_WORDS_PER_STEP = 4  # one step of the Philox counter yields four 64-bit numbers
_SMALLEST_UNIFORM = 2.0**-53  # keeps the inverse distribution function finite


@dataclass(frozen=True)
class SampleNormals:
    """Standard-normal draws of one or more samples.

    The leading axis is the sample axis when the object holds a batch.
    """

    initial: np.ndarray  # (..., 6) initial-state dispersion
    navigation: np.ndarray  # (..., n_maneuvers, 6) knowledge error at each maneuver
    execution: np.ndarray  # (..., n_maneuvers, 3) execution error of each maneuver
    process: np.ndarray  # (..., n_intervals, 6) process noise of each interval (may be empty)


def _check_seed(seed) -> int:
    seed = int(seed)
    if not 0 <= seed < 2**64:
        raise ValueError("the seed must be an integer between 0 and 2**64 - 1")
    return seed


def _normal_block(seed: int, source: int, first_index: int, count: int, width: int) -> np.ndarray:
    """Standard normals of ``count`` consecutive samples; row ``j`` belongs to sample ``first_index + j``."""
    if width == 0 or count == 0:
        return np.empty((count, width))
    steps = -(-width // _WORDS_PER_STEP)  # counter steps per sample, rounded up
    generator = np.random.Philox(key=np.array([seed, source], dtype=np.uint64))
    generator.advance(first_index * steps)
    uniform = np.random.Generator(generator).random((count, steps * _WORDS_PER_STEP))[:, :width]
    return ndtri(np.maximum(uniform, _SMALLEST_UNIFORM))


def _runs(indices: np.ndarray):
    """Split sample indices into runs of consecutive values: (position, first index, length)."""
    if indices.size == 0:
        return
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [indices.size]])
    for start, end in zip(starts, ends):
        yield int(start), int(indices[start]), int(end - start)


def draw_normals_batch(seed: int, indices, n_maneuvers: int, n_process_intervals: int = 0) -> SampleNormals:
    """Draw the standard normals of several samples.

    Row ``j`` of every array belongs to sample ``indices[j]`` and has the same
    values as ``draw_normals(seed, indices[j], ...)``.
    """
    seed = _check_seed(seed)
    indices = np.asarray(indices, dtype=np.int64).ravel()
    if indices.size and indices.min() < 0:
        raise ValueError("sample index must be non-negative")
    count = indices.size
    widths = {_INITIAL: 6, _NAVIGATION: 6 * n_maneuvers, _EXECUTION: 3 * n_maneuvers, _PROCESS: 6 * n_process_intervals}
    blocks = {source: np.empty((count, width)) for source, width in widths.items()}
    for position, first_index, length in _runs(indices):
        for source, width in widths.items():
            blocks[source][position : position + length] = _normal_block(seed, source, first_index, length, width)
    return SampleNormals(
        initial=blocks[_INITIAL],
        navigation=blocks[_NAVIGATION].reshape(count, n_maneuvers, 6),
        execution=blocks[_EXECUTION].reshape(count, n_maneuvers, 3),
        process=blocks[_PROCESS].reshape(count, n_process_intervals, 6),
    )


def draw_normals(seed: int, index: int, n_maneuvers: int, n_process_intervals: int = 0) -> SampleNormals:
    """Draw the standard normals of sample ``index``."""
    if index < 0:
        raise ValueError("sample index must be non-negative")
    batch = draw_normals_batch(seed, [index], n_maneuvers, n_process_intervals)
    return SampleNormals(batch.initial[0], batch.navigation[0], batch.execution[0], batch.process[0])
