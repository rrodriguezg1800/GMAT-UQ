# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Closed-loop Monte Carlo engine.

Every sample carries the true state through the full nonlinear dynamics.  At a
correction maneuver the flight software only knows an estimate (truth plus a
knowledge error), designs the burn from it, and the burn is then executed with
an execution error on the true state.

Samples are independent, so they are distributed over worker processes.  The
random stream of a sample depends only on the master seed and the sample
index, which makes the results identical for any number of workers.

Consider parameters (for example ``Cd``) take a true value per sample, drawn
once and held for the whole mission.  The truth is propagated with that value;
corrections are designed with the nominal value, as flight software would.
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .dynamics.base import Dynamics, PropagationError
from .guidance import linear_correction, solve_correction
from .linalg import covariance_factor
from .mission import Mission, Nominal, compute_nominal
from .sampling import draw_normals
from .stats import summarize_magnitudes

__all__ = ["MonteCarloResult", "run_monte_carlo", "simulate_sample"]

GUIDANCE_MODES = ("nonlinear", "linear")


@dataclass(frozen=True)
class _Context:
    mission: Mission
    nominal: Nominal
    dynamics: Dynamics
    seed: int
    guidance: str
    tolerance: float
    max_iterations: int
    noise_factors: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class MonteCarloResult:
    """Output of :func:`run_monte_carlo`.  Arrays are ordered by sample index."""

    mission: Mission
    nominal: Nominal
    seed: int
    guidance: str
    indices: np.ndarray  # (N,)
    deviations_before: np.ndarray  # (N, n_nodes, 6) true deviation arriving at each node
    final_deviation: np.ndarray  # (N, 6) true deviation leaving the last node
    commanded_dv: np.ndarray  # (N, n_maneuvers, 3) km/s
    execution_error: np.ndarray  # (N, n_maneuvers, 3) km/s
    converged: np.ndarray  # (N, n_maneuvers) bool; True for open-loop and linear-guidance maneuvers
    iterations: np.ndarray  # (N, n_maneuvers) trajectory evaluations used by the corrector
    failed: np.ndarray  # (N,) bool; the sample could not be propagated to the end
    target_deviation: np.ndarray | None  # (N, m) final target minus its nominal value
    workers: int
    elapsed_seconds: float
    consider_deviation: np.ndarray | None = None  # (N, n_consider) true minus nominal parameter values

    @property
    def n_samples(self) -> int:
        return int(self.indices.size)

    @property
    def valid(self) -> np.ndarray:
        """Samples that reached the final time."""
        return ~self.failed

    @property
    def n_failed(self) -> int:
        return int(self.failed.sum())

    @property
    def n_unconverged(self) -> int:
        """Samples in which at least one correction did not converge."""
        return int((~self.converged[self.valid]).any(axis=1).sum()) if self.converged.size else 0

    @property
    def dv_magnitudes(self) -> np.ndarray:
        return np.linalg.norm(self.commanded_dv[self.valid], axis=2)

    @property
    def total_dv(self) -> np.ndarray:
        return self.dv_magnitudes.sum(axis=1)

    @property
    def final_covariance(self) -> np.ndarray:
        """Sample covariance of the true deviation at the final time."""
        return np.cov(self.final_deviation[self.valid], rowvar=False, ddof=1)

    def covariance_before(self, node: int) -> np.ndarray:
        return np.cov(self.deviations_before[self.valid, node], rowvar=False, ddof=1)

    @property
    def target_covariance(self) -> np.ndarray | None:
        if self.target_deviation is None:
            return None
        data = self.target_deviation[self.valid]
        return np.atleast_2d(np.cov(data, rowvar=False, ddof=1))

    def dv_statistics(self, probability: float = 0.99, confidence: float = 0.95) -> dict:
        magnitudes = self.dv_magnitudes
        per_maneuver = {
            plan.maneuver.name: summarize_magnitudes(magnitudes[:, index], probability, confidence)
            for index, plan in enumerate(self.nominal.plans)
        }
        return {
            "total": summarize_magnitudes(self.total_dv, probability, confidence),
            "per_maneuver": per_maneuver,
            "nominal_total": float(sum(np.linalg.norm(p.maneuver.dv_vector) for p in self.nominal.plans)),
        }


def _noise_factors(nominal: Nominal) -> tuple[np.ndarray, ...]:
    return tuple(covariance_factor(q) for q in nominal.process_noise)


def simulate_sample(context: _Context, index: int) -> dict:
    """Run one closed-loop sample and return its raw outputs."""
    mission, nominal, dynamics = context.mission, context.nominal, context.dynamics
    plans = nominal.plans
    n = nominal.n_nodes
    intervals = n - 1 if context.noise_factors else 0
    normals = draw_normals(context.seed, index, len(plans), intervals, nominal.n_consider)
    plan_index = {plan.node: k for k, plan in enumerate(plans)}
    parameter_error = normals.consider * mission.consider_sigmas
    # The truth flies with the true parameters; corrections are designed with the nominal ones.
    truth = {"parameters": nominal.parameters_for(parameter_error)} if nominal.n_consider else {}

    deviations = np.full((n, 6), np.nan)
    commanded = np.full((len(plans), 3), np.nan)
    execution = np.zeros((len(plans), 3))
    converged = np.ones(len(plans), dtype=bool)
    iterations = np.zeros(len(plans), dtype=np.int64)
    final = np.full(6, np.nan)
    target_deviation = None
    if mission.final_target is not None:
        target_deviation = np.full(mission.final_target.size, np.nan)
    failed = False

    state = nominal.states_before[0].copy()
    if mission.initial_dispersion is not None:
        state = state + mission.initial_dispersion.apply(normals.initial)
    try:
        for node in range(n):
            deviations[node] = state - nominal.states_before[node]
            if node in plan_index:
                k = plan_index[node]
                plan = plans[k]
                maneuver = plan.maneuver
                if maneuver.closed_loop:
                    estimate = state
                    if maneuver.navigation is not None:
                        estimate = state + maneuver.navigation.apply(normals.navigation[k])
                    if context.guidance == "linear":
                        command = linear_correction(plan, nominal, estimate)
                    else:
                        solution = solve_correction(
                            dynamics, nominal, plan, estimate,
                            tolerance=context.tolerance, max_iterations=context.max_iterations,
                        )
                        command = solution.dv
                        converged[k] = solution.converged
                        iterations[k] = solution.iterations
                else:
                    command = maneuver.dv_vector
                if maneuver.error_model is not None and not maneuver.error_model.is_zero:
                    execution[k] = maneuver.error_model.apply(command, normals.execution[k])
                commanded[k] = command
                state = state.copy()
                state[3:] += command + execution[k]
            if node < n - 1:
                state = np.array(dynamics.propagate(state, nominal.times[node], nominal.times[node + 1], **truth), dtype=float)
                if context.noise_factors:
                    state = state + context.noise_factors[node] @ normals.process[node]
                if not np.all(np.isfinite(state)):
                    raise PropagationError("non-finite state")
        final = state - nominal.states_after[-1]
        if mission.final_target is not None:
            target_deviation = mission.final_target.residual(state, nominal.final_target_value)
    except (PropagationError, ValueError, FloatingPointError):
        failed = True
        final = np.full(6, np.nan)
        if target_deviation is not None:
            target_deviation = np.full(mission.final_target.size, np.nan)

    return {
        "index": index,
        "deviations": deviations,
        "final": final,
        "commanded": commanded,
        "execution": execution,
        "converged": converged,
        "iterations": iterations,
        "failed": failed,
        "target": target_deviation,
        "consider": parameter_error,
    }


# -- worker-process plumbing ---------------------------------------------------------------------
_worker_context: _Context | None = None


def _init_worker(context: _Context) -> None:
    global _worker_context
    _worker_context = context


def _run_chunk(indices) -> list[dict]:
    return [simulate_sample(_worker_context, int(index)) for index in indices]


def _chunks(indices: np.ndarray, workers: int, chunk_size: int | None) -> list[np.ndarray]:
    if chunk_size is None:
        chunk_size = max(1, int(np.ceil(indices.size / (workers * 4))))
    return [indices[start : start + chunk_size] for start in range(0, indices.size, chunk_size)]


def run_monte_carlo(
    mission: Mission,
    dynamics: Dynamics,
    n_samples: int,
    seed: int = 0,
    workers: int = 1,
    guidance: str = "nonlinear",
    nominal: Nominal | None = None,
    first_index: int = 0,
    tolerance: float = 1e-10,
    max_iterations: int = 30,
    chunk_size: int | None = None,
    progress=None,
) -> MonteCarloResult:
    """Run a closed-loop Monte Carlo analysis.

    Parameters
    ----------
    mission, dynamics:
        What to analyse and how to propagate it.
    n_samples:
        Number of samples; their indices are ``first_index ... first_index + n_samples - 1``.
    seed:
        Master seed.  The same seed and indices always give the same samples.
    workers:
        Number of worker processes.  With more than one, ``mission`` and
        ``dynamics`` must be picklable, and on Windows the calling script needs
        the usual ``if __name__ == "__main__":`` guard.
    guidance:
        ``"nonlinear"`` re-targets each correction with the full dynamics;
        ``"linear"`` applies the linear guidance gain (much cheaper, and exactly
        what the linear covariance engine assumes).
    nominal:
        A nominal trajectory computed earlier, to avoid propagating it again.
    tolerance, max_iterations:
        Controls of the nonlinear corrector (see :func:`gmat_uq.guidance.solve_correction`).
    chunk_size:
        Samples per task sent to a worker.
    progress:
        Optional callable ``progress(done, total)`` invoked as chunks finish.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if guidance not in GUIDANCE_MODES:
        raise ValueError(f"guidance must be one of {GUIDANCE_MODES}")
    if first_index < 0:
        raise ValueError("first_index must be non-negative")
    if nominal is None:
        nominal = compute_nominal(mission, dynamics)
    context = _Context(mission, nominal, dynamics, int(seed), guidance, float(tolerance), int(max_iterations), _noise_factors(nominal))
    indices = np.arange(first_index, first_index + n_samples, dtype=np.int64)
    workers = min(int(workers), n_samples)

    started = time.perf_counter()
    records: list[dict] = []
    if workers == 1:
        for chunk in _chunks(indices, 1, chunk_size or max(1, n_samples // 20)):
            records.extend(simulate_sample(context, int(index)) for index in chunk)
            if progress is not None:
                progress(len(records), n_samples)
    else:
        chunks = _chunks(indices, workers, chunk_size)
        spawn = multiprocessing.get_context("spawn")  # GMAT handles do not survive a fork
        with ProcessPoolExecutor(max_workers=workers, mp_context=spawn, initializer=_init_worker, initargs=(context,)) as pool:
            for chunk_records in pool.map(_run_chunk, chunks):
                records.extend(chunk_records)
                if progress is not None:
                    progress(len(records), n_samples)
    elapsed = time.perf_counter() - started

    records.sort(key=lambda record: record["index"])
    has_target = mission.final_target is not None
    return MonteCarloResult(
        mission=mission,
        nominal=nominal,
        seed=int(seed),
        guidance=guidance,
        indices=np.array([r["index"] for r in records], dtype=np.int64),
        deviations_before=np.array([r["deviations"] for r in records]),
        final_deviation=np.array([r["final"] for r in records]),
        commanded_dv=np.array([r["commanded"] for r in records]).reshape(len(records), len(nominal.plans), 3),
        execution_error=np.array([r["execution"] for r in records]).reshape(len(records), len(nominal.plans), 3),
        converged=np.array([r["converged"] for r in records], dtype=bool).reshape(len(records), len(nominal.plans)),
        iterations=np.array([r["iterations"] for r in records], dtype=np.int64).reshape(len(records), len(nominal.plans)),
        failed=np.array([r["failed"] for r in records], dtype=bool),
        target_deviation=np.array([r["target"] for r in records]) if has_target else None,
        workers=workers,
        elapsed_seconds=float(elapsed),
        consider_deviation=np.array([r["consider"] for r in records]).reshape(len(records), nominal.n_consider)
        if nominal.n_consider
        else None,
    )
