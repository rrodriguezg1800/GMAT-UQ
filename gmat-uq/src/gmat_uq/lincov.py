# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Linear covariance engine.

The engine propagates ``D``, the covariance of the true deviation from the
nominal trajectory, through coast arcs, open-loop burns and closed-loop
correction maneuvers:

* coast from node ``i`` to ``i + 1``::

      D_before[i + 1] = Phi_i  D_after[i]  Phi_i^T + Q_i

* open-loop burn with execution-error covariance ``P_exec``::

      D_after = D_before + B P_exec B^T

* closed-loop correction designed from an estimate with knowledge covariance
  ``P_nav``, using the linear guidance gain ``K``::

      S       = K (D_before + P_nav) K^T                       (commanded correction)
      D_after = (I - B K) D_before (I - B K)^T + B K P_nav K^T B^T + B P_exec B^T

The knowledge error is taken as independent of the true dispersion.

Consider parameters (uncertain model parameters such as ``Cd``) augment the
state: ``Phi`` becomes ``[[Phi, Psi], [0, I]]``, where ``Psi`` is the
sensitivity of the state to the parameters over the interval.  The guidance
gain acts on the orbit state only, because the parameters are not estimated.

Delta-V statistics are not Gaussian (they are magnitudes), so they are obtained
by sampling the same linear system, which costs matrix products only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics.base import Dynamics
from .linalg import covariance_factor, symmetrize, velocity_input_matrix
from .mission import Mission, Nominal, compute_nominal
from .sampling import draw_normals_batch
from .stats import error_ellipse, summarize_magnitudes

__all__ = ["LinCovResult", "LinearSamples", "ManeuverCovariance", "run_lincov", "sample_linear"]

_HISTORY_LIMIT = 2_000_000  # samples times nodes


@dataclass(frozen=True)
class ManeuverCovariance:
    """Linear statistics of one maneuver."""

    name: str
    time: float
    node: int
    closed_loop: bool
    dv_nominal: np.ndarray  # (3,) km/s
    correction_covariance: np.ndarray  # S, covariance of the commanded delta-V about its nominal
    execution_covariance: np.ndarray  # expected execution-error covariance


@dataclass(frozen=True)
class LinearSamples:
    """Samples of the linear system (deviations from the nominal)."""

    indices: np.ndarray  # (N,) sample indices
    deviations_before: np.ndarray | None  # (N, n_nodes, 6) deviation arriving at each node, if kept
    final_deviation: np.ndarray  # (N, 6) deviation leaving the last node
    commanded_dv: np.ndarray  # (N, n_maneuvers, 3) commanded delta-V vectors, km/s
    execution_error: np.ndarray  # (N, n_maneuvers, 3)

    @property
    def dv_magnitudes(self) -> np.ndarray:
        """Commanded delta-V magnitude of every maneuver, (N, n_maneuvers), km/s."""
        return np.linalg.norm(self.commanded_dv, axis=2)

    @property
    def total_dv(self) -> np.ndarray:
        """Sum of the commanded delta-V magnitudes of every sample, (N,), km/s."""
        return self.dv_magnitudes.sum(axis=1)


@dataclass(frozen=True)
class LinCovResult:
    """Output of :func:`run_lincov`."""

    mission: Mission
    nominal: Nominal
    covariance_before: np.ndarray  # (n_nodes, 6, 6) dispersion arriving at each node
    covariance_after: np.ndarray  # (n_nodes, 6, 6) dispersion leaving each node
    maneuvers: tuple[ManeuverCovariance, ...]
    samples: LinearSamples | None
    seed: int
    # (n_nodes, 6 + m, 6 + m): covariance of the state augmented with the consider parameters
    augmented_after: np.ndarray | None = None

    @property
    def times(self) -> np.ndarray:
        return self.nominal.times

    @property
    def final_covariance(self) -> np.ndarray:
        return self.covariance_after[-1]

    @property
    def target_covariance(self) -> np.ndarray | None:
        """Dispersion at the final time mapped into the mission's final target, if any."""
        jacobian = self.nominal.final_target_jacobian
        if jacobian is None:
            return None
        return symmetrize(jacobian @ self.final_covariance @ jacobian.T)

    def sigmas(self, node: int = -1, after: bool = True) -> np.ndarray:
        """One-sigma dispersion of the six state components at a node."""
        covariance = self.covariance_after[node] if after else self.covariance_before[node]
        return np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    def position_sigma_rss(self, node: int = -1, after: bool = True) -> float:
        return float(np.sqrt(np.sum(self.sigmas(node, after)[:3] ** 2)))

    def velocity_sigma_rss(self, node: int = -1, after: bool = True) -> float:
        return float(np.sqrt(np.sum(self.sigmas(node, after)[3:] ** 2)))

    def dv_statistics(self, probability: float = 0.99, confidence: float = 0.95) -> dict | None:
        """Delta-V statistics from the linear samples (km/s), or ``None`` without samples or maneuvers."""
        if self.samples is None or not self.nominal.plans:
            return None
        magnitudes = self.samples.dv_magnitudes
        per_maneuver = {
            plan.maneuver.name: summarize_magnitudes(magnitudes[:, index], probability, confidence)
            for index, plan in enumerate(self.nominal.plans)
        }
        return {
            "total": summarize_magnitudes(self.samples.total_dv, probability, confidence),
            "per_maneuver": per_maneuver,
            "nominal_total": float(sum(np.linalg.norm(p.maneuver.dv_vector) for p in self.nominal.plans)),
        }

    def target_ellipse(self, n_sigma: float = 1.0):
        """Dispersion ellipse in the final target, when that target has two components."""
        covariance = self.target_covariance
        if covariance is None or covariance.shape != (2, 2):
            return None
        return error_ellipse(covariance, n_sigma)


def _initial_covariance(mission: Mission) -> np.ndarray:
    if mission.initial_dispersion is None:
        return np.zeros((6, 6))
    return np.array(mission.initial_dispersion.covariance)


def run_lincov(
    mission: Mission,
    dynamics: Dynamics,
    nominal: Nominal | None = None,
    dv_samples: int = 20000,
    seed: int = 0,
) -> LinCovResult:
    """Run the linear covariance analysis of a mission.

    Parameters
    ----------
    mission, dynamics:
        What to analyse and how to propagate it.
    nominal:
        A nominal trajectory computed earlier with :func:`compute_nominal`, to
        avoid propagating it again.
    dv_samples:
        Number of samples of the linear system used for delta-V statistics.
        Zero skips them.
    seed:
        Master seed of those samples.
    """
    if nominal is None:
        nominal = compute_nominal(mission, dynamics)
    n = nominal.n_nodes
    m = nominal.n_consider
    size = 6 + m
    input_matrix = np.zeros((size, 3))
    input_matrix[3:6, :] = np.eye(3)
    identity = np.eye(size)

    before = np.empty((n, 6, 6))
    after = np.empty((n, 6, 6))
    augmented = np.empty((n, size, size))
    maneuvers = []
    covariance = np.zeros((size, size))
    covariance[:6, :6] = _initial_covariance(mission)
    if m:
        covariance[6:, 6:] = np.diag(mission.consider_sigmas**2)
    for node in range(n):
        before[node] = covariance[:6, :6]
        plan = nominal.plan_at(node)
        if plan is not None:
            maneuver = plan.maneuver
            nominal_dv = maneuver.dv_vector
            if maneuver.closed_loop:
                gain = np.zeros((3, size))
                gain[:, :6] = plan.gain  # the parameters are not estimated, so the gain ignores them
                knowledge = np.zeros((size, size))
                if maneuver.navigation is not None:
                    knowledge[:6, :6] = maneuver.navigation.covariance
                correction = symmetrize(gain @ (covariance + knowledge) @ gain.T)
                closed = identity - input_matrix @ gain
                covariance = closed @ covariance @ closed.T + input_matrix @ gain @ knowledge @ gain.T @ input_matrix.T
            else:
                correction = np.zeros((3, 3))
            if maneuver.error_model is not None and not maneuver.error_model.is_zero:
                execution = maneuver.error_model.expected_covariance(nominal_dv, correction)
            else:
                execution = np.zeros((3, 3))
            covariance = symmetrize(covariance + input_matrix @ execution @ input_matrix.T)
            maneuvers.append(
                ManeuverCovariance(maneuver.name, maneuver.time, node, maneuver.closed_loop, nominal_dv, correction, execution)
            )
        after[node] = covariance[:6, :6]
        augmented[node] = covariance
        if node < n - 1:
            transition = nominal.augmented_transition(node)
            covariance = transition @ covariance @ transition.T
            if nominal.process_noise:
                covariance[:6, :6] = covariance[:6, :6] + nominal.process_noise[node]
            covariance = symmetrize(covariance)

    samples = None
    if dv_samples > 0:
        # The per-node history of every sample is kept only while it stays small (about 100 MB).
        keep_history = dv_samples * n <= _HISTORY_LIMIT
        samples = sample_linear(mission, nominal, np.arange(dv_samples), seed, keep_history)
    return LinCovResult(mission, nominal, before, after, tuple(maneuvers), samples, seed, augmented)


def sample_linear(mission: Mission, nominal: Nominal, indices, seed: int = 0, keep_history: bool = True) -> LinearSamples:
    """Sample the linear system for the given sample indices.

    The draws are the ones the Monte Carlo engine uses for the same indices and
    seed, so the two engines can be compared sample by sample.  With
    ``keep_history=False`` the deviations at intermediate nodes are not stored
    (``deviations_before`` is ``None``), which saves memory on long node grids.
    """
    indices = np.asarray(indices, dtype=np.int64).ravel()
    count = indices.size
    n = nominal.n_nodes
    plans = nominal.plans
    intervals = n - 1 if nominal.process_noise else 0
    normals = draw_normals_batch(seed, indices, len(plans), intervals, nominal.n_consider)

    deviation = np.zeros((count, 6))
    if mission.initial_dispersion is not None:
        deviation = mission.initial_dispersion.apply(normals.initial)
    parameter_error = normals.consider * mission.consider_sigmas if nominal.n_consider else None
    noise_factors = [covariance_factor(q) for q in nominal.process_noise]

    deviations_before = np.empty((count, n, 6)) if keep_history else None
    commanded = np.zeros((count, len(plans), 3))
    execution = np.zeros((count, len(plans), 3))
    plan_index = {plan.node: index for index, plan in enumerate(plans)}
    for node in range(n):
        if keep_history:
            deviations_before[:, node] = deviation
        if node in plan_index:
            index = plan_index[node]
            plan = plans[index]
            maneuver = plan.maneuver
            command = np.tile(maneuver.dv_vector, (count, 1))
            if maneuver.closed_loop:
                estimate = deviation
                if maneuver.navigation is not None:
                    estimate = deviation + maneuver.navigation.apply(normals.navigation[:, index])
                command = command - estimate @ plan.gain.T
            if maneuver.error_model is not None and not maneuver.error_model.is_zero:
                execution[:, index] = maneuver.error_model.apply(command, normals.execution[:, index])
            commanded[:, index] = command
            deviation = deviation.copy()
            deviation[:, 3:] += command - maneuver.dv_vector + execution[:, index]
        if node < n - 1:
            propagated = deviation @ nominal.stms[node].T
            if parameter_error is not None:
                propagated = propagated + parameter_error @ nominal.consider_sensitivities[node].T
            deviation = propagated
            if noise_factors:
                deviation = deviation + normals.process[:, node] @ noise_factors[node].T
    return LinearSamples(indices, deviations_before, deviation, commanded, execution)
