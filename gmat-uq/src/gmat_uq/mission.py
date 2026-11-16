# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Mission description and nominal trajectory.

A mission is an initial state, a list of impulsive maneuvers at fixed times and
a final time.  The nominal trajectory is computed once and shared by the linear
covariance, sigma-point and Monte Carlo engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics.base import Dynamics
from .linalg import velocity_input_matrix
from .models import ManeuverErrorModel, NavigationErrorModel, ProcessNoise, StateDispersion
from .targets import Target

__all__ = ["Maneuver", "Mission", "Nominal", "ManeuverPlan", "compute_nominal"]

_TIME_TOLERANCE = 1e-9  # seconds; times closer than this are the same node


@dataclass(frozen=True, eq=False)
class Maneuver:
    """Impulsive maneuver at a fixed time.

    Parameters
    ----------
    name:
        Unique label used in reports.
    time:
        Seconds from the mission epoch.
    dv:
        Nominal commanded delta-V in the inertial frame, km/s.  Use zeros for a
        purely statistical correction.
    error_model:
        Execution-error model, or ``None`` for a perfect burn.
    target:
        If given, the maneuver is a closed-loop correction: its delta-V is
        recomputed from the estimated state so that ``target`` keeps its nominal
        value at ``target_time``.
    target_time:
        Seconds from the mission epoch at which ``target`` is evaluated.  It
        must be later than ``time`` and no later than the mission's final time.
    navigation:
        Knowledge error of the state estimate used to design the correction, or
        ``None`` for perfect knowledge.  Ignored for open-loop maneuvers.
    """

    name: str
    time: float
    dv: tuple[float, float, float] = (0.0, 0.0, 0.0)
    error_model: ManeuverErrorModel | None = None
    target: Target | None = None
    target_time: float | None = None
    navigation: NavigationErrorModel | None = None

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ValueError("a maneuver needs a name")
        dv = np.asarray(self.dv, dtype=float).reshape(-1)
        if dv.shape != (3,) or not np.all(np.isfinite(dv)):
            raise ValueError(f"maneuver '{self.name}': dv must hold three finite values")
        object.__setattr__(self, "dv", tuple(float(v) for v in dv))
        object.__setattr__(self, "time", float(self.time))
        if (self.target is None) != (self.target_time is None):
            raise ValueError(f"maneuver '{self.name}': give both target and target_time, or neither")
        if self.target is not None:
            object.__setattr__(self, "target_time", float(self.target_time))
            if self.target_time <= self.time + _TIME_TOLERANCE:
                raise ValueError(f"maneuver '{self.name}': target_time must be later than the maneuver")
            if not 1 <= self.target.size <= 3:
                raise ValueError(f"maneuver '{self.name}': a correction can control one to three target components")

    @property
    def closed_loop(self) -> bool:
        return self.target is not None

    @property
    def dv_vector(self) -> np.ndarray:
        return np.array(self.dv)


@dataclass(frozen=True, eq=False)
class Mission:
    """Initial state, maneuver schedule, uncertainty sources and reporting options.

    Parameters
    ----------
    initial_state:
        Nominal state at ``initial_time``: ``[x, y, z, vx, vy, vz]`` in km and km/s.
    final_time:
        End of the analysis, seconds from the mission epoch.
    maneuvers:
        Impulsive maneuvers, in any order.
    initial_dispersion:
        Dispersion of the true initial state, or ``None``.
    process_noise:
        Unmodelled-acceleration process noise, or ``None``.
    report_times:
        Extra times at which the dispersion covariance is reported.
    final_target:
        Optional target in which the final dispersion is also reported (for
        example the B-plane).
    initial_time:
        Start of the analysis, seconds from the mission epoch.
    """

    initial_state: tuple[float, ...]
    final_time: float
    maneuvers: tuple[Maneuver, ...] = ()
    initial_dispersion: StateDispersion | None = None
    process_noise: ProcessNoise | None = None
    report_times: tuple[float, ...] = ()
    final_target: Target | None = None
    initial_time: float = 0.0

    def __post_init__(self) -> None:
        state = np.asarray(self.initial_state, dtype=float).reshape(-1)
        if state.shape != (6,) or not np.all(np.isfinite(state)):
            raise ValueError("initial_state must hold six finite values")
        object.__setattr__(self, "initial_state", tuple(float(v) for v in state))
        object.__setattr__(self, "initial_time", float(self.initial_time))
        object.__setattr__(self, "final_time", float(self.final_time))
        if self.final_time <= self.initial_time:
            raise ValueError("final_time must be later than initial_time")
        maneuvers = tuple(sorted(self.maneuvers, key=lambda m: m.time))
        object.__setattr__(self, "maneuvers", maneuvers)
        names = [m.name for m in maneuvers]
        if len(set(names)) != len(names):
            raise ValueError("maneuver names must be unique")
        for previous, current in zip(maneuvers, maneuvers[1:]):
            if current.time - previous.time <= _TIME_TOLERANCE:
                raise ValueError(f"maneuvers '{previous.name}' and '{current.name}' share the same time")
        for maneuver in maneuvers:
            if not self.initial_time - _TIME_TOLERANCE <= maneuver.time <= self.final_time + _TIME_TOLERANCE:
                raise ValueError(f"maneuver '{maneuver.name}' lies outside the mission time span")
            if maneuver.closed_loop and maneuver.target_time > self.final_time + _TIME_TOLERANCE:
                raise ValueError(f"maneuver '{maneuver.name}': target_time is later than final_time")
        reports = tuple(sorted(float(t) for t in self.report_times))
        for time in reports:
            if not self.initial_time <= time <= self.final_time:
                raise ValueError("report_times must lie inside the mission time span")
        object.__setattr__(self, "report_times", reports)

    @property
    def state_vector(self) -> np.ndarray:
        return np.array(self.initial_state)

    def node_times(self) -> np.ndarray:
        """Sorted, de-duplicated times at which the engines stop.

        Nodes are the initial and final times, every maneuver and target time,
        every report time and, when process noise is active, enough extra
        points to keep every interval shorter than ``process_noise.max_step``.
        """
        times = [self.initial_time, self.final_time, *self.report_times]
        for maneuver in self.maneuvers:
            times.append(maneuver.time)
            if maneuver.closed_loop:
                times.append(maneuver.target_time)
        ordered = sorted(min(max(t, self.initial_time), self.final_time) for t in times)
        nodes = [ordered[0]]
        for time in ordered[1:]:
            if time - nodes[-1] > _TIME_TOLERANCE:
                nodes.append(time)
        if self.process_noise is not None:
            refined = [nodes[0]]
            for time in nodes[1:]:
                span = time - refined[-1]
                pieces = int(np.ceil(span / self.process_noise.max_step - 1e-12))
                start = refined[-1]
                for index in range(1, pieces):
                    refined.append(start + span * index / pieces)
                refined.append(time)
            nodes = refined
        return np.array(nodes)


@dataclass(frozen=True)
class ManeuverPlan:
    """A maneuver placed on the node grid, with its linear guidance law."""

    maneuver: Maneuver
    node: int
    target_node: int | None = None
    target_goal: np.ndarray | None = None  # nominal target value at the target node
    sensitivity: np.ndarray | None = None  # M = d(target)/d(state at the maneuver), size x 6
    gain: np.ndarray | None = None  # K, 3 x 6: commanded correction = -K (estimated deviation)


@dataclass(frozen=True)
class Nominal:
    """Nominal trajectory on the node grid."""

    times: np.ndarray  # (n,)
    states_before: np.ndarray  # (n, 6) state arriving at each node, before any maneuver there
    states_after: np.ndarray  # (n, 6) state leaving each node, after any maneuver there
    stms: np.ndarray  # (n - 1, 6, 6) transition matrix from node i to node i + 1
    plans: tuple[ManeuverPlan, ...]
    final_target_value: np.ndarray | None = None
    final_target_jacobian: np.ndarray | None = None
    process_noise: tuple[np.ndarray, ...] = field(default=())  # per-interval Q, empty when inactive

    @property
    def n_nodes(self) -> int:
        return len(self.times)

    def plan_at(self, node: int) -> ManeuverPlan | None:
        for plan in self.plans:
            if plan.node == node:
                return plan
        return None

    def stm_between(self, start: int, end: int) -> np.ndarray:
        """Transition matrix from the state leaving node ``start`` to the state arriving at node ``end``."""
        matrix = np.eye(6)
        for index in range(start, end):
            matrix = self.stms[index] @ matrix
        return matrix


def _node_index(times: np.ndarray, time: float) -> int:
    index = int(np.argmin(np.abs(times - time)))
    if abs(times[index] - time) > 10.0 * _TIME_TOLERANCE:
        raise RuntimeError(f"time {time} is not on the node grid")  # pragma: no cover - internal consistency
    return index


def compute_nominal(mission: Mission, dynamics: Dynamics) -> Nominal:
    """Propagate the nominal trajectory and build the linear guidance law of every correction."""
    times = mission.node_times()
    n = len(times)
    maneuver_at = {_node_index(times, m.time): m for m in mission.maneuvers}

    before = np.empty((n, 6))
    after = np.empty((n, 6))
    stms = np.empty((n - 1, 6, 6))
    state = mission.state_vector
    for node in range(n):
        before[node] = state
        if node in maneuver_at:
            state = state.copy()
            state[3:] += maneuver_at[node].dv_vector
        after[node] = state
        if node < n - 1:
            state, stms[node] = dynamics.propagate_stm(after[node], times[node], times[node + 1])
            state = np.asarray(state, dtype=float)

    partial = Nominal(times, before, after, stms, ())
    input_matrix = velocity_input_matrix()
    plans = []
    for node, maneuver in sorted(maneuver_at.items()):
        if not maneuver.closed_loop:
            plans.append(ManeuverPlan(maneuver, node))
            continue
        target_node = _node_index(times, maneuver.target_time)
        goal = maneuver.target(before[target_node])
        sensitivity = maneuver.target.jacobian(before[target_node]) @ partial.stm_between(node, target_node)
        control = sensitivity @ input_matrix  # effect of a velocity change at the maneuver on the target
        rank = np.linalg.matrix_rank(control, tol=1e-10 * max(np.linalg.norm(control), np.finfo(float).tiny))
        if rank < maneuver.target.size:
            raise ValueError(
                f"maneuver '{maneuver.name}' cannot control its target: the sensitivity of "
                f"{list(maneuver.target.labels)} to the delta-V has rank {rank}"
            )
        gain = np.linalg.pinv(control) @ sensitivity
        plans.append(ManeuverPlan(maneuver, node, target_node, goal, sensitivity, gain))

    final_value = final_jacobian = None
    if mission.final_target is not None:
        final_value = mission.final_target(after[-1])
        final_jacobian = mission.final_target.jacobian(after[-1])

    noise = ()
    if mission.process_noise is not None:
        noise = tuple(mission.process_noise.covariance(times[i + 1] - times[i]) for i in range(n - 1))

    return Nominal(times, before, after, stms, tuple(plans), final_value, final_jacobian, noise)
