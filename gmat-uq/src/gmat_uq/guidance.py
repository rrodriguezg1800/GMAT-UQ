# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Nonlinear targeting of impulsive maneuvers (differential corrector).

Given a state at the maneuver, the corrector finds the delta-V that makes a
target reach a goal value at a later time, propagating with the full dynamics.
It is used in two places:

* :func:`solve_correction` re-targets a correction maneuver inside the Monte
  Carlo engine, from the estimated state, so that the target returns to its
  nominal value.  Later maneuvers on the way to the target are assumed to be
  executed at their nominal values, exactly as in the linear guidance law.
* :func:`design_maneuver` is the same iteration as a stand-alone tool, handy
  for building a nominal trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics.base import Dynamics, PropagationError
from .mission import ManeuverPlan, Nominal
from .targets import Target

__all__ = ["CorrectionSolution", "solve_correction", "linear_correction", "design_maneuver"]

_MAX_DV = 1.0e3  # km/s; a commanded delta-V beyond this means the iteration has diverged


@dataclass(frozen=True)
class CorrectionSolution:
    dv: np.ndarray  # commanded delta-V, km/s
    converged: bool
    iterations: int  # number of trajectory evaluations


def linear_correction(plan: ManeuverPlan, nominal: Nominal, estimate) -> np.ndarray:
    """Commanded delta-V from the linear guidance law: nominal minus gain times estimated deviation."""
    deviation = np.asarray(estimate, dtype=float) - nominal.states_before[plan.node]
    return plan.maneuver.dv_vector - plan.gain @ deviation


def _legs(plan: ManeuverPlan, nominal: Nominal) -> list[tuple[float, float, np.ndarray | None]]:
    """Coast legs from the maneuver to its target, split at intermediate maneuvers.

    Each entry is ``(start_time, end_time, dv_applied_at_end)``.
    """
    stops = [plan.node]
    burns: dict[int, np.ndarray] = {}
    for other in nominal.plans:
        if plan.node < other.node < plan.target_node:
            stops.append(other.node)
            burns[other.node] = other.maneuver.dv_vector
    stops.append(plan.target_node)
    times = nominal.times
    return [(times[a], times[b], burns.get(b)) for a, b in zip(stops, stops[1:])]


def _newton(
    dynamics: Dynamics,
    state_before,
    legs,
    target: Target,
    goal,
    guess: np.ndarray,
    tolerance: float,
    max_iterations: int,
    fixed_sensitivity: np.ndarray | None = None,
) -> CorrectionSolution:
    """Newton iteration on the delta-V applied to ``state_before``.

    With fewer target components than controls the update is the minimum-norm
    one, so the solution stays close to the starting guess.  On failure the
    starting guess is returned with ``converged=False``.

    When ``fixed_sensitivity`` (the partials of the target with respect to the
    delta-V, evaluated on the nominal trajectory) is given, the iteration
    starts as a chord method: the trajectory is re-propagated at every step but
    the partials are not recomputed, which avoids integrating a state
    transition matrix for every sample.  Near the nominal this converges in a
    few cheap steps.  If a step fails to at least halve the previous update,
    the iteration switches to a full Newton method with fresh partials.
    """
    state_before = np.asarray(state_before, dtype=float).reshape(6)
    dv = np.array(guess, dtype=float)
    fallback = CorrectionSolution(np.array(guess, dtype=float), False, max_iterations)
    chord_inverse = None if fixed_sensitivity is None else np.linalg.pinv(fixed_sensitivity)
    previous_step = None
    for iteration in range(1, max_iterations + 1):
        state = state_before.copy()
        state[3:] += dv
        try:
            if chord_inverse is not None:
                for start, end, burn in legs:
                    state = np.array(dynamics.propagate(state, start, end), dtype=float)
                    if burn is not None:
                        state[3:] += burn
                update = -chord_inverse @ target.residual(state, goal)
            else:
                transition = np.eye(6)
                for start, end, burn in legs:
                    state, stm = dynamics.propagate_stm(state, start, end)
                    state = np.array(state, dtype=float)
                    transition = stm @ transition
                    if burn is not None:
                        state[3:] += burn
                sensitivity = target.jacobian(state) @ transition[:, 3:]
                update = -np.linalg.pinv(sensitivity) @ target.residual(state, goal)
        except (PropagationError, ValueError, FloatingPointError):
            return CorrectionSolution(fallback.dv, False, iteration)
        if not np.all(np.isfinite(update)):
            return CorrectionSolution(fallback.dv, False, iteration)
        dv = dv + update
        step = float(np.linalg.norm(update))
        if np.linalg.norm(dv) > _MAX_DV:
            return CorrectionSolution(fallback.dv, False, iteration)
        if step <= tolerance:
            return CorrectionSolution(dv, True, iteration)
        if chord_inverse is not None and previous_step is not None and step > 0.5 * previous_step:
            chord_inverse = None  # the nominal partials are no longer good enough here
        previous_step = step
    return fallback


def solve_correction(
    dynamics: Dynamics,
    nominal: Nominal,
    plan: ManeuverPlan,
    estimate,
    dv_guess=None,
    tolerance: float = 1e-10,
    max_iterations: int = 30,
    reuse_nominal_partials: bool = True,
) -> CorrectionSolution:
    """Re-target a correction maneuver from an estimated state.

    Parameters
    ----------
    dynamics, nominal, plan:
        The propagator, the nominal trajectory and the maneuver being designed.
    estimate:
        Estimated state arriving at the maneuver (before the burn).
    dv_guess:
        Starting delta-V; defaults to the linear guidance solution.
    tolerance:
        The iteration stops when the delta-V update is smaller than this (km/s).
    max_iterations:
        Maximum number of trajectory evaluations.
    reuse_nominal_partials:
        Start with the partials of the nominal trajectory instead of
        recomputing them at every step (see :func:`_newton`).  The solution is
        the same to within ``tolerance``; only the cost differs.

    Notes
    -----
    If the iteration fails, the starting guess is returned with
    ``converged=False``.
    """
    estimate = np.asarray(estimate, dtype=float).reshape(6)
    guess = linear_correction(plan, nominal, estimate) if dv_guess is None else np.asarray(dv_guess, dtype=float).reshape(3)
    fixed = plan.sensitivity[:, 3:] if reuse_nominal_partials else None
    return _newton(
        dynamics, estimate, _legs(plan, nominal), plan.maneuver.target, plan.target_goal, guess, tolerance, max_iterations, fixed
    )


def design_maneuver(
    dynamics: Dynamics,
    state,
    time: float,
    target: Target,
    target_time: float,
    goal,
    dv_guess=(0.0, 0.0, 0.0),
    tolerance: float = 1e-10,
    max_iterations: int = 50,
) -> CorrectionSolution:
    """Find the delta-V at ``time`` that gives ``target`` the value ``goal`` at ``target_time``.

    This is a plain differential corrector for building nominal trajectories.
    ``state`` is the state arriving at the maneuver, before the burn.  The
    iteration is local: it needs a starting guess close enough to a solution.
    """
    if target_time <= time:
        raise ValueError("target_time must be later than the maneuver time")
    goal = np.atleast_1d(np.asarray(goal, dtype=float))
    if goal.shape != (target.size,):
        raise ValueError(f"goal must hold {target.size} value(s)")
    if not 1 <= target.size <= 3:
        raise ValueError("a maneuver can control one to three target components")
    guess = np.asarray(dv_guess, dtype=float).reshape(3)
    legs = [(float(time), float(target_time), None)]
    return _newton(dynamics, state, legs, target, goal, guess, tolerance, max_iterations)
