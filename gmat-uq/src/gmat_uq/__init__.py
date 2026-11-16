# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""GMAT-UQ: linear covariance and navigation-aware Monte Carlo analysis for GMAT.

The package answers two mission-design questions that GMAT does not answer on
its own: how much delta-V margin a trajectory needs, and how accurately the
spacecraft is delivered, given injection errors, maneuver execution errors and
navigation errors.

Typical use::

    from gmat_uq import (Mission, Maneuver, StateDispersion, ManeuverErrorModel,
                         NavigationErrorModel, ParameterTarget, TwoBody,
                         run_lincov, run_sigma_point_check, run_monte_carlo)

See the README in the ``gmat-uq`` folder for a walk-through.
"""

from .consistency import ConsistencyReport, compare_engines
from .dynamics import (
    CR3BP,
    Dynamics,
    GmatDynamics,
    GmatNotFoundError,
    PropagationError,
    TwoBody,
    TwoBodyJ2,
)
from .guidance import design_maneuver, solve_correction
from .lincov import LinCovResult, run_lincov, sample_linear
from .mission import Maneuver, Mission, Nominal, compute_nominal
from .models import ManeuverErrorModel, NavigationErrorModel, ProcessNoise, StateDispersion
from .montecarlo import MonteCarloResult, run_monte_carlo
from .sigma import SigmaPointResult, run_sigma_point_check
from .stats import (
    compare_covariance,
    error_ellipse,
    quantile_with_interval,
    wilks_sample_size,
)
from .targets import FunctionTarget, ParameterTarget, Target

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "ConsistencyReport",
    "compare_engines",
    "CR3BP",
    "Dynamics",
    "GmatDynamics",
    "GmatNotFoundError",
    "PropagationError",
    "TwoBody",
    "TwoBodyJ2",
    "design_maneuver",
    "solve_correction",
    "LinCovResult",
    "run_lincov",
    "sample_linear",
    "Maneuver",
    "Mission",
    "Nominal",
    "compute_nominal",
    "ManeuverErrorModel",
    "NavigationErrorModel",
    "ProcessNoise",
    "StateDispersion",
    "MonteCarloResult",
    "run_monte_carlo",
    "SigmaPointResult",
    "run_sigma_point_check",
    "compare_covariance",
    "error_ellipse",
    "quantile_with_interval",
    "wilks_sample_size",
    "FunctionTarget",
    "ParameterTarget",
    "Target",
]
