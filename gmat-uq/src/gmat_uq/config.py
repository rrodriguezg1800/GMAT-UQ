# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Load an analysis from a TOML or JSON configuration file.

The file mirrors the resources of the GMAT-UQ design: error models and
navigation models are declared once by name and attached to maneuvers.  See
``docs/configuration.md`` for the full reference.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dynamics import CR3BP, Dynamics, GmatDynamics, TwoBody, TwoBodyJ2
from .dynamics.ode import EARTH_J2, EARTH_MU, EARTH_RADIUS
from .mission import Maneuver, Mission
from .models import (
    ConsiderParameter,
    ManeuverErrorModel,
    NavigationErrorModel,
    ProcessNoise,
    StateDispersion,
    rotate_covariance_to_inertial,
)
from .targets import ParameterTarget

__all__ = ["AnalysisConfig", "ConfigError", "load_config", "build_from_dict"]

METHODS = ("lincov", "sigma_point", "monte_carlo")


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass
class AnalysisConfig:
    """Everything needed to run an analysis."""

    mission: Mission
    dynamics: Dynamics
    methods: tuple[str, ...] = METHODS
    samples: int = 1000
    seed: int = 0
    workers: int = 1
    guidance: str = "nonlinear"
    dv_samples: int = 20000
    threshold: float = 0.1
    probability: float = 0.99
    confidence: float = 0.95
    report: str | None = None
    source: str | None = field(default=None, repr=False)


def _check_keys(section: dict, allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"unknown key(s) in {where}: {', '.join(sorted(unknown))}")


def _require(section: dict, key: str, where: str):
    if key not in section:
        raise ConfigError(f"missing '{key}' in {where}")
    return section[key]


def _covariance(section: dict, where: str, base: Path) -> tuple[np.ndarray, str]:
    """Read a covariance given as sigmas, a matrix or a file; return it with its frame name."""
    _check_keys(section, {"sigmas", "covariance", "covariance_file", "block", "frame"}, where)
    if "block" in section and "covariance_file" not in section:
        raise ConfigError(f"{where}: 'block' only applies to a covariance_file")
    given = [key for key in ("sigmas", "covariance", "covariance_file") if key in section]
    if len(given) != 1:
        raise ConfigError(f"{where} needs exactly one of 'sigmas', 'covariance' or 'covariance_file'")
    try:
        if "sigmas" in section:
            covariance = StateDispersion.from_sigmas(section["sigmas"]).covariance
        elif "covariance" in section:
            covariance = StateDispersion(np.array(section["covariance"], dtype=float)).covariance
        else:
            covariance = StateDispersion.from_file(base / section["covariance_file"], int(section.get("block", -1))).covariance
    except (ValueError, OSError) as error:
        raise ConfigError(f"{where}: {error}") from error
    return np.array(covariance), str(section.get("frame", "inertial"))


def _build_dynamics(section: dict, base: Path) -> Dynamics:
    kind = str(_require(section, "type", "[dynamics]")).lower()
    try:
        if kind in ("two_body", "twobody"):
            _check_keys(section, {"type", "mu"}, "[dynamics]")
            return TwoBody(mu=float(section.get("mu", EARTH_MU)))
        if kind in ("two_body_j2", "j2"):
            _check_keys(section, {"type", "mu", "j2", "radius"}, "[dynamics]")
            return TwoBodyJ2(
                mu=float(section.get("mu", EARTH_MU)),
                j2=float(section.get("j2", EARTH_J2)),
                radius=float(section.get("radius", EARTH_RADIUS)),
            )
        if kind == "cr3bp":
            _check_keys(section, {"type", "mass_ratio"}, "[dynamics]")
            return CR3BP(mass_ratio=float(section.get("mass_ratio", CR3BP().mass_ratio)))
        if kind == "gmat":
            _check_keys(section, {"type", "script", "spacecraft", "propagator", "gmat_root", "stm_method"}, "[dynamics]")
            return GmatDynamics(
                script=str(base / _require(section, "script", "[dynamics]")),
                spacecraft=str(section.get("spacecraft", "Sat")),
                propagator=str(section.get("propagator", "Prop")),
                gmat_root=section.get("gmat_root"),
                stm_method=str(section.get("stm_method", "auto")),
            )
    except (TypeError, ValueError, FileNotFoundError) as error:
        if isinstance(error, ConfigError):
            raise
        raise ConfigError(f"[dynamics]: {error}") from error
    raise ConfigError(f"unknown dynamics type '{kind}' (use two_body, two_body_j2, cr3bp or gmat)")


def _build_target(spec, dynamics: Dynamics, time: float, where: str) -> ParameterTarget:
    """Build a target from a list of parameter names, or from a table that also names a body.

    ``target = ["X", "Y", "Z"]`` refers to the central body of the dynamics.
    ``target = { parameters = ["BdotT", "BdotR"], body = "Luna" }`` evaluates the
    parameters relative to another body at the time of the target; this needs
    GMAT dynamics, which provide the body's ephemeris and gravitational parameter.
    """
    mu = dynamics.mu
    origin = None
    names = spec
    if isinstance(spec, dict):
        _check_keys(spec, {"parameters", "body"}, f"{where} target")
        names = _require(spec, "parameters", f"{where} target")
        body = spec.get("body")
        if body is not None:
            if not isinstance(dynamics, GmatDynamics):
                raise ConfigError(f"{where}: a target relative to a body ('{body}') needs dynamics of type 'gmat'")
            try:
                if str(body).lower() != dynamics.central_body.lower():
                    origin = tuple(dynamics.body_state(str(body), time))
                    mu = dynamics.body_mu(str(body))
            except ValueError as error:
                raise ConfigError(f"{where}: {error}") from error
    try:
        return ParameterTarget(tuple(names) if not isinstance(names, str) else (names,), mu, origin)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{where}: {error}") from error


def _states_at(times, initial_state, initial_time, burns, dynamics: Dynamics) -> dict[float, np.ndarray]:
    """Nominal state arriving at each requested time, used to orient local frames."""
    events = sorted(set(times) | {time for time, _ in burns})
    burn_at = dict(burns)
    state = np.array(initial_state, dtype=float)
    current = initial_time
    result = {}
    for time in events:
        state = np.asarray(dynamics.propagate(state, current, time), dtype=float)
        current = time
        if time in times:
            result[time] = state.copy()
        if time in burn_at:
            state = state.copy()
            state[3:] += burn_at[time]
    return result


def build_from_dict(data: dict, base: Path | str = ".") -> AnalysisConfig:
    """Build an :class:`AnalysisConfig` from a parsed configuration dictionary."""
    base = Path(base)
    _check_keys(data, {"dynamics", "mission", "error_models", "navigation", "maneuver", "consider", "analysis"}, "the configuration")
    dynamics = _build_dynamics(_require(data, "dynamics", "the configuration"), base)

    mission_section = dict(_require(data, "mission", "the configuration"))
    _check_keys(
        mission_section,
        {"initial_state", "initial_time", "final_time", "report_times", "final_target", "initial_dispersion", "process_noise", "epoch_a1mjd"},
        "[mission]",
    )
    if "initial_state" in mission_section:
        initial_state = np.array(mission_section["initial_state"], dtype=float)
    elif isinstance(dynamics, GmatDynamics):
        initial_state = dynamics.initial_state()
    else:
        raise ConfigError("missing 'initial_state' in [mission]")
    if initial_state.shape != (6,):
        raise ConfigError("[mission] initial_state must hold six values")
    # Times may be given from an epoch other than the script's (this is how the GMAT-UQ plugin
    # passes the moment RunUncertaintyAnalysis executes).  They are shifted to the script's epoch.
    offset = 0.0
    if "epoch_a1mjd" in mission_section:
        if not isinstance(dynamics, GmatDynamics):
            raise ConfigError("[mission] epoch_a1mjd needs dynamics of type 'gmat'")
        offset = (float(mission_section["epoch_a1mjd"]) - dynamics.epoch_a1mjd) * 86400.0

    def shifted(value) -> float:
        return float(value) + offset

    initial_time = shifted(mission_section.get("initial_time", 0.0))
    final_time = shifted(_require(mission_section, "final_time", "[mission]"))

    error_models = {}
    for name, section in data.get("error_models", {}).items():
        try:
            error_models[name] = ManeuverErrorModel(**section)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"[error_models.{name}]: {error}") from error

    navigation_raw = {
        name: _covariance(section, f"[navigation.{name}]", base) for name, section in data.get("navigation", {}).items()
    }

    maneuver_sections = data.get("maneuver", [])
    if not isinstance(maneuver_sections, list):
        raise ConfigError("maneuvers must be given as an array of tables: [[maneuver]]")
    burns = []
    for section in maneuver_sections:
        _check_keys(section, {"name", "time", "dv", "error_model", "navigation", "target", "target_time"}, "[[maneuver]]")
        burns.append((shifted(_require(section, "time", "[[maneuver]]")), np.array(section.get("dv", [0.0, 0.0, 0.0]), dtype=float)))

    # Local frames (RIC, VNB) are oriented with the nominal state at the time they apply.
    local_times = set()
    for section in maneuver_sections:
        name = section.get("navigation")
        if name is not None:
            if name not in navigation_raw:
                raise ConfigError(f"maneuver '{section.get('name')}' refers to unknown navigation model '{name}'")
            if navigation_raw[name][1].lower() not in ("inertial", "mj2000eq"):
                local_times.add(shifted(section["time"]))
    local_states = _states_at(local_times, initial_state, initial_time, burns, dynamics) if local_times else {}

    maneuvers = []
    for section in maneuver_sections:
        name = str(_require(section, "name", "[[maneuver]]"))
        where = f"maneuver '{name}'"
        time = shifted(section["time"])
        error_model = None
        if "error_model" in section:
            if section["error_model"] not in error_models:
                raise ConfigError(f"{where} refers to unknown error model '{section['error_model']}'")
            error_model = error_models[section["error_model"]]
        navigation = None
        if "navigation" in section:
            covariance, frame = navigation_raw[section["navigation"]]
            if frame.lower() not in ("inertial", "mj2000eq"):
                try:
                    covariance = rotate_covariance_to_inertial(covariance, local_states[time], frame)
                except ValueError as error:
                    raise ConfigError(f"[navigation.{section['navigation']}]: {error}") from error
            navigation = NavigationErrorModel(covariance)
        target = None
        if "target" in section:
            if "target_time" not in section:
                raise ConfigError(f"{where}: a target needs a target_time")
            target = _build_target(section["target"], dynamics, shifted(section["target_time"]), where)
        try:
            maneuvers.append(
                Maneuver(
                    name=name,
                    time=time,
                    dv=tuple(np.array(section.get("dv", [0.0, 0.0, 0.0]), dtype=float)),
                    error_model=error_model,
                    target=target,
                    target_time=None if "target_time" not in section else shifted(section["target_time"]),
                    navigation=navigation,
                )
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{where}: {error}") from error

    initial_dispersion = None
    if "initial_dispersion" in mission_section:
        covariance, frame = _covariance(mission_section["initial_dispersion"], "[mission.initial_dispersion]", base)
        try:
            initial_dispersion = StateDispersion(rotate_covariance_to_inertial(covariance, initial_state, frame))
        except ValueError as error:
            raise ConfigError(f"[mission.initial_dispersion]: {error}") from error

    process_noise = None
    if "process_noise" in mission_section:
        section = mission_section["process_noise"]
        _check_keys(section, {"accel_sigma", "max_step"}, "[mission.process_noise]")
        try:
            process_noise = ProcessNoise(
                tuple(_require(section, "accel_sigma", "[mission.process_noise]")), float(section.get("max_step", 600.0))
            )
        except (TypeError, ValueError) as error:
            raise ConfigError(f"[mission.process_noise]: {error}") from error

    consider_sections = data.get("consider", [])
    if not isinstance(consider_sections, list):
        raise ConfigError("consider parameters must be given as an array of tables: [[consider]]")
    consider = []
    for section in consider_sections:
        _check_keys(section, {"name", "sigma"}, "[[consider]]")
        name = str(_require(section, "name", "[[consider]]"))
        try:
            item = ConsiderParameter(name, float(_require(section, "sigma", "[[consider]]")))
            dynamics.parameter_value(item.name)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"consider parameter '{name}': {error}") from error
        consider.append(item)

    final_target = None
    if "final_target" in mission_section:
        final_target = _build_target(mission_section["final_target"], dynamics, final_time, "[mission] final_target")

    try:
        mission = Mission(
            initial_state=tuple(initial_state),
            final_time=final_time,
            maneuvers=tuple(maneuvers),
            initial_dispersion=initial_dispersion,
            process_noise=process_noise,
            report_times=tuple(shifted(t) for t in mission_section.get("report_times", ())),
            final_target=final_target,
            initial_time=initial_time,
            consider=tuple(consider),
        )
    except (TypeError, ValueError) as error:
        raise ConfigError(f"[mission]: {error}") from error

    analysis = dict(data.get("analysis", {}))
    _check_keys(
        analysis,
        {"methods", "samples", "seed", "workers", "guidance", "dv_samples", "threshold", "probability", "confidence", "report"},
        "[analysis]",
    )
    methods = analysis.get("methods", list(METHODS))
    if isinstance(methods, str):
        methods = [methods]
    methods = tuple(str(method).lower() for method in methods)
    for method in methods:
        if method not in METHODS:
            raise ConfigError(f"unknown method '{method}' in [analysis] (use {', '.join(METHODS)})")
    if not methods:
        raise ConfigError("[analysis] methods is empty")
    guidance = str(analysis.get("guidance", "nonlinear")).lower()
    if guidance not in ("nonlinear", "linear"):
        raise ConfigError("[analysis] guidance must be 'nonlinear' or 'linear'")
    config = AnalysisConfig(
        mission=mission,
        dynamics=dynamics,
        methods=methods,
        samples=int(analysis.get("samples", 1000)),
        seed=int(analysis.get("seed", 0)),
        workers=int(analysis.get("workers", 1)),
        guidance=guidance,
        dv_samples=int(analysis.get("dv_samples", 20000)),
        threshold=float(analysis.get("threshold", 0.1)),
        probability=float(analysis.get("probability", 0.99)),
        confidence=float(analysis.get("confidence", 0.95)),
        report=None if analysis.get("report") is None else str(base / analysis["report"]),
    )
    if config.samples < 1 or config.workers < 1 or config.dv_samples < 0:
        raise ConfigError("[analysis] samples and workers must be at least 1, and dv_samples non-negative")
    if not 0.0 < config.probability < 1.0 or not 0.0 < config.confidence < 1.0:
        raise ConfigError("[analysis] probability and confidence must lie strictly between 0 and 1")
    if config.threshold <= 0.0:
        raise ConfigError("[analysis] threshold must be positive")
    return config


def load_config(path) -> AnalysisConfig:
    """Load an analysis configuration from a ``.toml`` or ``.json`` file.

    Relative paths inside the file are resolved against the file's folder.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"cannot parse {path.name}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a table at the top level")
    try:
        config = build_from_dict(data, path.parent)
    except ConfigError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        # A value of the wrong kind (text where a number is expected, a table where a list is...).
        raise ConfigError(f"invalid value in {path.name}: {error}") from error
    config.source = str(path)
    return config
