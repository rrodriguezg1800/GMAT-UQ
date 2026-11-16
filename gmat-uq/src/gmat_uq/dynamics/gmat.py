# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Dynamics backed by a GMAT installation, through the GMAT Python API (``gmatpy``).

The force model, the integrator and the epoch come from an ordinary GMAT script:
GMAT-UQ loads the script, takes the named ``Spacecraft`` and ``Propagator`` and
uses them to propagate states and state transition matrices.  The mission
sequence of the script is not executed.

Only one GMAT configuration can be active per Python process, because the GMAT
engine is a process-wide singleton.  Monte Carlo workers each load their own.
"""

from __future__ import annotations

import itertools
import os
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import Dynamics, PropagationError, finite_difference_stm

__all__ = ["GmatDynamics", "GmatNotFoundError", "find_gmat_root", "load_gmat", "resources_only_script"]

_STATE_FIELDS = ("X", "Y", "Z", "VX", "VY", "VZ")
_PROPAGATOR_FIELDS = (
    "InitialStepSize",
    "Accuracy",
    "MinStep",
    "MaxStep",
    "LowerError",
    "TargetError",
    "MinimumReduction",
    "MaximumReduction",
    "MinimumTolerance",
)

# Resources that write files or open windows when GMAT initialises them.  GMAT deletes an
# existing report file as soon as its ReportFile is initialised, so these are never loaded.
_OUTPUT_RESOURCES = frozenset(
    name.lower()
    for name in (
        "ReportFile",
        "EphemerisFile",
        "XYPlot",
        "OrbitView",
        "GroundTrackPlot",
        "GroundTrack",
        "OpenFramesInterface",
        "OpenFramesView",
        "OpenFramesVector",
        "OpenFramesSensorMask",
        "DynamicDataDisplay",
        "Enhanced3DView",
    )
)
_CREATE = re.compile(r"^Create\s+(\w+)\s+(.*)$", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"^(?:GMAT\s+)?(\w+)\s*[.=]", re.IGNORECASE)
_BEGIN_SEQUENCE = re.compile(r"^BeginMissionSequence\b", re.IGNORECASE)

_gmat_module = None
_gmat_root: Path | None = None
_tokens = itertools.count(1)
_active_token: int | None = None  # token of the GmatDynamics whose objects are loaded in the engine


def _strip_comment(line: str) -> str:
    """Remove a trailing ``%`` comment, leaving quoted strings intact."""
    in_quote = False
    for position, character in enumerate(line):
        if character == "'":
            in_quote = not in_quote
        elif character == "%" and not in_quote:
            return line[:position]
    return line


def resources_only_script(text: str) -> str:
    """Return a GMAT script that keeps the resources of ``text`` and nothing else.

    The mission sequence is dropped, because GMAT-UQ never runs it, and so are
    the resources that produce output (reports, ephemeris files, plots).  The
    result is what GMAT-UQ actually loads, which guarantees that analysing a
    script never touches the files that script writes.
    """
    removed: set[str] = set()
    kept: list[str] = []
    skipping_continuation = False
    for line in text.splitlines():
        code = _strip_comment(line).strip()
        continues = code.endswith("...")
        if skipping_continuation:
            skipping_continuation = continues
            continue
        if _BEGIN_SEQUENCE.match(code):
            break
        drop = False
        create = _CREATE.match(code)
        if create and create.group(1).lower() in _OUTPUT_RESOURCES:
            names = create.group(2).replace("...", " ").rstrip("; ").replace(",", " ").split()
            removed.update(name.lower() for name in names)
            drop = True
        else:
            assignment = _ASSIGNMENT.match(code)
            if assignment and assignment.group(1).lower() in removed:
                drop = True
        if drop:
            skipping_continuation = continues
            continue
        kept.append(line)
    kept.append("BeginMissionSequence;")
    return "\n".join(kept) + "\n"


class GmatNotFoundError(RuntimeError):
    """Raised when no usable GMAT installation can be located."""


def find_gmat_root(gmat_root=None) -> Path:
    """Locate the top-level GMAT folder (the one that contains ``bin`` and ``api``).

    The explicit argument wins, then the ``GMAT_ROOT`` environment variable.
    """
    candidate = gmat_root or os.environ.get("GMAT_ROOT")
    if not candidate:
        raise GmatNotFoundError(
            "GMAT installation not found: pass gmat_root or set the GMAT_ROOT environment "
            "variable to the folder that contains GMAT's 'bin' directory"
        )
    root = Path(candidate).expanduser().resolve()
    if not (root / "bin" / "gmatpy").is_dir():
        raise GmatNotFoundError(f"'{root}' does not look like a GMAT installation (no bin/gmatpy folder)")
    return root


def _ensure_api_startup_file(root: Path) -> Path:
    """Return the API startup file, creating it the way GMAT's BuildApiStartupFile.py does."""
    startup = root / "bin" / "api_startup_file.txt"
    if startup.exists():
        return startup
    source = root / "bin" / "gmat_startup_file.txt"
    if not source.exists():
        raise GmatNotFoundError(f"'{source}' is missing; cannot build the GMAT API startup file")
    text = source.read_text(encoding="utf-8", errors="replace").replace("..", root.as_posix())
    try:
        startup.write_text(text, encoding="utf-8")
    except OSError as error:
        raise GmatNotFoundError(
            f"cannot write '{startup}' ({error}); run 'python BuildApiStartupFile.py' in GMAT's api folder"
        ) from error
    return startup


def load_gmat(gmat_root=None):
    """Import ``gmatpy`` from a GMAT installation and return the module (cached per process)."""
    global _gmat_module, _gmat_root
    root = find_gmat_root(gmat_root)
    if _gmat_module is not None:
        if root != _gmat_root:
            raise RuntimeError(
                f"GMAT is already loaded from '{_gmat_root}'; a process cannot load a second installation"
            )
        return _gmat_module
    startup = _ensure_api_startup_file(root)
    bin_path = str(root / "bin")
    if bin_path not in sys.path:
        sys.path.insert(1, bin_path)
    try:
        import gmatpy  # type: ignore[import-not-found]
    except ImportError as error:
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise GmatNotFoundError(
            f"could not import gmatpy from '{bin_path}' with Python {version}: {error}"
        ) from error
    gmatpy.Setup(str(startup))
    _gmat_module, _gmat_root = gmatpy, root
    return gmatpy


@dataclass(eq=False)
class GmatDynamics(Dynamics):
    """Propagate with the force model and integrator defined in a GMAT script.

    Parameters
    ----------
    script:
        Path to a GMAT script that creates the spacecraft, force model and
        propagator.  Its mission sequence is ignored.
    spacecraft, propagator:
        Names of the ``Spacecraft`` and ``Propagator`` resources to use.
    gmat_root:
        Top-level GMAT folder.  Defaults to the ``GMAT_ROOT`` environment variable.
    stm_method:
        ``"variational"`` integrates GMAT's own state transition matrix,
        ``"finite_difference"`` differentiates state propagations numerically,
        and ``"auto"`` (default) uses the first and falls back to the second if
        GMAT cannot set up STM propagation for the script.

    Notes
    -----
    States are Cartesian, in the spacecraft's coordinate system, which must use
    MJ2000Eq axes and be centred on the force model's central body.  Time zero
    is the spacecraft epoch given in the script.
    """

    script: str
    spacecraft: str = "Sat"
    propagator: str = "Prop"
    gmat_root: str | None = None
    stm_method: str = "auto"
    mu: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.stm_method not in ("auto", "variational", "finite_difference"):
            raise ValueError("stm_method must be 'auto', 'variational' or 'finite_difference'")
        self.script = str(Path(self.script).expanduser().resolve())
        if not Path(self.script).is_file():
            raise FileNotFoundError(f"GMAT script not found: {self.script}")
        self.gmat_root = str(find_gmat_root(self.gmat_root))
        self._engine = None
        self._token = 0
        self._connect()

    # -- pickling: only the configuration travels to worker processes ---------------------------
    def __getstate__(self) -> dict:
        return {
            "script": self.script,
            "spacecraft": self.spacecraft,
            "propagator": self.propagator,
            "gmat_root": self.gmat_root,
            "stm_method": self.stm_method,
            "mu": self.mu,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._engine = None
        self._token = 0

    # -- engine management ----------------------------------------------------------------------
    def _load_resources(self, gmat) -> None:
        """Load the script's resources through a temporary, output-free copy."""
        source = Path(self.script)
        text = resources_only_script(source.read_text(encoding="utf-8", errors="replace"))
        # The copy sits next to the original so that relative paths keep working.
        for folder in (source.parent, Path(tempfile.gettempdir())):
            try:
                handle = tempfile.NamedTemporaryFile(
                    "w", dir=folder, prefix="gmat_uq_tmp_", suffix=".script", delete=False, encoding="utf-8"
                )
            except OSError:
                continue
            try:
                with handle:
                    handle.write(text)
                loaded = gmat.LoadScript(handle.name)
            finally:
                try:
                    os.unlink(handle.name)
                except OSError:
                    pass
            if not loaded:
                raise RuntimeError(f"GMAT could not load the resources of '{self.script}'")
            return
        raise RuntimeError(f"cannot write a temporary copy of '{self.script}'")

    def _connect(self) -> None:
        global _active_token
        gmat = load_gmat(self.gmat_root)
        gmat.Clear()
        self._load_resources(gmat)
        try:
            sat = gmat.GetObject(self.spacecraft)
            prop = gmat.GetObject(self.propagator)
        except Exception as error:  # gmatpy raises its own exception type
            raise ValueError(
                f"the script does not define Spacecraft '{self.spacecraft}' and Propagator '{self.propagator}': {error}"
            ) from error
        if sat is None or prop is None:
            raise ValueError(
                f"the script does not define Spacecraft '{self.spacecraft}' and Propagator '{self.propagator}'"
            )
        prop.SetField("MaxStepAttempts", 1_000_000)  # Step(dt) gives up after this many internal steps

        # A second propagator, with its own copy of the spacecraft, integrates the STM.  Keeping
        # it separate lets plain propagations (most of a Monte Carlo run) skip that cost.
        stm_sat = stm_prop = None
        stm_error = None
        if self.stm_method != "finite_difference":
            try:
                stm_sat = gmat.Construct("Spacecraft", "GmatUQ_StmSat")
                stm_sat.Copy(sat)
                stm_prop = gmat.Construct("Propagator", "GmatUQ_StmProp")
                integrator = gmat.Construct(prop.GetField("Type"), "GmatUQ_StmIntegrator")
                stm_prop.SetReference(integrator)
                stm_prop.SetReference(gmat.GetObject(prop.GetField("FM")))
                for name in _PROPAGATOR_FIELDS:
                    try:
                        stm_prop.SetField(name, float(prop.GetField(name)))
                    except Exception:
                        pass  # the field does not exist for this integrator type
                stm_prop.SetField("MaxStepAttempts", 1_000_000)
            except Exception as error:
                stm_sat = stm_prop = None
                stm_error = error

        gmat.Initialize()
        prop.AddPropObject(sat)
        prop.PrepareInternals()

        epoch = float(sat.GetEpoch())
        display = np.array([float(sat.GetField(name)) for name in _STATE_FIELDS])
        internal = np.array(prop.GetPropagator().GetState()[:6])
        # The engines work with the numbers GMAT integrates, so the spacecraft's display state
        # must be that same state: inertial MJ2000Eq axes, origin at the central body.
        position_scale = max(float(np.linalg.norm(display[:3])), 1.0)
        velocity_scale = max(float(np.linalg.norm(display[3:])), 1e-3)
        mismatch = (
            np.linalg.norm(display[:3] - internal[:3]) > 1e-6 * position_scale
            or np.linalg.norm(display[3:] - internal[3:]) > 1e-6 * velocity_scale
        )
        try:
            frame = gmat.GetObject(sat.GetField("CoordinateSystem"))
            mismatch = mismatch or str(frame.GetField("Axes")) != "MJ2000Eq"
        except Exception:
            pass  # the numerical comparison above is the check that matters
        if mismatch:
            raise ValueError(
                f"Spacecraft '{self.spacecraft}' must use an MJ2000Eq coordinate system centred on the "
                f"force model's central body ({self._central_body(gmat, prop)}); its coordinate system is "
                f"'{sat.GetField('CoordinateSystem')}'"
            )
        central = self._central_body(gmat, prop)
        mu = float(gmat.GetObject(central).GetField("Mu"))
        fixed_step = None
        if str(gmat.GetObject(prop.GetField("FM")).GetField("ErrorControl")).strip().lower() == "none":
            fixed_step = abs(float(prop.GetField("InitialStepSize")))
            if not fixed_step > 0.0:
                raise ValueError("the propagator's InitialStepSize must be non-zero when ErrorControl is None")

        use_variational = False
        if stm_prop is not None:
            try:
                stm_prop.GetODEModel().SetSolarSystem(gmat.GetSolarSystem())
                stm_prop.AddPropObject(stm_sat)
                stm_prop.GetPropStateManager().SetProperty("STM", stm_sat)
                stm_prop.PrepareInternals()
                use_variational = True
            except Exception as error:
                stm_error = error
        if stm_error is not None:
            if self.stm_method == "variational":
                raise RuntimeError(f"GMAT could not set up STM propagation: {stm_error}") from stm_error
            warnings.warn(
                f"GMAT could not set up STM propagation ({stm_error}); using finite differences instead",
                RuntimeWarning,
                stacklevel=3,
            )

        self._engine = {
            "gmat": gmat,
            "sat": sat,
            "prop": prop,
            "stm_sat": stm_sat,
            "stm_prop": stm_prop,
            "variational": use_variational,
            "fixed_step": fixed_step,
            "epoch": epoch,
            "initial_state": display,
            "central_body": central,
        }
        self.mu = mu
        self._token = next(_tokens)
        _active_token = self._token

    @staticmethod
    def _central_body(gmat, prop) -> str:
        return str(gmat.GetObject(prop.GetField("FM")).GetField("CentralBody"))

    def _handles(self) -> dict:
        if self._engine is None or _active_token != self._token:
            # Either this object was unpickled in a worker process, or another
            # GmatDynamics has replaced the objects in the engine since.
            self._connect()
        return self._engine

    # -- public API -----------------------------------------------------------------------------
    @property
    def epoch_a1mjd(self) -> float:
        """Epoch of time zero, as a GMAT A.1 modified Julian date."""
        return self._handles()["epoch"]

    @property
    def central_body(self) -> str:
        return self._handles()["central_body"]

    @property
    def uses_variational_stm(self) -> bool:
        return bool(self._handles()["variational"])

    def initial_state(self) -> np.ndarray:
        """Cartesian state of the spacecraft as given in the script (km, km/s)."""
        return self._handles()["initial_state"].copy()

    def _body(self, name: str):
        engine = self._handles()
        try:
            body = engine["gmat"].GetObject(name)
            body.GetMJ2000State(engine["epoch"])
        except Exception as error:
            raise ValueError(f"'{name}' is not a celestial body known to GMAT: {error}") from error
        return body

    def body_mu(self, name: str) -> float:
        """Gravitational parameter of a celestial body, km^3/s^2 (GMAT's value)."""
        return float(self._body(name).GetField("Mu"))

    def body_state(self, name: str, time: float) -> np.ndarray:
        """State of a celestial body relative to the central body, in MJ2000Eq axes.

        ``time`` is in seconds from the mission epoch.  The result can be used
        as the ``origin`` of a :class:`gmat_uq.targets.ParameterTarget` to
        target orbit parameters at that body (for example a lunar B-plane).
        """
        engine = self._handles()
        epoch = engine["epoch"] + float(time) / 86400.0

        def state_of(body) -> np.ndarray:
            vector = body.GetMJ2000State(epoch)
            if hasattr(vector, "Get"):
                return np.array([vector.Get(index) for index in range(6)])
            return np.array([vector[index] for index in range(6)])

        return state_of(self._body(name)) - state_of(self._body(engine["central_body"]))

    @staticmethod
    def _load_state(sat, epoch: float, state: np.ndarray) -> None:
        sat.SetRealParameter("A1Epoch", epoch)
        for name, value in zip(_STATE_FIELDS, state):
            sat.SetField(name, float(value))

    @staticmethod
    def _step(prop, dt: float, fixed_step: float | None):
        """Advance the propagator by ``dt`` seconds and return its integrator.

        With error control GMAT sub-steps adaptively inside one ``Step`` call.
        Without it (``ForceModel.ErrorControl = None``) a ``Step`` call is a
        single integrator step, so the interval is walked in steps of the
        propagator's ``InitialStepSize``, as GMAT's ``Propagate`` command does.
        """
        try:
            prop.PrepareInternals()
            integrator = prop.GetPropagator()
            if fixed_step is None:
                if not integrator.Step(dt):
                    raise PropagationError(f"GMAT could not propagate over {dt} s (the integrator stopped early)")
                return integrator
            direction = 1.0 if dt > 0.0 else -1.0
            whole_steps = int(abs(dt) // fixed_step)
            remainder = abs(dt) - whole_steps * fixed_step
            for _ in range(whole_steps):
                if not integrator.Step(direction * fixed_step):
                    raise PropagationError(f"GMAT could not take a fixed step of {fixed_step} s")
            if remainder > 1e-9:
                if not integrator.Step(direction * remainder):
                    raise PropagationError(f"GMAT could not take a final step of {remainder} s")
            return integrator
        except PropagationError:
            raise
        except Exception as error:  # gmatpy raises its own exception type for force-model errors
            raise PropagationError(f"GMAT failed while propagating over {dt} s: {error}") from error

    def propagate(self, state, t0: float, t1: float) -> np.ndarray:
        state = np.asarray(state, dtype=float).reshape(6)
        dt = float(t1) - float(t0)
        if dt == 0.0:
            return state.copy()
        engine = self._handles()
        self._load_state(engine["sat"], engine["epoch"] + float(t0) / 86400.0, state)
        integrator = self._step(engine["prop"], dt, engine["fixed_step"])
        return np.array(integrator.GetState()[:6])

    def propagate_stm(self, state, t0: float, t1: float):
        state = np.asarray(state, dtype=float).reshape(6)
        dt = float(t1) - float(t0)
        if dt == 0.0:
            return state.copy(), np.eye(6)
        engine = self._handles()
        if not engine["variational"]:
            return self.propagate(state, t0, t1), finite_difference_stm(self.propagate, state, t0, t1)
        sat = engine["stm_sat"]
        self._load_state(sat, engine["epoch"] + float(t0) / 86400.0, state)
        for row in range(6):
            for column in range(6):
                sat.SetRealParameter("STM", 1.0 if row == column else 0.0, row, column)
        integrator = self._step(engine["stm_prop"], dt, engine["fixed_step"])
        integrator.UpdateSpaceObject()
        matrix = sat.GetRmatrixParameter("STM")
        stm = np.array([[matrix.GetElement(row, column) for column in range(6)] for row in range(6)])
        return np.array(integrator.GetState()[:6]), stm

    def describe(self) -> dict:
        engine = self._handles()
        return {
            "type": "GmatDynamics",
            "script": self.script,
            "spacecraft": self.spacecraft,
            "propagator": self.propagator,
            "central_body": engine["central_body"],
            "mu": self.mu,
            "epoch_a1mjd": engine["epoch"],
            "stm": "variational" if engine["variational"] else "finite_difference",
            "step_control": "adaptive" if engine["fixed_step"] is None else f"fixed {engine['fixed_step']:g} s",
        }
