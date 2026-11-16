"""GMAT script templates used by the GMAT integration tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

SPACECRAFT = """\
Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '01 Jan 2026 12:00:00.000';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Cartesian;
Sat.X = {x};
Sat.Y = {y};
Sat.Z = {z};
Sat.VX = {vx};
Sat.VY = {vy};
Sat.VZ = {vz};
"""

LEO_STATE = dict(x=7000, y=0, z=1300, vx=0, vy=7.35, vz=1.0)
FLYBY_STATE = dict(x=-50000, y=30000, z=10000, vx=6.0, vy=-1.5, vz=0.5)

POINT_MASS = SPACECRAFT + """\
Create ForceModel FM;
FM.CentralBody = Earth;
FM.PointMasses = {{Earth}};
FM.Drag = None;
FM.SRP = Off;
FM.ErrorControl = RSSStep;
Create Propagator Prop;
Prop.FM = FM;
Prop.Type = PrinceDormand78;
Prop.InitialStepSize = 60;
Prop.Accuracy = 1e-13;
Prop.MinStep = 0;
Prop.MaxStep = 2700;
BeginMissionSequence;
"""

# Earth gravity field plus the Moon and the Sun.  {error_control} is RSSStep (adaptive steps) or
# None (fixed steps, which GMAT's native covariance propagation requires).
HIGH_FIDELITY = SPACECRAFT + """\
Create ForceModel FM;
FM.CentralBody = Earth;
FM.PrimaryBodies = {{Earth}};
FM.GravityField.Earth.Degree = 4;
FM.GravityField.Earth.Order = 4;
FM.PointMasses = {{Luna, Sun}};
FM.Drag = None;
FM.SRP = Off;
FM.ErrorControl = {error_control};
Create Propagator Prop;
Prop.FM = FM;
Prop.Type = RungeKutta89;
Prop.InitialStepSize = 20;
Prop.Accuracy = 1e-13;
Prop.MinStep = 0;
Prop.MaxStep = 2700;
"""

REPORTS = """\
Create ReportFile StateReport;
StateReport.Filename = '{state_file}';
StateReport.WriteHeaders = False;
StateReport.Precision = 16;
StateReport.ColumnWidth = 26;
StateReport.WriteReport = False;
Create ReportFile CovReport ...
   ;
CovReport.Filename = '{cov_file}';
CovReport.WriteHeaders = False;
CovReport.Precision = 16;
CovReport.ColumnWidth = 26;
CovReport.WriteReport = False;
"""

NATIVE_COVARIANCE_SEQUENCE = """\
BeginMissionSequence;
Sat.OrbitErrorCovariance = diag([{diagonal}]);
Propagate Prop(Sat, 'Covariance') {{Sat.ElapsedSecs = {duration}}};
Report StateReport Sat.EarthMJ2000Eq.X Sat.EarthMJ2000Eq.Y Sat.EarthMJ2000Eq.Z Sat.EarthMJ2000Eq.VX Sat.EarthMJ2000Eq.VY Sat.EarthMJ2000Eq.VZ;
Report CovReport Sat.EarthMJ2000Eq.OrbitErrorCovariance;
"""

PARAMETER_REPORT = SPACECRAFT + """\
Create ReportFile Rep;
Rep.Filename = '{report_file}';
Rep.WriteHeaders = False;
Rep.Precision = 16;
Rep.ColumnWidth = 26;
Rep.WriteReport = False;
BeginMissionSequence;
Report Rep Sat.EarthMJ2000Eq.BdotT Sat.EarthMJ2000Eq.BdotR Sat.Earth.SMA Sat.Earth.ECC Sat.EarthMJ2000Eq.INC Sat.EarthMJ2000Eq.RAAN Sat.EarthMJ2000Eq.AOP Sat.Earth.TA Sat.Earth.RMAG Sat.Earth.C3Energy Sat.Earth.RadPer Sat.EarthMJ2000Eq.VMAG;
"""
PARAMETER_NAMES = ("BdotT", "BdotR", "SMA", "ECC", "INC", "RAAN", "AOP", "TA", "RMAG", "C3Energy", "RadPer", "VMAG")


def write_script(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def run_gmat_console(gmat_root: str, script: Path) -> None:
    """Run a script with GMAT's own console application, exactly as a GMAT user would."""
    binary = Path(gmat_root) / "bin" / "GmatConsole.exe"
    if not binary.exists():
        binary = Path(gmat_root) / "bin" / "GmatConsole"
    if not binary.exists():
        binary = sorted((Path(gmat_root) / "bin").glob("GmatConsole*"))[0]
    completed = subprocess.run(
        [str(binary), "--run", str(script)], capture_output=True, text=True, timeout=600, cwd=str(binary.parent)
    )
    if completed.returncode != 0:
        raise RuntimeError(f"GmatConsole failed:\n{completed.stdout[-3000:]}\n{completed.stderr[-2000:]}")

# Low Earth orbit with drag (exponential atmosphere, no space-weather files needed) and SRP,
# used for consider parameters such as Cd and Cr.
DRAG = """\
Create Spacecraft Sat;
Sat.DateFormat = UTCGregorian;
Sat.Epoch = '01 Jan 2026 12:00:00.000';
Sat.CoordinateSystem = EarthMJ2000Eq;
Sat.DisplayStateType = Cartesian;
Sat.X = 6678.1363;
Sat.Y = 0;
Sat.Z = 0;
Sat.VX = 0;
Sat.VY = 6.785;
Sat.VZ = 3.9;
Sat.DryMass = 500;
Sat.Cd = 2.2;
Sat.DragArea = 4;
Sat.Cr = 1.8;
Sat.SRPArea = 4;
Create ForceModel FM;
FM.CentralBody = Earth;
FM.PrimaryBodies = {Earth};
FM.GravityField.Earth.Degree = 2;
FM.GravityField.Earth.Order = 0;
FM.Drag.AtmosphereModel = Exponential;
FM.SRP = On;
FM.ErrorControl = RSSStep;
Create Propagator Prop;
Prop.FM = FM;
Prop.Type = PrinceDormand78;
Prop.InitialStepSize = 60;
Prop.Accuracy = 1e-12;
Prop.MinStep = 0;
Prop.MaxStep = 600;
BeginMissionSequence;
"""
