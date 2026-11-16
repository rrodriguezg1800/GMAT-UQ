//------------------------------------------------------------------------------
//                              UQResources
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------

#include "UQResources.hpp"

namespace
{
   UQField Real_(const char *name, Real value = 0.0)
   {
      UQField f = {name, Gmat::REAL_TYPE, false, value, ""};
      return f;
   }
   UQField Int_(const char *name, Integer value)
   {
      UQField f = {name, Gmat::INTEGER_TYPE, false, (Real)value, ""};
      return f;
   }
   UQField Text(const char *name, const char *value = "")
   {
      UQField f = {name, Gmat::STRING_TYPE, false, 0.0, value};
      return f;
   }
   UQField Names(const char *name)
   {
      UQField f = {name, Gmat::STRINGARRAY_TYPE, false, 0.0, ""};
      return f;
   }
   UQField Vector(const char *name)
   {
      UQField f = {name, Gmat::RVECTOR_TYPE, false, 0.0, ""};
      return f;
   }
   const char *DefaultPython()
   {
   #ifdef _WIN32
      return "python";
   #else
      return "python3";
   #endif
   }
}

//------------------------------------------------------------------------------
// ManeuverErrorModel
//------------------------------------------------------------------------------
const std::vector<UQField>& ManeuverErrorModel::Fields()
{
   static const std::vector<UQField> table = {
      Real_("FixedMagnitudeSigma"),          // km/s
      Real_("ProportionalMagnitudeSigma"),   // fraction of the commanded delta-V
      Real_("FixedPointingSigma"),           // km/s per axis
      Real_("ProportionalPointingSigma"),    // rad per axis
   };
   return table;
}

ManeuverErrorModel::ManeuverErrorModel(const std::string &name) :
   UQObject("ManeuverErrorModel", name, Fields())
{
}

GmatBase* ManeuverErrorModel::Clone() const
{
   return new ManeuverErrorModel(*this);
}

//------------------------------------------------------------------------------
// StateDispersion and NavigationErrorModel
//------------------------------------------------------------------------------
const std::vector<UQField>& StateDispersion::Fields()
{
   static const std::vector<UQField> table = {
      Vector("Sigmas"),                      // six values (km, km/s), or two: position, velocity
      Text("CovarianceFile"),                // 6x6 matrix, as a GMAT ReportFile writes it
      Text("Frame", "MJ2000Eq"),             // MJ2000Eq, RIC or VNB
   };
   return table;
}

StateDispersion::StateDispersion(const std::string &name) :
   UQObject("StateDispersion", name, Fields())
{
}

StateDispersion::StateDispersion(const std::string &typeStr, const std::string &name) :
   UQObject(typeStr, name, Fields())
{
}

GmatBase* StateDispersion::Clone() const
{
   return new StateDispersion(*this);
}

NavigationErrorModel::NavigationErrorModel(const std::string &name) :
   StateDispersion("NavigationErrorModel", name)
{
}

GmatBase* NavigationErrorModel::Clone() const
{
   return new NavigationErrorModel(*this);
}

//------------------------------------------------------------------------------
// UncertainManeuver
//------------------------------------------------------------------------------
const std::vector<UQField>& UncertainManeuver::Fields()
{
   static const std::vector<UQField> table = {
      Real_("Time"),                         // s after RunUncertaintyAnalysis
      Vector("DeltaV"),                      // nominal delta-V, km/s, MJ2000Eq axes
      Text("ErrorModel"),                    // a ManeuverErrorModel
      Text("Navigation"),                    // a NavigationErrorModel (corrections only)
      Names("Target"),                       // makes the maneuver a correction, e.g. {BdotT, BdotR}
      Text("TargetBody"),                    // evaluate the target relative to this body
      Real_("TargetTime"),                   // s after RunUncertaintyAnalysis
   };
   return table;
}

UncertainManeuver::UncertainManeuver(const std::string &name) :
   UQObject("UncertainManeuver", name, Fields())
{
}

GmatBase* UncertainManeuver::Clone() const
{
   return new UncertainManeuver(*this);
}

//------------------------------------------------------------------------------
// UncertaintyAnalysis
//------------------------------------------------------------------------------
const std::vector<UQField>& UncertaintyAnalysis::Fields()
{
   static const std::vector<UQField> table = {
      Text("Spacecraft"),
      Text("Propagator"),
      Real_("Duration"),                     // s after RunUncertaintyAnalysis
      Text("InitialDispersion"),             // a StateDispersion
      Names("Maneuvers"),                    // UncertainManeuver resources
      Names("ConsiderParameters"),           // Spacecraft fields, e.g. {Cd, Cr}
      Vector("ConsiderSigmas"),              // one sigma per consider parameter
      Names("FinalTarget"),                  // dispersion also reported in these parameters
      Text("FinalTargetBody"),
      Names("Methods"),                      // LinCov, SigmaPoint, MonteCarlo (default: all)
      Int_("Samples", 1000),
      Int_("Seed", 0),
      Int_("Workers", 1),
      Text("Guidance", "Nonlinear"),         // Nonlinear or Linear
      Int_("DeltaVSamples", 20000),
      Real_("NonlinearityThreshold", 0.1),
      Real_("Probability", 0.99),
      Real_("Confidence", 0.95),
      Text("ReportFile"),                    // default: <output folder>/<name>.json
      Text("PythonExecutable", DefaultPython()),
   };
   return table;
}

const StringArray& UncertaintyAnalysis::ResultNames()
{
   static const StringArray names = {
      "DeltaV99", "DeltaVMean", "MonteCarloDeltaV99", "FinalPositionSigma",
      "FinalVelocitySigma", "NonlinearityIndex", "LinearTrusted", "EnginesAgree",
   };
   return names;
}

UncertaintyAnalysis::UncertaintyAnalysis(const std::string &name) :
   UQObject("UncertaintyAnalysis", name, Fields())
{
}

GmatBase* UncertaintyAnalysis::Clone() const
{
   return new UncertaintyAnalysis(*this);
}
