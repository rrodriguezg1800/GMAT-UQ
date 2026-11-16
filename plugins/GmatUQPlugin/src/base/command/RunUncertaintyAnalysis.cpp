//------------------------------------------------------------------------------
//                           RunUncertaintyAnalysis
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------

#include "RunUncertaintyAnalysis.hpp"
#include "UQResources.hpp"
#include "CommandException.hpp"
#include "FileManager.hpp"
#include "MessageInterface.hpp"
#include "Moderator.hpp"
#include "SpaceObject.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#ifndef _WIN32
   #include <sys/wait.h>
#endif

namespace fs = std::filesystem;

namespace
{
   const char *STATE_FIELDS[6] = {"X", "Y", "Z", "VX", "VY", "VZ"};

   std::string Number(Real value)
   {
      if (!std::isfinite(value))
         throw CommandException("RunUncertaintyAnalysis: a value is not a finite number");
      std::ostringstream text;
      text << std::setprecision(17) << value;
      std::string out = text.str();
      // TOML needs a digit after the exponent marker and accepts plain integers as floats
      // only when written with a decimal point in some readers; keep it explicit.
      if (out.find_first_of(".eEn") == std::string::npos)
         out += ".0";
      return out;
   }

   std::string Numbers(const std::vector<Real> &values)
   {
      std::string out = "[";
      for (size_t i = 0; i < values.size(); ++i)
         out += (i ? ", " : "") + Number(values[i]);
      return out + "]";
   }

   /// Basic TOML string; names in GMAT scripts never contain quotes or backslashes
   std::string Quoted(const std::string &text)
   {
      std::string out = "\"";
      for (char c : text)
      {
         if (c == '"' || c == '\\')
            out += '\\';
         out += c;
      }
      return out + "\"";
   }

   std::string QuotedList(const StringArray &items)
   {
      std::string out = "[";
      for (size_t i = 0; i < items.size(); ++i)
         out += (i ? ", " : "") + Quoted(items[i]);
      return out + "]";
   }

   std::string Target(const StringArray &names, const std::string &body)
   {
      if (body.empty())
         return QuotedList(names);
      return "{ parameters = " + QuotedList(names) + ", body = " + Quoted(body) + " }";
   }

   std::string Method(const std::string &name)
   {
      if (name == "LinCov" || name == "LinearCovariance")  return "lincov";
      if (name == "SigmaPoint")                            return "sigma_point";
      if (name == "MonteCarlo")                            return "monte_carlo";
      throw CommandException("RunUncertaintyAnalysis: unknown method '" + name +
                             "' (use LinCov, SigmaPoint or MonteCarlo)");
   }

   std::string Guidance(const std::string &name)
   {
      if (name == "Nonlinear" || name == "nonlinear")  return "nonlinear";
      if (name == "Linear" || name == "linear")        return "linear";
      throw CommandException("RunUncertaintyAnalysis: Guidance must be Nonlinear or Linear, not '" +
                             name + "'");
   }

   std::string CovarianceSection(UQObject *source, const std::string &header)
   {
      std::string out = header + "\n";
      const std::vector<Real> &sigmas = source->GetReals("Sigmas");
      std::string file = source->GetString("CovarianceFile");
      if (!sigmas.empty() && !file.empty())
         throw CommandException(source->GetName() + ": set either Sigmas or CovarianceFile, not both");
      if (!sigmas.empty())
      {
         if (sigmas.size() != 6 && sigmas.size() != 2)
            throw CommandException(source->GetName() + ".Sigmas needs six values, or two (position, velocity)");
         out += "sigmas = " + Numbers(sigmas) + "\n";
      }
      else if (!file.empty())
         out += "covariance_file = " + Quoted(fs::absolute(fs::path(file)).generic_string()) + "\n";
      else
         throw CommandException(source->GetName() + ": set Sigmas or CovarianceFile");
      out += "frame = " + Quoted(source->GetString("Frame")) + "\n\n";
      return out;
   }

   std::string Quote(const std::string &path)
   {
      return "\"" + path + "\"";
   }

   std::string Tail(const std::string &file, size_t lines)
   {
      std::ifstream in(file.c_str());
      std::vector<std::string> all;
      std::string line;
      while (std::getline(in, line))
         all.push_back(line);
      std::string out;
      for (size_t i = all.size() > lines ? all.size() - lines : 0; i < all.size(); ++i)
         out += all[i] + "\n";
      return out;
   }
}

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------
RunUncertaintyAnalysis::RunUncertaintyAnalysis() :
   GmatCommand    ("RunUncertaintyAnalysis")
{
   objectTypeNames.push_back("RunUncertaintyAnalysis");
}

RunUncertaintyAnalysis::~RunUncertaintyAnalysis()
{
}

RunUncertaintyAnalysis::RunUncertaintyAnalysis(const RunUncertaintyAnalysis &other) :
   GmatCommand    (other),
   analysisName   (other.analysisName)
{
}

RunUncertaintyAnalysis& RunUncertaintyAnalysis::operator=(const RunUncertaintyAnalysis &other)
{
   if (this != &other)
   {
      GmatCommand::operator=(other);
      analysisName = other.analysisName;
   }
   return *this;
}

GmatBase* RunUncertaintyAnalysis::Clone() const
{
   return new RunUncertaintyAnalysis(*this);
}

//------------------------------------------------------------------------------
// Parsing:  RunUncertaintyAnalysis <UncertaintyAnalysis name>
//------------------------------------------------------------------------------
bool RunUncertaintyAnalysis::InterpretAction()
{
   std::string text = generatingString;
   std::string::size_type semicolon = text.find(';');
   if (semicolon != std::string::npos)
      text = text.substr(0, semicolon);
   std::stringstream words(text);
   std::string keyword, name, extra;
   words >> keyword >> name >> extra;
   if (name.empty() || !extra.empty())
      throw CommandException("RunUncertaintyAnalysis takes exactly one argument, the name of an "
                             "UncertaintyAnalysis: \"" + generatingString + "\"");
   analysisName = name;
   return true;
}

bool RunUncertaintyAnalysis::RenameRefObject(const UnsignedInt type, const std::string &oldName,
                                             const std::string &newName)
{
   if (analysisName == oldName)
      analysisName = newName;
   return true;
}

const std::string& RunUncertaintyAnalysis::GetGeneratingString(Gmat::WriteMode mode,
                                                               const std::string &prefix,
                                                               const std::string &useName)
{
   generatingString = prefix + "RunUncertaintyAnalysis " + analysisName + ";";
   return GmatCommand::GetGeneratingString(mode, prefix, useName);
}

//------------------------------------------------------------------------------
// Resources
//------------------------------------------------------------------------------
UQObject* RunUncertaintyAnalysis::FindResource(const std::string &name, const std::string &type,
                                                const std::string &usedAs)
{
   GmatBase *object = FindObject(name);
   if (object == NULL)
      throw CommandException("RunUncertaintyAnalysis: " + usedAs + " '" + name +
                             "' is not defined in the script");
   if (object->GetTypeName() != type)
      throw CommandException("RunUncertaintyAnalysis: " + usedAs + " '" + name + "' is a " +
                             object->GetTypeName() + ", not a " + type);
   return static_cast<UQObject*>(object);
}

void RunUncertaintyAnalysis::CheckResources(UQObject *analysis)
{
   std::string satName = analysis->GetString("Spacecraft");
   if (satName.empty())
      throw CommandException(analysisName + ".Spacecraft is not set");
   GmatBase *sat = FindObject(satName);
   if (sat == NULL || !sat->IsOfType(Gmat::SPACECRAFT))
      throw CommandException(analysisName + ".Spacecraft: '" + satName + "' is not a Spacecraft");
   std::string propName = analysis->GetString("Propagator");
   if (propName.empty())
      throw CommandException(analysisName + ".Propagator is not set");
   GmatBase *prop = FindObject(propName);
   if (prop == NULL || !prop->IsOfType(Gmat::PROP_SETUP))
      throw CommandException(analysisName + ".Propagator: '" + propName + "' is not a Propagator");
   if (!(analysis->GetReal("Duration") > 0.0))
      throw CommandException(analysisName + ".Duration must be positive");
   if (!analysis->GetString("InitialDispersion").empty())
      FindResource(analysis->GetString("InitialDispersion"), "StateDispersion",
                   analysisName + ".InitialDispersion");
   const StringArray &maneuvers = analysis->GetStrings("Maneuvers");
   for (const std::string &name : maneuvers)
   {
      UQObject *maneuver = FindResource(name, "UncertainManeuver", analysisName + ".Maneuvers");
      if (!maneuver->GetString("ErrorModel").empty())
         FindResource(maneuver->GetString("ErrorModel"), "ManeuverErrorModel", name + ".ErrorModel");
      if (!maneuver->GetString("Navigation").empty())
         FindResource(maneuver->GetString("Navigation"), "NavigationErrorModel", name + ".Navigation");
   }
   if (analysis->GetStrings("ConsiderParameters").size() != analysis->GetReals("ConsiderSigmas").size())
      throw CommandException(analysisName + ": ConsiderParameters and ConsiderSigmas need the same "
                             "number of entries");
   for (const std::string &method : analysis->GetStrings("Methods"))
      Method(method);
   Guidance(analysis->GetString("Guidance"));
}

bool RunUncertaintyAnalysis::Initialize()
{
   bool ok = GmatCommand::Initialize();
   UQObject *analysis = FindResource(analysisName, "UncertaintyAnalysis", "the analysis");
   CheckResources(analysis);
   return ok;
}

//------------------------------------------------------------------------------
// Paths
//------------------------------------------------------------------------------
std::string RunUncertaintyAnalysis::ScriptFile() const
{
   std::string name = Moderator::Instance()->GetMainScriptFileName();
   if (name.empty())
      throw CommandException("RunUncertaintyAnalysis needs the mission to be run from a script file");
   fs::path path(name);
   if (path.is_relative())
   {
      fs::path fromWorking = fs::path(FileManager::Instance()->GetGmatWorkingDirectory()) / path;
      path = fs::exists(fromWorking) ? fromWorking : fs::absolute(path);
   }
   return path.lexically_normal().generic_string();
}

std::string RunUncertaintyAnalysis::OutputFolder() const
{
   fs::path folder(FileManager::Instance()->GetAbsPathname(FileManager::OUTPUT_PATH));
   fs::create_directories(folder);
   return folder.generic_string();
}

std::string RunUncertaintyAnalysis::GmatRoot() const
{
   fs::path root(FileManager::Instance()->GetAbsPathname(FileManager::ROOT_PATH));
   return fs::absolute(root).lexically_normal().generic_string();
}

//------------------------------------------------------------------------------
// Configuration
//------------------------------------------------------------------------------
std::string RunUncertaintyAnalysis::BuildConfiguration(UQObject *analysis, GmatBase *spacecraft,
                                                       const std::string &scriptFile,
                                                       const std::string &gmatRoot,
                                                       const std::string &reportFile)
{
   SpaceObject *sat = dynamic_cast<SpaceObject*>(spacecraft);
   if (sat == NULL)
      throw CommandException(analysisName + ".Spacecraft is not a Spacecraft");

   std::vector<Real> state;
   for (const char *field : STATE_FIELDS)
      state.push_back(spacecraft->GetRealParameter(field));

   std::ostringstream toml;
   toml << "# Written by RunUncertaintyAnalysis for " << analysisName << ".\n"
        << "# Times are seconds after the epoch below (the moment the command executed).\n\n";

   toml << "[dynamics]\n"
        << "type = \"gmat\"\n"
        << "script = " << Quoted(scriptFile) << "\n"
        << "spacecraft = " << Quoted(spacecraft->GetName()) << "\n"
        << "propagator = " << Quoted(analysis->GetString("Propagator")) << "\n"
        << "gmat_root = " << Quoted(gmatRoot) << "\n\n";

   toml << "[mission]\n"
        << "epoch_a1mjd = " << Number(sat->GetEpoch()) << "\n"
        << "initial_state = " << Numbers(state) << "\n"
        << "final_time = " << Number(analysis->GetReal("Duration")) << "\n";
   if (!analysis->GetStrings("FinalTarget").empty())
      toml << "final_target = " << Target(analysis->GetStrings("FinalTarget"),
                                           analysis->GetString("FinalTargetBody")) << "\n";
   toml << "\n";

   if (!analysis->GetString("InitialDispersion").empty())
   {
      UQObject *dispersion = FindResource(analysis->GetString("InitialDispersion"), "StateDispersion",
                                          analysisName + ".InitialDispersion");
      toml << CovarianceSection(dispersion, "[mission.initial_dispersion]");
   }

   StringArray errorModels, navigationModels;
   const StringArray &maneuvers = analysis->GetStrings("Maneuvers");
   for (const std::string &name : maneuvers)
   {
      UQObject *maneuver = FindResource(name, "UncertainManeuver", analysisName + ".Maneuvers");
      std::string error = maneuver->GetString("ErrorModel");
      std::string navigation = maneuver->GetString("Navigation");
      if (!error.empty() && std::find(errorModels.begin(), errorModels.end(), error) == errorModels.end())
         errorModels.push_back(error);
      if (!navigation.empty() &&
          std::find(navigationModels.begin(), navigationModels.end(), navigation) == navigationModels.end())
         navigationModels.push_back(navigation);
   }
   for (const std::string &name : errorModels)
   {
      UQObject *model = FindResource(name, "ManeuverErrorModel", "error model");
      toml << "[error_models." << name << "]\n"
           << "fixed_magnitude_sigma = " << Number(model->GetReal("FixedMagnitudeSigma")) << "\n"
           << "proportional_magnitude_sigma = " << Number(model->GetReal("ProportionalMagnitudeSigma")) << "\n"
           << "fixed_pointing_sigma = " << Number(model->GetReal("FixedPointingSigma")) << "\n"
           << "proportional_pointing_sigma = " << Number(model->GetReal("ProportionalPointingSigma")) << "\n\n";
   }
   for (const std::string &name : navigationModels)
      toml << CovarianceSection(FindResource(name, "NavigationErrorModel", "navigation model"),
                         "[navigation." + name + "]");

   for (const std::string &name : maneuvers)
   {
      UQObject *maneuver = FindResource(name, "UncertainManeuver", analysisName + ".Maneuvers");
      toml << "[[maneuver]]\n"
           << "name = " << Quoted(name) << "\n"
           << "time = " << Number(maneuver->GetReal("Time")) << "\n";
      const std::vector<Real> &dv = maneuver->GetReals("DeltaV");
      if (!dv.empty())
      {
         if (dv.size() != 3)
            throw CommandException(name + ".DeltaV needs three values (km/s)");
         toml << "dv = " << Numbers(dv) << "\n";
      }
      if (!maneuver->GetString("ErrorModel").empty())
         toml << "error_model = " << Quoted(maneuver->GetString("ErrorModel")) << "\n";
      if (!maneuver->GetStrings("Target").empty())
      {
         toml << "target = " << Target(maneuver->GetStrings("Target"), maneuver->GetString("TargetBody")) << "\n"
              << "target_time = " << Number(maneuver->GetReal("TargetTime")) << "\n";
         if (!maneuver->GetString("Navigation").empty())
            toml << "navigation = " << Quoted(maneuver->GetString("Navigation")) << "\n";
      }
      else if (!maneuver->GetString("Navigation").empty())
         throw CommandException(name + ".Navigation is only used by a correction (set " + name + ".Target)");
      toml << "\n";
   }

   const StringArray &considerNames = analysis->GetStrings("ConsiderParameters");
   const std::vector<Real> &considerSigmas = analysis->GetReals("ConsiderSigmas");
   for (size_t i = 0; i < considerNames.size(); ++i)
      toml << "[[consider]]\n"
           << "name = " << Quoted(considerNames[i]) << "\n"
           << "sigma = " << Number(considerSigmas[i]) << "\n\n";

   StringArray methods;
   for (const std::string &method : analysis->GetStrings("Methods"))
      methods.push_back(Method(method));
   if (methods.empty())
      methods = {"lincov", "sigma_point", "monte_carlo"};
   toml << "[analysis]\n"
        << "methods = " << QuotedList(methods) << "\n"
        << "samples = " << analysis->GetInteger("Samples") << "\n"
        << "seed = " << analysis->GetInteger("Seed") << "\n"
        << "workers = " << analysis->GetInteger("Workers") << "\n"
        << "guidance = " << Quoted(Guidance(analysis->GetString("Guidance"))) << "\n"
        << "dv_samples = " << analysis->GetInteger("DeltaVSamples") << "\n"
        << "threshold = " << Number(analysis->GetReal("NonlinearityThreshold")) << "\n"
        << "probability = " << Number(analysis->GetReal("Probability")) << "\n"
        << "confidence = " << Number(analysis->GetReal("Confidence")) << "\n"
        << "report = " << Quoted(reportFile) << "\n";
   return toml.str();
}

//------------------------------------------------------------------------------
// Results
//------------------------------------------------------------------------------
void RunUncertaintyAnalysis::ReadResults(UQObject *analysis, const std::string &file)
{
   static const std::map<std::string, std::string> keys = {
      {"delta_v_99", "DeltaV99"},
      {"delta_v_mean", "DeltaVMean"},
      {"monte_carlo_delta_v_99", "MonteCarloDeltaV99"},
      {"final_position_sigma", "FinalPositionSigma"},
      {"final_velocity_sigma", "FinalVelocitySigma"},
      {"nonlinearity_index", "NonlinearityIndex"},
      {"linear_trusted", "LinearTrusted"},
      {"engines_agree", "EnginesAgree"},
   };
   for (const std::string &name : UncertaintyAnalysis::ResultNames())
      analysis->SetResult(name, -1.0);

   std::ifstream in(file.c_str());
   if (!in)
      throw CommandException("RunUncertaintyAnalysis: the GMAT-UQ engine wrote no results to " + file);
   std::string line;
   while (std::getline(in, line))
   {
      std::string::size_type equals = line.find('=');
      if (equals == std::string::npos)
         continue;
      std::string key = line.substr(0, equals), value = line.substr(equals + 1);
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t\r") + 1);
      std::map<std::string, std::string>::const_iterator it = keys.find(key);
      if (it != keys.end())
         analysis->SetResult(it->second, std::strtod(value.c_str(), NULL));
   }
}

//------------------------------------------------------------------------------
// Execution
//------------------------------------------------------------------------------
bool RunUncertaintyAnalysis::Execute()
{
   UQObject *analysis = FindResource(analysisName, "UncertaintyAnalysis", "the analysis");
   CheckResources(analysis);
   GmatBase *spacecraft = FindObject(analysis->GetString("Spacecraft"));

   std::string folder = OutputFolder();
   std::string base = folder + "/" + analysisName;
   std::string configFile = base + "_config.toml";
   std::string valuesFile = base + "_results.txt";
   std::string logFile = base + "_log.txt";
   std::string reportFile = analysis->GetString("ReportFile");
   if (reportFile.empty())
      reportFile = base + ".json";
   else
      reportFile = fs::absolute(fs::path(reportFile)).generic_string();

   {
      std::ofstream out(configFile.c_str());
      if (!out)
         throw CommandException("RunUncertaintyAnalysis: cannot write " + configFile);
      out << BuildConfiguration(analysis, spacecraft, ScriptFile(), GmatRoot(), reportFile);
   }
   std::remove(valuesFile.c_str());

   std::string command = Quote(analysis->GetString("PythonExecutable")) + " -m gmat_uq run " +
                         Quote(configFile) + " --quiet --values " + Quote(valuesFile) +
                         " > " + Quote(logFile) + " 2>&1";
   #ifdef _WIN32
      command = "\"" + command + "\"";   // cmd.exe strips one pair of outer quotes
   #endif

   MessageInterface::ShowMessage("RunUncertaintyAnalysis %s: running the GMAT-UQ engines\n",
                                 analysisName.c_str());
   int status = std::system(command.c_str());
   #ifndef _WIN32
      if (status != -1 && WIFEXITED(status))
         status = WEXITSTATUS(status);
   #endif

   // Exit status 2 means: the analysis ran, but the linear results should not be trusted.
   if (status != 0 && status != 2)
      throw CommandException("RunUncertaintyAnalysis " + analysisName + " failed (status " +
                             std::to_string(status) + "). Last lines of " + logFile + ":\n" +
                             Tail(logFile, 15));

   ReadResults(analysis, valuesFile);
   MessageInterface::ShowMessage(
      "RunUncertaintyAnalysis %s: delta-V 99 %.6f km/s, final position sigma %.6f km, "
      "linear results %s. Report: %s\n",
      analysisName.c_str(), analysis->GetResult("DeltaV99"), analysis->GetResult("FinalPositionSigma"),
      analysis->GetResult("LinearTrusted") == 1.0 ? "trusted" :
         (analysis->GetResult("LinearTrusted") == 0.0 ? "NOT trusted" : "not checked"),
      reportFile.c_str());

   BuildCommandSummary(true);
   return true;
}
