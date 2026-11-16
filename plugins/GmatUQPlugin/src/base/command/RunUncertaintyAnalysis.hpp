//------------------------------------------------------------------------------
//                           RunUncertaintyAnalysis
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------
/**
 * Mission-sequence command that runs an UncertaintyAnalysis:
 *
 *    RunUncertaintyAnalysis UQ;
 *
 * When it executes, the command takes the spacecraft's current epoch and
 * state, writes the analysis described by the GMAT-UQ resources to a
 * configuration file, runs the GMAT-UQ engines (a Python package that drives
 * GMAT through its API, with the force model and propagator of this script),
 * and stores the main results in the UncertaintyAnalysis resource, where the
 * rest of the mission sequence can use them (for example UQ.DeltaV99).
 */
//------------------------------------------------------------------------------

#ifndef RunUncertaintyAnalysis_hpp
#define RunUncertaintyAnalysis_hpp

#include "GmatUQDefs.hpp"
#include "GmatCommand.hpp"

class UQObject;

class GMATUQ_API RunUncertaintyAnalysis : public GmatCommand
{
public:
   RunUncertaintyAnalysis();
   virtual ~RunUncertaintyAnalysis();
   RunUncertaintyAnalysis(const RunUncertaintyAnalysis &other);
   RunUncertaintyAnalysis& operator=(const RunUncertaintyAnalysis &other);

   virtual GmatBase*    Clone() const;
   virtual bool         InterpretAction();
   virtual bool         Initialize();
   virtual bool         Execute();

   virtual bool         RenameRefObject(const UnsignedInt type,
                                        const std::string &oldName,
                                        const std::string &newName);
   virtual const std::string&
                        GetGeneratingString(Gmat::WriteMode mode = Gmat::SCRIPTING,
                                            const std::string &prefix = "",
                                            const std::string &useName = "");

   /// Builds the GMAT-UQ configuration text (TOML)
   std::string          BuildConfiguration(UQObject *analysis, GmatBase *spacecraft,
                                           const std::string &scriptFile,
                                           const std::string &gmatRoot,
                                           const std::string &reportFile);

   DEFAULT_TO_NO_CLONES

protected:
   std::string          analysisName;

   UQObject*            FindResource(const std::string &name,
                                     const std::string &type,
                                     const std::string &usedAs);
   void                 CheckResources(UQObject *analysis);
   std::string          ScriptFile() const;
   std::string          OutputFolder() const;
   std::string          GmatRoot() const;
   void                 ReadResults(UQObject *analysis, const std::string &file);
};

#endif // RunUncertaintyAnalysis_hpp
