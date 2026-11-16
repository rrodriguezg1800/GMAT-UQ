//------------------------------------------------------------------------------
//                              UQResources
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------
/**
 * The script resources added by GMAT-UQ:
 *
 *  - ManeuverErrorModel   Gates execution-error model of a maneuver
 *  - StateDispersion      dispersion of the initial state
 *  - NavigationErrorModel knowledge error at a correction maneuver
 *  - UncertainManeuver    an impulsive maneuver, open-loop or a correction
 *  - UncertaintyAnalysis  what to analyse, how, and the results
 *
 * Times are seconds after the moment RunUncertaintyAnalysis executes.
 * Lengths are km, velocities km/s, angles rad (pointing errors).
 */
//------------------------------------------------------------------------------

#ifndef UQResources_hpp
#define UQResources_hpp

#include "UQObject.hpp"

class GMATUQ_API ManeuverErrorModel : public UQObject
{
public:
   ManeuverErrorModel(const std::string &name);
   virtual GmatBase* Clone() const;
   static const std::vector<UQField>& Fields();
   DEFAULT_TO_NO_CLONES
};

class GMATUQ_API StateDispersion : public UQObject
{
public:
   StateDispersion(const std::string &name);
   virtual GmatBase* Clone() const;
   static const std::vector<UQField>& Fields();
   DEFAULT_TO_NO_CLONES
protected:
   StateDispersion(const std::string &typeStr, const std::string &name);
};

class GMATUQ_API NavigationErrorModel : public StateDispersion
{
public:
   NavigationErrorModel(const std::string &name);
   virtual GmatBase* Clone() const;
   DEFAULT_TO_NO_CLONES
};

class GMATUQ_API UncertainManeuver : public UQObject
{
public:
   UncertainManeuver(const std::string &name);
   virtual GmatBase* Clone() const;
   static const std::vector<UQField>& Fields();
   DEFAULT_TO_NO_CLONES
};

class GMATUQ_API UncertaintyAnalysis : public UQObject
{
public:
   UncertaintyAnalysis(const std::string &name);
   virtual GmatBase* Clone() const;
   static const std::vector<UQField>& Fields();
   /// Names of the result fields, in the order the engine reports them
   static const StringArray& ResultNames();
   DEFAULT_TO_NO_CLONES
};

#endif // UQResources_hpp
