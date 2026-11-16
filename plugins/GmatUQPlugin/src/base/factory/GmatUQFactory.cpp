//------------------------------------------------------------------------------
//                              GmatUQFactory
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------

#include "GmatUQFactory.hpp"
#include "GmatType.hpp"
#include "RunUncertaintyAnalysis.hpp"
#include "UQResultParameter.hpp"
#include "UQResources.hpp"

#include <algorithm>

namespace
{
   void AddResourceTypes(StringArray &creatables)
   {
      if (creatables.empty())
      {
         creatables.push_back("ManeuverErrorModel");
         creatables.push_back("StateDispersion");
         creatables.push_back("NavigationErrorModel");
         creatables.push_back("UncertainManeuver");
         creatables.push_back("UncertaintyAnalysis");
      }
   }

   void AddCommandTypes(StringArray &creatables)
   {
      if (creatables.empty())
         creatables.push_back("RunUncertaintyAnalysis");
   }
}

//------------------------------------------------------------------------------
// UQResourceFactory
//------------------------------------------------------------------------------
UQResourceFactory::UQResourceFactory() :
   Factory(GmatType::RegisterType("UncertaintyResource"))
{
   AddResourceTypes(creatables);
}

UQResourceFactory::~UQResourceFactory()
{
}

UQResourceFactory::UQResourceFactory(const UQResourceFactory &other) :
   Factory(other)
{
   AddResourceTypes(creatables);
}

UQResourceFactory& UQResourceFactory::operator=(const UQResourceFactory &other)
{
   if (this != &other)
   {
      Factory::operator=(other);
      AddResourceTypes(creatables);
   }
   return *this;
}

GmatBase* UQResourceFactory::CreateObject(const std::string &ofType, const std::string &withName)
{
   if (ofType == "ManeuverErrorModel")    return new ManeuverErrorModel(withName);
   if (ofType == "StateDispersion")       return new StateDispersion(withName);
   if (ofType == "NavigationErrorModel")  return new NavigationErrorModel(withName);
   if (ofType == "UncertainManeuver")     return new UncertainManeuver(withName);
   if (ofType == "UncertaintyAnalysis")   return new UncertaintyAnalysis(withName);
   return NULL;
}

//------------------------------------------------------------------------------
// UQParameterFactory: UQ.DeltaV99 and the other results
//------------------------------------------------------------------------------
UQParameterFactory::UQParameterFactory() :
   Factory                 (Gmat::PARAMETER),
   registrationComplete    (false)
{
   if (creatables.empty())
      creatables = UQResultParameter::ResultTypes();
}

UQParameterFactory::~UQParameterFactory()
{
}

UQParameterFactory::UQParameterFactory(const UQParameterFactory &other) :
   Factory                 (other),
   registrationComplete    (false)
{
   if (creatables.empty())
      creatables = UQResultParameter::ResultTypes();
}

UQParameterFactory& UQParameterFactory::operator=(const UQParameterFactory &other)
{
   if (this != &other)
   {
      Factory::operator=(other);
      if (creatables.empty())
         creatables = UQResultParameter::ResultTypes();
   }
   return *this;
}

GmatBase* UQParameterFactory::CreateObject(const std::string &ofType, const std::string &withName)
{
   return CreateParameter(ofType, withName);
}

Parameter* UQParameterFactory::CreateParameter(const std::string &ofType, const std::string &withName)
{
   const StringArray &types = UQResultParameter::ResultTypes();
   if (std::find(types.begin(), types.end(), ofType) == types.end())
      return NULL;
   return new UQResultParameter(withName, ofType);
}

StringArray UQParameterFactory::GetListOfCreatableObjects(const std::string &qualifier)
{
   // Creating one instance of each type registers it with GMAT's parameter
   // database, which is how the script interpreter learns that UQ.DeltaV99 is a
   // parameter of an UncertaintyResource.
   if (!registrationComplete)
   {
      for (const std::string &type : UQResultParameter::ResultTypes())
         delete CreateParameter(type, "DefaultUQ." + type);
      registrationComplete = true;
   }
   return Factory::GetListOfCreatableObjects(qualifier);
}

//------------------------------------------------------------------------------
// UQCommandFactory
//------------------------------------------------------------------------------
UQCommandFactory::UQCommandFactory() :
   Factory(Gmat::COMMAND)
{
   AddCommandTypes(creatables);
}

UQCommandFactory::~UQCommandFactory()
{
}

UQCommandFactory::UQCommandFactory(const UQCommandFactory &other) :
   Factory(other)
{
   AddCommandTypes(creatables);
}

UQCommandFactory& UQCommandFactory::operator=(const UQCommandFactory &other)
{
   if (this != &other)
   {
      Factory::operator=(other);
      AddCommandTypes(creatables);
   }
   return *this;
}

GmatCommand* UQCommandFactory::CreateCommand(const std::string &ofType, const std::string &withName)
{
   if (ofType == "RunUncertaintyAnalysis")
      return new RunUncertaintyAnalysis();
   return NULL;
}
