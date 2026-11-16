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
#include "UQResources.hpp"

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
