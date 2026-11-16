//------------------------------------------------------------------------------
//                           UQResultParameter
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------

#include "UQResultParameter.hpp"
#include "UQObject.hpp"
#include "GmatType.hpp"
#include "ParameterException.hpp"
#include "StringUtil.hpp"

namespace
{
   UnsignedInt UQType()
   {
      return GmatType::RegisterType("UncertaintyResource");
   }

   // RefData compares types by name through a table of built-in types, which does
   // not cover plugin types, so this class compares type ids instead.
   const std::string VALID_OBJECT_TYPES[1] = {"UncertaintyResource"};
}

//------------------------------------------------------------------------------
// UQResultData
//------------------------------------------------------------------------------
UQResultData::UQResultData() :
   RefData     ("", "", UQType()),
   mAnalysis   (NULL)
{
}

UQResultData::~UQResultData()
{
}

UQResultData::UQResultData(const UQResultData &other) :
   RefData     (other),
   mAnalysis   (other.mAnalysis)
{
}

UQResultData& UQResultData::operator=(const UQResultData &other)
{
   if (this != &other)
   {
      RefData::operator=(other);
      mAnalysis = other.mAnalysis;
   }
   return *this;
}

bool UQResultData::ValidateRefObjects(GmatBase *param)
{
   return FindFirstObject(UQType()) != NULL;
}

const std::string* UQResultData::GetValidObjectList() const
{
   return VALID_OBJECT_TYPES;
}

void UQResultData::InitializeRefObjects()
{
   mAnalysis = dynamic_cast<UQObject*>(FindFirstObject(UQType()));
   if (mAnalysis == NULL)
      throw ParameterException("Cannot find the UncertaintyAnalysis of the parameter " +
                               mActualParamName);
}

bool UQResultData::IsValidObjectType(UnsignedInt type)
{
   return type == UQType();
}

bool UQResultData::AddRefObject(const UnsignedInt type, const std::string &name,
                                GmatBase *obj, bool replaceName)
{
   if (type == UQType())
      mAnalysis = dynamic_cast<UQObject*>(obj);
   return RefData::AddRefObject(type, name, obj, replaceName);
}

UQObject* UQResultData::Analysis()
{
   if (mAnalysis == NULL)
      mAnalysis = dynamic_cast<UQObject*>(FindFirstObject(UQType()));
   return mAnalysis;
}

//------------------------------------------------------------------------------
// UQResultParameter
//------------------------------------------------------------------------------
const StringArray& UQResultParameter::ResultTypes()
{
   static const StringArray types = {
      "DeltaV99", "DeltaVMean", "MonteCarloDeltaV99", "FinalPositionSigma",
      "FinalVelocitySigma", "NonlinearityIndex", "LinearTrusted", "EnginesAgree",
   };
   return types;
}

std::string UQResultParameter::Description(const std::string &type)
{
   if (type == "DeltaV99")            return "Delta-V 99th percentile, linear covariance";
   if (type == "DeltaVMean")          return "Mean total delta-V, linear covariance";
   if (type == "MonteCarloDeltaV99")  return "Delta-V 99th percentile, Monte Carlo";
   if (type == "FinalPositionSigma")  return "Final position dispersion, RSS of the 1-sigma axes";
   if (type == "FinalVelocitySigma")  return "Final velocity dispersion, RSS of the 1-sigma axes";
   if (type == "NonlinearityIndex")   return "Largest sigma-point nonlinearity index";
   if (type == "LinearTrusted")       return "1 if no arc is flagged by the sigma-point check, else 0";
   if (type == "EnginesAgree")        return "1 if linear covariance and Monte Carlo agree, else 0";
   return type;
}

std::string UQResultParameter::Unit(const std::string &type)
{
   if (type == "DeltaV99" || type == "DeltaVMean" || type == "MonteCarloDeltaV99" ||
       type == "FinalVelocitySigma")
      return "Km/s";
   if (type == "FinalPositionSigma")
      return "Km";
   return "";
}

UQResultParameter::UQResultParameter(const std::string &name, const std::string &typeStr,
                                     GmatBase *obj) :
   RealVar(name, "", typeStr, GmatParam::SYSTEM_PARAM, obj, Description(typeStr),
           Unit(typeStr), GmatParam::NO_DEP, UQType(), false, false, true, true)
{
   objectTypeNames.push_back("UQResultParameter");
   mNeedCoordSystem = false;
   AddRefObject(obj);
   std::string type, ownerName, dep;
   GmatStringUtil::ParseParameter(name, type, ownerName, dep);
   mOwnerName = ownerName;
   mDepObjectName = "";
   mNeedExternalClone = true;   // read the runtime copy of the analysis
}

UQResultParameter::~UQResultParameter()
{
}

UQResultParameter::UQResultParameter(const UQResultParameter &other) :
   RealVar        (other),
   UQResultData   (other)
{
}

UQResultParameter& UQResultParameter::operator=(const UQResultParameter &other)
{
   if (this != &other)
   {
      RealVar::operator=(other);
      UQResultData::operator=(other);
   }
   return *this;
}

GmatBase* UQResultParameter::Clone() const
{
   return new UQResultParameter(*this);
}

bool UQResultParameter::Evaluate()
{
   UQObject *analysis = Analysis();
   if (analysis == NULL)
      return false;
   mRealValue = analysis->GetResult(GetTypeName());
   return true;
}

Real UQResultParameter::EvaluateReal()
{
   Evaluate();
   return mRealValue;
}

Integer UQResultParameter::GetNumRefObjects() const
{
   return UQResultData::GetNumRefObjects();
}

bool UQResultParameter::AddRefObject(GmatBase *obj, bool replaceName)
{
   if (obj == NULL)
      return false;
   return UQResultData::AddRefObject(obj->GetType(), obj->GetName(), obj, replaceName);
}

bool UQResultParameter::Validate()
{
   return ValidateRefObjects(this);
}

bool UQResultParameter::Initialize()
{
   InitializeRefObjects();
   return true;
}

bool UQResultParameter::RenameRefObject(const UnsignedInt type, const std::string &oldName,
                                        const std::string &newName)
{
   return UQResultData::RenameRefObject(type, oldName, newName);
}

std::string UQResultParameter::GetRefObjectName(const UnsignedInt type) const
{
   return UQResultData::GetRefObjectName(type);
}

const StringArray& UQResultParameter::GetRefObjectNameArray(const UnsignedInt type)
{
   return UQResultData::GetRefObjectNameArray(type);
}

bool UQResultParameter::SetRefObjectName(const UnsignedInt type, const std::string &name)
{
   return UQResultData::SetRefObjectName(type, name);
}

GmatBase* UQResultParameter::GetRefObject(const UnsignedInt type, const std::string &name)
{
   return UQResultData::GetRefObject(type, name);
}

bool UQResultParameter::SetRefObject(GmatBase *obj, const UnsignedInt type,
                                     const std::string &name)
{
   if (type == UQType())
      mAnalysis = dynamic_cast<UQObject*>(obj);
   return UQResultData::SetRefObject(obj, type, name);
}

const std::string UQResultParameter::GetExternalCloneName(Integer whichOne)
{
   return mOwnerName;
}

void UQResultParameter::SetExternalClone(GmatBase *clone)
{
   if (clone == NULL)
      throw ParameterException("UQResultParameter: external clone pointer is NULL");
   UQResultData::AddRefObject(clone->GetType(), clone->GetName(), clone, true);
   mAnalysis = dynamic_cast<UQObject*>(clone);
}
