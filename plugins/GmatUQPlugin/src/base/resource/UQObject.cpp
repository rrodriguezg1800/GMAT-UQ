//------------------------------------------------------------------------------
//                              UQObject
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------

#include "UQObject.hpp"
#include "GmatBaseException.hpp"
#include "GmatType.hpp"

#include <algorithm>

namespace
{
   /// All GMAT-UQ resources share one registered object type
   UnsignedInt UQType()
   {
      return GmatType::RegisterType("UncertaintyResource");
   }
}

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------
UQObject::UQObject(const std::string &typeStr, const std::string &name,
                   const std::vector<UQField> &fieldTable) :
   GmatBase       (UQType(), typeStr, name),
   fields         (&fieldTable)
{
   objectTypes.push_back(UQType());
   objectTypeNames.push_back("UncertaintyResource");
   objectTypeNames.push_back(typeStr);
   parameterCount = GmatBaseParamCount + (Integer)fields->size();

   for (Integer i = 0; i < (Integer)fields->size(); ++i)
   {
      const UQField &f = (*fields)[i];
      Integer id = GmatBaseParamCount + i;
      switch (f.type)
      {
         case Gmat::REAL_TYPE:        reals[id] = f.realDefault;                 break;
         case Gmat::INTEGER_TYPE:     integers[id] = (Integer)f.realDefault;     break;
         case Gmat::BOOLEAN_TYPE:     booleans[id] = (f.realDefault != 0.0);     break;
         case Gmat::STRING_TYPE:      strings[id] = f.stringDefault;             break;
         case Gmat::STRINGARRAY_TYPE: stringArrays[id] = StringArray();          break;
         case Gmat::RVECTOR_TYPE:     realArrays[id] = std::vector<Real>();      break;
         default:
            throw GmatBaseException("UQObject: unsupported field type for " + f.name);
      }
   }
}

UQObject::~UQObject()
{
}

UQObject::UQObject(const UQObject &other) :
   GmatBase       (other),
   fields         (other.fields),
   reals          (other.reals),
   integers       (other.integers),
   booleans       (other.booleans),
   strings        (other.strings),
   stringArrays   (other.stringArrays),
   realArrays     (other.realArrays),
   assigned       (other.assigned)
{
}

UQObject& UQObject::operator=(const UQObject &other)
{
   if (this != &other)
   {
      GmatBase::operator=(other);
      fields       = other.fields;
      reals        = other.reals;
      integers     = other.integers;
      booleans     = other.booleans;
      strings      = other.strings;
      stringArrays = other.stringArrays;
      realArrays   = other.realArrays;
      assigned     = other.assigned;
   }
   return *this;
}

void UQObject::Copy(const GmatBase *orig)
{
   const UQObject *source = dynamic_cast<const UQObject*>(orig);
   if (source == NULL || source->GetTypeName() != GetTypeName())
      throw GmatBaseException("Cannot copy a " + orig->GetTypeName() +
                              " into the " + GetTypeName() + " " + instanceName);
   operator=(*source);
}

bool UQObject::RenameRefObject(const UnsignedInt type, const std::string &oldName,
                               const std::string &newName)
{
   // References to other resources are stored by name in string fields.
   for (std::map<Integer, std::string>::iterator it = strings.begin(); it != strings.end(); ++it)
      if (it->second == oldName)
         it->second = newName;
   for (std::map<Integer, StringArray>::iterator it = stringArrays.begin(); it != stringArrays.end(); ++it)
      std::replace(it->second.begin(), it->second.end(), oldName, newName);
   return true;
}

//------------------------------------------------------------------------------
// Helpers
//------------------------------------------------------------------------------
const UQField& UQObject::Field(const Integer id) const
{
   return (*fields)[id - GmatBaseParamCount];
}

Integer UQObject::CheckedID(const std::string &label) const
{
   Integer id = GetParameterID(label);
   if (id < GmatBaseParamCount)
      throw GmatBaseException(GetTypeName() + " has no field named " + label);
   return id;
}

void UQObject::RequireType(const Integer id, Gmat::ParameterType type) const
{
   if (id < GmatBaseParamCount || id >= parameterCount)
      throw GmatBaseException("Invalid field id on " + GetTypeName() + " " + instanceName);
   if (Field(id).type != type)
      throw GmatBaseException("The field " + Field(id).name + " of " + GetTypeName() +
                              " " + instanceName + " is a " +
                              PARAM_TYPE_STRING[Field(id).type] + ", not a " +
                              PARAM_TYPE_STRING[type]);
}

void UQObject::RequireWritable(const Integer id) const
{
   if (Field(id).readOnly)
      throw GmatBaseException("The field " + Field(id).name + " of " + GetTypeName() +
                              " " + instanceName + " is a result of RunUncertaintyAnalysis "
                              "and cannot be set");
}

Real UQObject::GetReal(const std::string &field) const
{
   return GetRealParameter(CheckedID(field));
}

Integer UQObject::GetInteger(const std::string &field) const
{
   return GetIntegerParameter(CheckedID(field));
}

bool UQObject::GetBoolean(const std::string &field) const
{
   return GetBooleanParameter(CheckedID(field));
}

std::string UQObject::GetString(const std::string &field) const
{
   return GetStringParameter(CheckedID(field));
}

const StringArray& UQObject::GetStrings(const std::string &field) const
{
   return GetStringArrayParameter(CheckedID(field));
}

const std::vector<Real>& UQObject::GetReals(const std::string &field) const
{
   Integer id = CheckedID(field);
   RequireType(id, Gmat::RVECTOR_TYPE);
   return realArrays.find(id)->second;
}

bool UQObject::IsSet(const std::string &field) const
{
   std::map<Integer, bool>::const_iterator it = assigned.find(CheckedID(field));
   return it != assigned.end() && it->second;
}

void UQObject::SetResult(const std::string &field, Real value)
{
   Integer id = CheckedID(field);
   RequireType(id, Gmat::REAL_TYPE);
   reals[id] = value;
}

//------------------------------------------------------------------------------
// Parameter descriptions
//------------------------------------------------------------------------------
std::string UQObject::GetParameterText(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
      return Field(id).name;
   return GmatBase::GetParameterText(id);
}

Integer UQObject::GetParameterID(const std::string &str) const
{
   for (Integer i = 0; i < (Integer)fields->size(); ++i)
      if ((*fields)[i].name == str)
         return GmatBaseParamCount + i;
   return GmatBase::GetParameterID(str);
}

Gmat::ParameterType UQObject::GetParameterType(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
      return Field(id).type;
   return GmatBase::GetParameterType(id);
}

std::string UQObject::GetParameterTypeString(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
      return PARAM_TYPE_STRING[Field(id).type];
   return GmatBase::GetParameterTypeString(id);
}

bool UQObject::IsParameterReadOnly(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
      return Field(id).readOnly;   // results are not written back to scripts
   return GmatBase::IsParameterReadOnly(id);
}

bool UQObject::IsParameterReadOnly(const std::string &label) const
{
   return IsParameterReadOnly(GetParameterID(label));
}

bool UQObject::IsParameterCommandModeSettable(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
      return !Field(id).readOnly;
   return GmatBase::IsParameterCommandModeSettable(id);
}

//------------------------------------------------------------------------------
// Real fields and real vectors
//------------------------------------------------------------------------------
Real UQObject::GetRealParameter(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::REAL_TYPE);
      return reals.find(id)->second;
   }
   return GmatBase::GetRealParameter(id);
}

Real UQObject::SetRealParameter(const Integer id, const Real value)
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      if (Field(id).type == Gmat::INTEGER_TYPE)
      {
         SetIntegerParameter(id, (Integer)value);
         return value;
      }
      RequireType(id, Gmat::REAL_TYPE);
      RequireWritable(id);
      reals[id] = value;
      assigned[id] = true;
      return value;
   }
   return GmatBase::SetRealParameter(id, value);
}

Real UQObject::GetRealParameter(const Integer id, const Integer index) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::RVECTOR_TYPE);
      const std::vector<Real> &values = realArrays.find(id)->second;
      if (index < 0 || index >= (Integer)values.size())
         throw GmatBaseException("Index out of range for " + instanceName + "." + Field(id).name);
      return values[index];
   }
   return GmatBase::GetRealParameter(id, index);
}

Real UQObject::SetRealParameter(const Integer id, const Real value, const Integer index)
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::RVECTOR_TYPE);
      RequireWritable(id);
      std::vector<Real> &values = realArrays[id];
      // A new assignment of the whole vector starts again at index 0.
      if (index == 0)
         values.clear();
      if (index < 0 || index > (Integer)values.size())
         throw GmatBaseException("Index out of range for " + instanceName + "." + Field(id).name);
      if (index == (Integer)values.size())
         values.push_back(value);
      else
         values[index] = value;
      assigned[id] = true;
      return value;
   }
   return GmatBase::SetRealParameter(id, value, index);
}

Real UQObject::GetRealParameter(const std::string &label) const
{
   return GetRealParameter(GetParameterID(label));
}

Real UQObject::SetRealParameter(const std::string &label, const Real value)
{
   return SetRealParameter(GetParameterID(label), value);
}

Real UQObject::GetRealParameter(const std::string &label, const Integer index) const
{
   return GetRealParameter(GetParameterID(label), index);
}

Real UQObject::SetRealParameter(const std::string &label, const Real value, const Integer index)
{
   return SetRealParameter(GetParameterID(label), value, index);
}

const Rvector& UQObject::GetRvectorParameter(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::RVECTOR_TYPE);
      const std::vector<Real> &values = realArrays.find(id)->second;
      rvectorBuffer.SetSize((Integer)values.size());
      for (Integer i = 0; i < (Integer)values.size(); ++i)
         rvectorBuffer[i] = values[i];
      return rvectorBuffer;
   }
   return GmatBase::GetRvectorParameter(id);
}

const Rvector& UQObject::SetRvectorParameter(const Integer id, const Rvector &value)
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::RVECTOR_TYPE);
      RequireWritable(id);
      std::vector<Real> &values = realArrays[id];
      values.clear();
      for (Integer i = 0; i < value.GetSize(); ++i)
         values.push_back(value[i]);
      assigned[id] = true;
      return GetRvectorParameter(id);
   }
   return GmatBase::SetRvectorParameter(id, value);
}

const Rvector& UQObject::GetRvectorParameter(const std::string &label) const
{
   return GetRvectorParameter(GetParameterID(label));
}

const Rvector& UQObject::SetRvectorParameter(const std::string &label, const Rvector &value)
{
   return SetRvectorParameter(GetParameterID(label), value);
}

//------------------------------------------------------------------------------
// Integer and Boolean fields
//------------------------------------------------------------------------------
Integer UQObject::GetIntegerParameter(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::INTEGER_TYPE);
      return integers.find(id)->second;
   }
   return GmatBase::GetIntegerParameter(id);
}

Integer UQObject::SetIntegerParameter(const Integer id, const Integer value)
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::INTEGER_TYPE);
      RequireWritable(id);
      integers[id] = value;
      assigned[id] = true;
      return value;
   }
   return GmatBase::SetIntegerParameter(id, value);
}

Integer UQObject::GetIntegerParameter(const std::string &label) const
{
   return GetIntegerParameter(GetParameterID(label));
}

Integer UQObject::SetIntegerParameter(const std::string &label, const Integer value)
{
   return SetIntegerParameter(GetParameterID(label), value);
}

bool UQObject::GetBooleanParameter(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::BOOLEAN_TYPE);
      return booleans.find(id)->second;
   }
   return GmatBase::GetBooleanParameter(id);
}

bool UQObject::SetBooleanParameter(const Integer id, const bool value)
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::BOOLEAN_TYPE);
      RequireWritable(id);
      booleans[id] = value;
      assigned[id] = true;
      return value;
   }
   return GmatBase::SetBooleanParameter(id, value);
}

bool UQObject::GetBooleanParameter(const std::string &label) const
{
   return GetBooleanParameter(GetParameterID(label));
}

bool UQObject::SetBooleanParameter(const std::string &label, const bool value)
{
   return SetBooleanParameter(GetParameterID(label), value);
}

//------------------------------------------------------------------------------
// String fields and string arrays
//------------------------------------------------------------------------------
std::string UQObject::GetStringParameter(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::STRING_TYPE);
      return strings.find(id)->second;
   }
   return GmatBase::GetStringParameter(id);
}

bool UQObject::SetStringParameter(const Integer id, const std::string &value)
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireWritable(id);
      if (Field(id).type == Gmat::STRINGARRAY_TYPE)
      {
         // The script interpreter passes the items of {a, b} one at a time.
         StringArray &items = stringArrays[id];
         if (std::find(items.begin(), items.end(), value) != items.end())
            throw GmatBaseException("'" + value + "' appears twice in " +
                                    instanceName + "." + Field(id).name);
         items.push_back(value);
         assigned[id] = true;
         return true;
      }
      RequireType(id, Gmat::STRING_TYPE);
      strings[id] = value;
      assigned[id] = true;
      return true;
   }
   return GmatBase::SetStringParameter(id, value);
}

std::string UQObject::GetStringParameter(const Integer id, const Integer index) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::STRINGARRAY_TYPE);
      const StringArray &items = stringArrays.find(id)->second;
      if (index < 0 || index >= (Integer)items.size())
         throw GmatBaseException("Index out of range for " + instanceName + "." + Field(id).name);
      return items[index];
   }
   return GmatBase::GetStringParameter(id, index);
}

bool UQObject::SetStringParameter(const Integer id, const std::string &value, const Integer index)
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::STRINGARRAY_TYPE);
      RequireWritable(id);
      StringArray &items = stringArrays[id];
      if (index == 0)
         items.clear();
      if (index < 0 || index > (Integer)items.size())
         throw GmatBaseException("Index out of range for " + instanceName + "." + Field(id).name);
      if (index == (Integer)items.size())
         items.push_back(value);
      else
         items[index] = value;
      assigned[id] = true;
      return true;
   }
   return GmatBase::SetStringParameter(id, value, index);
}

std::string UQObject::GetStringParameter(const std::string &label) const
{
   return GetStringParameter(GetParameterID(label));
}

bool UQObject::SetStringParameter(const std::string &label, const std::string &value)
{
   return SetStringParameter(GetParameterID(label), value);
}

std::string UQObject::GetStringParameter(const std::string &label, const Integer index) const
{
   return GetStringParameter(GetParameterID(label), index);
}

bool UQObject::SetStringParameter(const std::string &label, const std::string &value,
                                  const Integer index)
{
   return SetStringParameter(GetParameterID(label), value, index);
}

const StringArray& UQObject::GetStringArrayParameter(const Integer id) const
{
   if (id >= GmatBaseParamCount && id < parameterCount)
   {
      RequireType(id, Gmat::STRINGARRAY_TYPE);
      return stringArrays.find(id)->second;
   }
   return GmatBase::GetStringArrayParameter(id);
}

const StringArray& UQObject::GetStringArrayParameter(const std::string &label) const
{
   return GetStringArrayParameter(GetParameterID(label));
}
