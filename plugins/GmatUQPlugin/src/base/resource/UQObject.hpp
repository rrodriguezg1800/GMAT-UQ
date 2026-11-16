//------------------------------------------------------------------------------
//                              UQObject
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------
/**
 * Base class of the GMAT-UQ script resources.
 *
 * Each resource is a set of named fields described by a static table.  This
 * class implements GMAT's parameter interface once for every resource, so a
 * resource only declares its fields.
 */
//------------------------------------------------------------------------------

#ifndef UQObject_hpp
#define UQObject_hpp

#include "GmatUQDefs.hpp"
#include "GmatBase.hpp"
#include "Rvector.hpp"

#include <map>
#include <string>
#include <vector>

/// Description of one field of a GMAT-UQ resource
struct GMATUQ_API UQField
{
   std::string          name;
   Gmat::ParameterType  type;       ///< REAL, INTEGER, BOOLEAN, STRING, STRINGARRAY or RVECTOR
   bool                 readOnly;   ///< true for results written by RunUncertaintyAnalysis
   Real                 realDefault;
   std::string          stringDefault;
};

class GMATUQ_API UQObject : public GmatBase
{
public:
   UQObject(const std::string &typeStr, const std::string &name,
            const std::vector<UQField> &fieldTable);
   virtual ~UQObject();
   UQObject(const UQObject &other);
   UQObject& operator=(const UQObject &other);

   virtual void         Copy(const GmatBase *orig);
   virtual bool         RenameRefObject(const UnsignedInt type,
                                        const std::string &oldName,
                                        const std::string &newName);

   // Convenience accessors used by the command
   Real                 GetReal(const std::string &field) const;
   Integer              GetInteger(const std::string &field) const;
   bool                 GetBoolean(const std::string &field) const;
   std::string          GetString(const std::string &field) const;
   const StringArray&   GetStrings(const std::string &field) const;
   const std::vector<Real>&
                        GetReals(const std::string &field) const;
   bool                 IsSet(const std::string &field) const;
   void                 SetResult(const std::string &field, Real value);

   // Parameter interface.  The using-declarations keep GmatBase's other
   // overloads visible next to the ones redefined here.
   using GmatBase::GetRealParameter;
   using GmatBase::SetRealParameter;
   using GmatBase::GetIntegerParameter;
   using GmatBase::SetIntegerParameter;
   using GmatBase::GetBooleanParameter;
   using GmatBase::SetBooleanParameter;
   using GmatBase::GetStringParameter;
   using GmatBase::SetStringParameter;
   using GmatBase::GetStringArrayParameter;
   using GmatBase::GetRvectorParameter;
   using GmatBase::SetRvectorParameter;
   using GmatBase::IsParameterReadOnly;

   virtual std::string  GetParameterText(const Integer id) const;
   virtual Integer      GetParameterID(const std::string &str) const;
   virtual Gmat::ParameterType
                        GetParameterType(const Integer id) const;
   virtual std::string  GetParameterTypeString(const Integer id) const;
   virtual bool         IsParameterReadOnly(const Integer id) const;
   virtual bool         IsParameterReadOnly(const std::string &label) const;
   virtual bool         IsParameterCommandModeSettable(const Integer id) const;

   virtual Real         GetRealParameter(const Integer id) const;
   virtual Real         SetRealParameter(const Integer id, const Real value);
   virtual Real         GetRealParameter(const Integer id,
                                         const Integer index) const;
   virtual Real         SetRealParameter(const Integer id, const Real value,
                                         const Integer index);
   virtual Real         GetRealParameter(const std::string &label) const;
   virtual Real         SetRealParameter(const std::string &label,
                                         const Real value);
   virtual Real         GetRealParameter(const std::string &label,
                                         const Integer index) const;
   virtual Real         SetRealParameter(const std::string &label,
                                         const Real value,
                                         const Integer index);

   virtual Integer      GetIntegerParameter(const Integer id) const;
   virtual Integer      SetIntegerParameter(const Integer id,
                                            const Integer value);
   virtual Integer      GetIntegerParameter(const std::string &label) const;
   virtual Integer      SetIntegerParameter(const std::string &label,
                                            const Integer value);

   virtual bool         GetBooleanParameter(const Integer id) const;
   virtual bool         SetBooleanParameter(const Integer id,
                                            const bool value);
   virtual bool         GetBooleanParameter(const std::string &label) const;
   virtual bool         SetBooleanParameter(const std::string &label,
                                            const bool value);

   virtual std::string  GetStringParameter(const Integer id) const;
   virtual bool         SetStringParameter(const Integer id,
                                           const std::string &value);
   virtual std::string  GetStringParameter(const Integer id,
                                           const Integer index) const;
   virtual bool         SetStringParameter(const Integer id,
                                           const std::string &value,
                                           const Integer index);
   virtual std::string  GetStringParameter(const std::string &label) const;
   virtual bool         SetStringParameter(const std::string &label,
                                           const std::string &value);
   virtual std::string  GetStringParameter(const std::string &label,
                                           const Integer index) const;
   virtual bool         SetStringParameter(const std::string &label,
                                           const std::string &value,
                                           const Integer index);
   virtual const StringArray&
                        GetStringArrayParameter(const Integer id) const;
   virtual const StringArray&
                        GetStringArrayParameter(const std::string &label) const;

   virtual const Rvector&
                        GetRvectorParameter(const Integer id) const;
   virtual const Rvector&
                        SetRvectorParameter(const Integer id,
                                            const Rvector &value);
   virtual const Rvector&
                        GetRvectorParameter(const std::string &label) const;
   virtual const Rvector&
                        SetRvectorParameter(const std::string &label,
                                            const Rvector &value);

protected:
   /// Field descriptions, shared by every object of one resource type
   const std::vector<UQField>          *fields;

   std::map<Integer, Real>             reals;
   std::map<Integer, Integer>          integers;
   std::map<Integer, bool>             booleans;
   std::map<Integer, std::string>      strings;
   std::map<Integer, StringArray>      stringArrays;
   std::map<Integer, std::vector<Real> > realArrays;
   /// Fields assigned in the script (results and defaults are not)
   std::map<Integer, bool>             assigned;
   /// Storage for the Rvector returned by GetRvectorParameter
   mutable Rvector                     rvectorBuffer;

   const UQField&       Field(const Integer id) const;
   Integer              CheckedID(const std::string &label) const;
   void                 RequireType(const Integer id,
                                    Gmat::ParameterType type) const;
   void                 RequireWritable(const Integer id) const;
};

#endif // UQObject_hpp
