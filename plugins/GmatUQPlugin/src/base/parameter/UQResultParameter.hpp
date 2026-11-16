//------------------------------------------------------------------------------
//                           UQResultParameter
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------
/**
 * GMAT parameters that expose the results of an UncertaintyAnalysis, such as
 * UQ.DeltaV99.  Like Sat.X, they can be reported, plotted and used on the
 * right-hand side of assignments.  Each reads the runtime copy of the
 * analysis, so it shows the results of the last RunUncertaintyAnalysis.
 * A value of -1 means the result has not been computed.
 */
//------------------------------------------------------------------------------

#ifndef UQResultParameter_hpp
#define UQResultParameter_hpp

#include "GmatUQDefs.hpp"
#include "RealVar.hpp"
#include "RefData.hpp"

class UQObject;

/// Reference-object bookkeeping: the UncertaintyAnalysis that owns the parameter
class GMATUQ_API UQResultData : public RefData
{
public:
   UQResultData();
   virtual ~UQResultData();
   UQResultData(const UQResultData &other);
   UQResultData& operator=(const UQResultData &other);

   virtual bool         ValidateRefObjects(GmatBase *param);
   virtual const std::string* GetValidObjectList() const;

protected:
   UQObject             *mAnalysis;

   virtual void         InitializeRefObjects();
   virtual bool         IsValidObjectType(UnsignedInt type);
   virtual bool         AddRefObject(const UnsignedInt type,
                                     const std::string &name, GmatBase *obj = NULL,
                                     bool replaceName = false);
   UQObject*            Analysis();
};

class GMATUQ_API UQResultParameter : public RealVar, public UQResultData
{
public:
   UQResultParameter(const std::string &name = "", const std::string &typeStr = "DeltaV99",
                     GmatBase *obj = NULL);
   virtual ~UQResultParameter();
   UQResultParameter(const UQResultParameter &other);
   UQResultParameter& operator=(const UQResultParameter &other);

   /// Result names, which are also the parameter type names
   static const StringArray& ResultTypes();
   static std::string   Description(const std::string &type);
   static std::string   Unit(const std::string &type);

   virtual GmatBase*    Clone() const;
   virtual bool         Evaluate();
   virtual Real         EvaluateReal();

   virtual Integer      GetNumRefObjects() const;
   virtual bool         AddRefObject(GmatBase *obj, bool replaceName = false);
   virtual bool         Validate();
   virtual bool         Initialize();
   virtual bool         RenameRefObject(const UnsignedInt type,
                                        const std::string &oldName,
                                        const std::string &newName);
   virtual std::string  GetRefObjectName(const UnsignedInt type) const;
   virtual const StringArray&
                        GetRefObjectNameArray(const UnsignedInt type);
   virtual bool         SetRefObjectName(const UnsignedInt type,
                                         const std::string &name);
   virtual GmatBase*    GetRefObject(const UnsignedInt type,
                                     const std::string &name);
   virtual bool         SetRefObject(GmatBase *obj, const UnsignedInt type,
                                     const std::string &name = "");
   virtual const std::string GetExternalCloneName(Integer whichOne);
   virtual void         SetExternalClone(GmatBase *clone);
};

#endif // UQResultParameter_hpp
