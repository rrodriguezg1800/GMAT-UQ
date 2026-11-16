//------------------------------------------------------------------------------
//                              GmatUQFactory
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------
/**
 * Factories of the GMAT-UQ resources and of the RunUncertaintyAnalysis command.
 */
//------------------------------------------------------------------------------

#ifndef GmatUQFactory_hpp
#define GmatUQFactory_hpp

#include "GmatUQDefs.hpp"
#include "Factory.hpp"

class GMATUQ_API UQResourceFactory : public Factory
{
public:
   UQResourceFactory();
   virtual ~UQResourceFactory();
   UQResourceFactory(const UQResourceFactory &other);
   UQResourceFactory& operator=(const UQResourceFactory &other);

   virtual GmatBase*    CreateObject(const std::string &ofType,
                                     const std::string &withName = "");
};

class GMATUQ_API UQParameterFactory : public Factory
{
public:
   UQParameterFactory();
   virtual ~UQParameterFactory();
   UQParameterFactory(const UQParameterFactory &other);
   UQParameterFactory& operator=(const UQParameterFactory &other);

   virtual GmatBase*    CreateObject(const std::string &ofType,
                                     const std::string &withName = "");
   virtual Parameter*   CreateParameter(const std::string &ofType,
                                        const std::string &withName = "");
   virtual StringArray  GetListOfCreatableObjects(const std::string &qualifier = "");

private:
   bool                 registrationComplete;
};

class GMATUQ_API UQCommandFactory : public Factory
{
public:
   UQCommandFactory();
   virtual ~UQCommandFactory();
   UQCommandFactory(const UQCommandFactory &other);
   UQCommandFactory& operator=(const UQCommandFactory &other);

   virtual GmatCommand* CreateCommand(const std::string &ofType,
                                      const std::string &withName = "");
};

#endif // GmatUQFactory_hpp
