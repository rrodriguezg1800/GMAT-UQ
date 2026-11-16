//------------------------------------------------------------------------------
//                           GmatPluginFunctions
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------

#include "GmatPluginFunctions.hpp"
#include "GmatUQFactory.hpp"
#include "MessageInterface.hpp"

extern "C"
{
   Integer GetFactoryCount()
   {
      return 3;
   }

   Factory* GetFactoryPointer(Integer index)
   {
      switch (index)
      {
         case 0:  return new UQResourceFactory;
         case 1:  return new UQCommandFactory;
         case 2:  return new UQParameterFactory;
         default: return NULL;
      }
   }

   void SetMessageReceiver(MessageReceiver *mr)
   {
      MessageInterface::SetMessageReceiver(mr);
   }
};
