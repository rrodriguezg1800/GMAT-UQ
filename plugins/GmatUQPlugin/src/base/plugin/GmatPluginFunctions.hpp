//------------------------------------------------------------------------------
//                           GmatPluginFunctions
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------
/**
 * Entry points GMAT calls when it loads the plugin library.
 */
//------------------------------------------------------------------------------

#ifndef GmatPluginFunctions_hpp
#define GmatPluginFunctions_hpp

#include "GmatUQDefs.hpp"
#include "Factory.hpp"

class MessageReceiver;

extern "C"
{
   Integer  GMATUQ_API GetFactoryCount();
   Factory  GMATUQ_API *GetFactoryPointer(Integer index);
   void     GMATUQ_API SetMessageReceiver(MessageReceiver *mr);
};

#endif // GmatPluginFunctions_hpp
