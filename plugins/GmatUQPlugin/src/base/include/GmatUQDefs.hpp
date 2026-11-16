//------------------------------------------------------------------------------
//                              GmatUQDefs
//------------------------------------------------------------------------------
// GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
//
// Copyright 2026 GMAT-UQ contributors
// Licensed under the Apache License, Version 2.0 (see License.txt in the
// repository root).
//------------------------------------------------------------------------------
/**
 * Export definitions for the GMAT-UQ plugin library.
 */
//------------------------------------------------------------------------------

#ifndef GmatUQDefs_hpp
#define GmatUQDefs_hpp

#include "gmatdefs.hpp"

#ifdef _WIN32
   #ifdef _DYNAMICLINK
      #ifdef GMATUQ_EXPORTS
         #define GMATUQ_API __declspec(dllexport)
      #else
         #define GMATUQ_API __declspec(dllimport)
      #endif
   #endif
#endif

#ifndef GMATUQ_API
   #define GMATUQ_API
#endif

#endif // GmatUQDefs_hpp
