# GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
#
# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the
# repository root).
#
# In-tree build, as part of GMAT's own CMake build (option PLUGIN_GMATUQ).
# To build the plugin against an installed GMAT instead, see ../../CMakeLists.txt.

MESSAGE("==============================")
MESSAGE("GMAT-UQ Plugin setup " ${VERSION})

SET(TargetName GmatUQ)

SET(PLUGIN_DIRS command factory include plugin resource)

SET(PLUGIN_SRCS
    command/RunUncertaintyAnalysis.cpp
    factory/GmatUQFactory.cpp
    plugin/GmatPluginFunctions.cpp
    resource/UQObject.cpp
    resource/UQResources.cpp
)

_SETUPPLUGIN(${TargetName} "${PLUGIN_DIRS}" "${PLUGIN_SRCS}" plugins)

SET_TARGET_PROPERTIES(${TargetName} PROPERTIES DEFINE_SYMBOL "GMATUQ_EXPORTS")
