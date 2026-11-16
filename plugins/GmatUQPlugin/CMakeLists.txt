# GMAT-UQ plugin for the General Mission Analysis Tool (GMAT)
#
# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the
# repository root).
#
# Stand-alone build of the plugin against an installed GMAT release.
#
#   cmake -S plugins/GmatUQPlugin -B build-uq -DCMAKE_BUILD_TYPE=Release \
#         -DGMAT_SOURCE_DIR=<GMAT source tree checked out at the release tag, e.g. R2026a> \
#         -DGMAT_INSTALL_DIR=<installed GMAT, the folder that contains bin and plugins> \
#         -DCSPICE_INCLUDE_DIR=<cspice/include from NAIF>
#   cmake --build build-uq --config Release
#
# The headers must come from the same release as the installed binaries: a
# plugin compiled against newer headers can disagree with GMAT on the layout
# of its classes.  CSPICE headers are needed because GMAT's solar-system
# headers include them (GMAT is built with SPICE).

cmake_minimum_required(VERSION 3.16)
project(GmatUQPlugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(GMAT_SOURCE_DIR "" CACHE PATH "GMAT source tree at the release tag of the installed GMAT")
set(GMAT_INSTALL_DIR "" CACHE PATH "Installed GMAT (contains bin/ and plugins/)")
set(CSPICE_INCLUDE_DIR "" CACHE PATH "CSPICE include folder")
foreach(required GMAT_SOURCE_DIR GMAT_INSTALL_DIR CSPICE_INCLUDE_DIR)
  if(NOT EXISTS "${${required}}")
    message(FATAL_ERROR "Set ${required} (see the comment at the top of this file)")
  endif()
endforeach()

# Every folder of GMAT's base and utility libraries holds public headers.
file(GLOB_RECURSE GMAT_HEADERS LIST_DIRECTORIES false
     "${GMAT_SOURCE_DIR}/src/base/*.hpp" "${GMAT_SOURCE_DIR}/src/gmatutil/*.hpp"
     "${GMAT_SOURCE_DIR}/src/gmatutil/*.h")
set(GMAT_INCLUDE_DIRS "")
foreach(header ${GMAT_HEADERS})
  get_filename_component(folder "${header}" DIRECTORY)
  list(APPEND GMAT_INCLUDE_DIRS "${folder}")
endforeach()
list(REMOVE_DUPLICATES GMAT_INCLUDE_DIRS)

set(PLUGIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src/base")
add_library(GmatUQ SHARED
  ${PLUGIN_ROOT}/command/RunUncertaintyAnalysis.cpp
  ${PLUGIN_ROOT}/factory/GmatUQFactory.cpp
  ${PLUGIN_ROOT}/parameter/UQResultParameter.cpp
  ${PLUGIN_ROOT}/plugin/GmatPluginFunctions.cpp
  ${PLUGIN_ROOT}/resource/UQObject.cpp
  ${PLUGIN_ROOT}/resource/UQResources.cpp
)
set_target_properties(GmatUQ PROPERTIES PREFIX "lib" DEFINE_SYMBOL "GMATUQ_EXPORTS")
target_include_directories(GmatUQ PRIVATE
  ${PLUGIN_ROOT}/command ${PLUGIN_ROOT}/factory ${PLUGIN_ROOT}/include
  ${PLUGIN_ROOT}/parameter ${PLUGIN_ROOT}/plugin ${PLUGIN_ROOT}/resource
  ${GMAT_INCLUDE_DIRS} "${CSPICE_INCLUDE_DIR}")

# The definitions GMAT itself is built with (see the top-level CMakeLists.txt of GMAT).
target_compile_definitions(GmatUQ PRIVATE
  _DYNAMICLINK __USE_SPICE__ NO_GCC_PRAGMA UNICODE _UNICODE USE_64_BIT_LONGS)

if(WIN32)
  target_compile_options(GmatUQ PRIVATE /wd4430 /wd4251 /wd4231 /wd4996 /EHsc)
  # GMAT's binary release ships DLLs without import libraries; build them from the exports.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(IMPORT_DIR "${CMAKE_CURRENT_BINARY_DIR}/gmat-import")
  foreach(lib libGmatBase libGmatUtil)
    set(dll "${GMAT_INSTALL_DIR}/bin/${lib}.dll")
    if(NOT EXISTS "${dll}")
      message(FATAL_ERROR "${dll} not found")
    endif()
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/make_import_lib.py"
              "${dll}" "${IMPORT_DIR}" "${CMAKE_LINKER}"
      RESULT_VARIABLE status)
    if(NOT status EQUAL 0)
      message(FATAL_ERROR "Could not build the import library of ${dll}")
    endif()
    target_link_libraries(GmatUQ PRIVATE "${IMPORT_DIR}/${lib}.lib")
  endforeach()
else()
  target_compile_definitions(GmatUQ PRIVATE LINUX_MAC)
  target_compile_options(GmatUQ PRIVATE -fPIC -fno-strict-aliasing)
  find_library(GMAT_BASE_LIB NAMES GmatBase libGmatBase PATHS "${GMAT_INSTALL_DIR}/bin" "${GMAT_INSTALL_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(GMAT_UTIL_LIB NAMES GmatUtil libGmatUtil PATHS "${GMAT_INSTALL_DIR}/bin" "${GMAT_INSTALL_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
  target_link_libraries(GmatUQ PRIVATE "${GMAT_BASE_LIB}" "${GMAT_UTIL_LIB}")
endif()

install(TARGETS GmatUQ
        RUNTIME DESTINATION plugins
        LIBRARY DESTINATION plugins)
