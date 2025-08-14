#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

cmake_minimum_required(VERSION 3.0...3.31)
# This was developed against CMake 3.0, and appears to comply with 3.5

PROJECT( cspline_external )

INCLUDE(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../ampl_function_demo/src/FindASL.cmake")

# Targets in this project
OPTION(BUILD_EXTERNAL_FCN_LIBRARY
  "Build the ASL external function example library" ON)

IF( BUILD_EXTERNAL_FCN_LIBRARY )
  ADD_LIBRARY( cspline_external SHARED "functions.cpp" )
  TARGET_LINK_LIBRARIES( cspline_external
    PUBLIC ${ASL_LIBRARY} ${CMAKE_DL_LIBS})
  TARGET_INCLUDE_DIRECTORIES( cspline_external
    PUBLIC ${ASL_INCLUDE_DIR}
    INTERFACE . )
  # If you need a CPP directive defined when building the library (e.g.,
  # for managing __declspec(dllimport) under Windows, uncomment the
  # following:
  #TARGET_COMPILE_DEFINITIONS( cspline_external PRIVATE BUILDING_ASL_DEMO )
  #SET_TARGET_PROPERTIES( cspline_external PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL( TARGETS cspline_external LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib )
  IF( BUILD_AMPLASL )
    # If we are building AMPL/asl (from FindASL), it is possible that we
    # are linking against it, so we will add the appropriate dependency
    add_dependencies(cspline_external ampl_asl)
  ENDIF()
ENDIF()
