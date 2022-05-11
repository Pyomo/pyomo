#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

cmake_minimum_required(VERSION 3.0)
# CMake 3.0 needed by AMPL/asl build

PROJECT( ampl_function_demo )

INCLUDE(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../ampl_function_demo/src/FindASL.cmake")

# Targets in this project
OPTION(BUILD_EXTERNAL_FCN_LIBRARY
  "Build the ASL external function example library" ON)

IF( BUILD_EXTERNAL_FCN_LIBRARY )
  ADD_LIBRARY( asl_external_demo SHARED "functions.c" )
  TARGET_LINK_LIBRARIES( asl_external_demo
    PUBLIC ${ASL_LIBRARY} ${CMAKE_DL_LIBS})
  TARGET_INCLUDE_DIRECTORIES( asl_external_demo
    PUBLIC ${ASL_INCLUDE_DIR}
    INTERFACE . )
  # If you need a CPP directive defined when building the library (e.g.,
  # for managing __declspec(dllimport) under Windows, uncomment the
  # following:
  #TARGET_COMPILE_DEFINITIONS( asl_external_demo PRIVATE BUILDING_ASL_DEMO )
  #SET_TARGET_PROPERTIES( asl_external_demo PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL( TARGETS asl_external_demo LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib )
  IF( BUILD_AMPLASL )
    # If we are building AMPL/asl (from FindASL), it is possible that we
    # are linking against it, so we will add the appropriate dependency
    add_dependencies(asl_external_demo ampl_asl)
  ENDIF()
ENDIF()
