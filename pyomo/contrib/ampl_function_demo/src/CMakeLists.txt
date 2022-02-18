cmake_minimum_required(VERSION 3.0)
# CMake 3.0 added GIT_SUBMODULES to ExternalProject_ADD, and without it
# the Ampl/MP checkout fails because one of the submodules (gecode) is a
# private repository.

PROJECT( amlp_function_demo )

INCLUDE(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../ampl_function_demo/src/FindASL.cmake")

# Targets in this project
OPTION(BUILD_EXTERNAL_FCN_LIBRARY
  "Build the ASL external function example library" ON)

IF( BUILD_EXTERNAL_FCN_LIBRARY )
  ADD_LIBRARY( function_ASL SHARED "functions.c" )
  TARGET_LINK_LIBRARIES( function_ASL PUBLIC ${ASL_LIBRARY} ${CMAKE_DL_LIBS})
  TARGET_INCLUDE_DIRECTORIES( function_ASL
    PUBLIC ${ASL_INCLUDE_DIR}
    INTERFACE . )
  #TARGET_COMPILE_DEFINITIONS( function_ASL PRIVATE BUILDING_FUNCTION_ASL )
  #SET_TARGET_PROPERTIES( function_ASL PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL( TARGETS function_ASL LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib )
  IF( BUILD_AMPLMP )
    # If we are building AMPL/MP (from FindASL), it is possible that we
    # are linking against it, so we will add the appropriate dependency
    add_dependencies(function_ASL amplmp)
  ENDIF()
ENDIF()
