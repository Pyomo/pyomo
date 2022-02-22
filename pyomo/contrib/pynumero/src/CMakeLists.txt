cmake_minimum_required(VERSION 3.0)
# CMake 3.0 added GIT_SUBMODULES to ExternalProject_ADD, and without it
# the Ampl/MP checkout fails because one of the submodules (gecode) is a
# private repository.

PROJECT( pynumero )

# Targets in this project
OPTION(BUILD_ASL "Build the PyNumero ASL interface" ON)
OPTION(BUILD_MA27 "Build the PyNumero ma27 interface" OFF)
OPTION(BUILD_MA57 "Build the PyNumero ma57 interface" OFF)

OPTION(ENABLE_HSL "Enable the HSL library interfaces" ON)
MARK_AS_ADVANCED(ENABLE_HSL)

#OPTION(STATIC_LINK "STATIC_LINK" OFF)

INCLUDE(
  "${CMAKE_CURRENT_SOURCE_DIR}/../../ampl_function_demo/src/FindASL.cmake")

SET(MA27_OBJECT "" CACHE FILEPATH
	"Path to compiled ma27d.o object. Must be compiled with -fPIC.")

# Use pkg-config to get the ASL/HSL directories from the Ipopt/COIN-OR build
FIND_PACKAGE(PkgConfig)
IF( PKG_CONFIG_FOUND )
  SET(_TMP "$ENV{PKG_CONFIG_PATH}")
  SET(ENV{PKG_CONFIG_PATH} "${IPOPT_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  pkg_check_modules(PC_COINHSL QUIET coinhsl)
  SET(ENV{PKG_CONFIG_PATH} "${_TMP}")
ENDIF()

# cmake does not search LD_LIBRARY_PATH by default.  So that libraries
# like HSL can be added through mechanisms like 'environment modules',
# we will explicitly add LD_LIBRARY_PATH to the search path
string(REPLACE ":" ";" LD_LIBRARY_DIR_LIST
  $ENV{LD_LIBRARY_PATH}:$ENV{DYLD_LIBRARY_PATH}
  )

FIND_LIBRARY(MA27_LIBRARY NAMES coinhsl libcoinhsl ma27 libma27
  HINTS "${CMAKE_INSTALL_PREFIX}/lib"
        "${IPOPT_DIR}/lib"
        "${PC_COINHSL_LIBDIR}"
        "${PC_COINHSL_LIBRARY_DIRS}"
        "${MA27_DIR}"
        "${MA27_DIR}/lib"
        ${LD_LIBRARY_DIR_LIST}
)
FIND_LIBRARY(MA57_LIBRARY NAMES coinhsl libcoinhsl ma57 libma57
  HINTS "${CMAKE_INSTALL_PREFIX}/lib"
        "${IPOPT_DIR}/lib"
        "${PC_COINHSL_LIBDIR}"
        "${PC_COINHSL_LIBRARY_DIRS}"
        "${MA57_DIR}"
        "${MA57_DIR}/lib"
        ${LD_LIBRARY_DIR_LIST}
)

# If we were able to find the HSL, we will automatically enable the ma27
# interface, as all versions of the HSL library contain ma27.
IF( MA27_LIBRARY OR MA27_OBJECT )
    set_property(CACHE BUILD_MA27 PROPERTY VALUE ON)
ENDIF()

#...but if the HSL interface is not enabled, do not build the MA* libraries
IF( NOT ENABLE_HSL )
    set_property(CACHE BUILD_MA27 PROPERTY VALUE OFF)
    set_property(CACHE BUILD_MA57 PROPERTY VALUE OFF)
ENDIF()


IF( BUILD_ASL )
  ADD_LIBRARY( pynumero_ASL SHARED
    "AmplInterface.cpp"
    "AmplInterface.hpp"
    "AssertUtils.hpp"
    )
  TARGET_LINK_LIBRARIES( pynumero_ASL PUBLIC ${ASL_LIBRARY} ${CMAKE_DL_LIBS})
  TARGET_INCLUDE_DIRECTORIES( pynumero_ASL
    PUBLIC ${ASL_INCLUDE_DIR}
    INTERFACE . )
  TARGET_COMPILE_DEFINITIONS( pynumero_ASL PRIVATE BUILDING_PYNUMERO_ASL )
  SET_TARGET_PROPERTIES( pynumero_ASL PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL( TARGETS pynumero_ASL LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib )
  IF( BUILD_AMPLASL )
    # If we are building AMPL/MP, it is possible that we are linking
    # against it, so we will add the appropriate dependency
    add_dependencies(pynumero_ASL ampl_asl)
  ENDIF()
ENDIF()

#
# build hsl interfaces
#

IF( BUILD_MA27 )
  ADD_LIBRARY( pynumero_MA27 SHARED "ma27Interface.cpp" )
  IF( MA27_OBJECT )
    TARGET_LINK_LIBRARIES( pynumero_MA27 ${MA27_OBJECT} )
  ELSE()
    TARGET_LINK_LIBRARIES( pynumero_MA27 ${MA27_LIBRARY} )
  ENDIF()
  TARGET_LINK_LIBRARIES( pynumero_ASL PUBLIC ${CMAKE_DL_LIBS} )
  TARGET_COMPILE_DEFINITIONS( pynumero_MA27 PRIVATE BUILDING_PYNUMERO_MA27 )
  SET_TARGET_PROPERTIES( pynumero_MA27 PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL(TARGETS pynumero_MA27 LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib )
ENDIF()

IF( BUILD_MA57 )
  ADD_LIBRARY( pynumero_MA57 SHARED "ma57Interface.cpp" )
  TARGET_LINK_LIBRARIES( pynumero_MA57 ${MA57_LIBRARY} ${CMAKE_DL_LIBS} )
  TARGET_COMPILE_DEFINITIONS( pynumero_MA27 PRIVATE BUILDING_PYNUMERO_MA57 )
  SET_TARGET_PROPERTIES( pynumero_MA57 PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL(TARGETS pynumero_MA57 LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib )
ENDIF()

#
# build the tests for the interfaces
#
add_subdirectory(tests)
