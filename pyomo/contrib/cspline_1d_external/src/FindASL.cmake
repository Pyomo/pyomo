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
# Minimum version inherited from AMPL/asl

include(ExternalProject)

# Dependencies that we manage / can install
SET(AMPLASL_TAG "9fb7cb8e4f68ed1c3bc066d191e63698b7d7d1d2" CACHE STRING
  "AMPL/asl git tag/branch to checkout and build")
# 9fb7cb8e4f68ed1c3bc066d191e63698b7d7d1d2 corresponds to ASLdate = 20211109
OPTION(BUILD_AMPLASL
  "Download and build AMPL/asl ${AMPLASL_TAG} from GitHub" OFF)

# Other build / environment options
OPTION(BUILD_AMPLASL_IF_NEEDED
  "Automatically enable AMPL/asl build if ASL not found" OFF)
MARK_AS_ADVANCED(BUILD_AMPLASL_IF_NEEDED)

OPTION(ASL_USE_PKG_CONFIG,
  "Use pkgconfig (if present) to attempt to locate the ASL" OFF)
#OPTION(STATIC_LINK "STATIC_LINK" OFF)

# We need the ASL. We can get it from Ipopt, AMPL/asl, or ASL (netlib)
SET(IPOPT_DIR "" CACHE PATH "Path to compiled Ipopt installation")
SET(AMPLASL_DIR "" CACHE PATH "Path to compiled AMPL/asl installation")
#SET(ASL_NETLIB_DIR "" CACHE PATH "Path to compiled ASL (netlib) installation")

# Use pkg-config to get the ASL directories from the Ipopt/COIN-OR build
FIND_PACKAGE(PkgConfig)
IF( PKG_CONFIG_FOUND AND ASL_USE_PKG_CONFIG )
  SET(_TMP "$ENV{PKG_CONFIG_PATH}")
  SET(ENV{PKG_CONFIG_PATH} "${IPOPT_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  pkg_check_modules(PC_COINASL QUIET coinasl)
  SET(ENV{PKG_CONFIG_PATH} "${_TMP}")
ENDIF()

# cmake does not search LD_LIBRARY_PATH by default.  So that libraries
# can be added through mechanisms like 'environment modules', we will explicitly
# add LD_LIBRARY_PATH to the search path
string(REPLACE ":" ";" LD_LIBRARY_DIR_LIST
  $ENV{LD_LIBRARY_PATH}:$ENV{DYLD_LIBRARY_PATH}
  )

# Note: the directory search order is intentional: first the modules we
# are creating, then directories specifically set by the user, and
# finally automatically located installations (e.g., from pkg-config)
FIND_PATH(ASL_INCLUDE_DIR asl_pfgh.h
  HINTS "${CMAKE_INSTALL_PREFIX}/include/asl"
        "${IPOPT_DIR}/include/coin-or/asl"
        "${IPOPT_DIR}/include/coin/ThirdParty"
        "${AMPLASL_DIR}/include/asl"
        "${PC_COINASL_INCLUDEDIR}"
        "${PC_COINASL_INCLUDE_DIRS}"
  PATH_SUFFIXES asl
)
FIND_LIBRARY(ASL_LIBRARY NAMES asl coinasl
  HINTS "${CMAKE_INSTALL_PREFIX}/lib"
        "${IPOPT_DIR}/lib"
        "${AMPLASL_DIR}/lib"
        "${PC_COINASL_LIBDIR}"
        "${PC_COINASL_LIBRARY_DIRS}"
        ${LD_LIBRARY_DIR_LIST}
)

# If BUILD_AMPLASL_IF_NEEDED is set and we couldn't find / weren't
# pointed to an ASL build, then we will forcibly enable the AMPL/asl build
# to provide the ASL.
IF( BUILD_AMPLASL_IF_NEEDED AND (NOT ASL_LIBRARY OR NOT ASL_INCLUDE_DIR) )
    set_property(CACHE BUILD_AMPLASL PROPERTY VALUE ON)
ENDIF()

IF( BUILD_AMPLASL )
  get_filename_component(ABS_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
  ExternalProject_Add(ampl_asl
    GIT_TAG ${AMPLASL_TAG}
    GIT_REPOSITORY https://github.com/ampl/asl.git
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${ABS_INSTALL_PREFIX}
    UPDATE_DISCONNECTED TRUE
    )
  # Update the ASL paths (if necessary).  Since these do not (yet)
  # exist, we need to bypass find_path / find_library and explicitly set
  # the directories that this build will create.  However, we will only
  # do this if the paths have not already been set (so users can always
  # override what we do here)
  IF(NOT ASL_INCLUDE_DIR OR NOT ASL_LIBRARY)
    set_property(CACHE ASL_INCLUDE_DIR PROPERTY VALUE
      "${ABS_INSTALL_PREFIX}/include/asl")
    IF( WIN32 )
      set_property(CACHE ASL_LIBRARY PROPERTY VALUE
        "${ABS_INSTALL_PREFIX}/lib/asl.lib")
    ELSE()
      set_property(CACHE ASL_LIBRARY PROPERTY VALUE
        "${ABS_INSTALL_PREFIX}/lib/libasl.a")
    ENDIF()
  ENDIF()
ENDIF()
