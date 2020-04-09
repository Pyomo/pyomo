cmake_minimum_required(VERSION 3.2)
PROJECT( pynumero )

include(ExternalProject)

# Targets in this project
OPTION(BUILD_ASL "Build the PyNumero ASL interface" ON)
OPTION(BUILD_MA27 "Build the PyNumero ma27 interface" OFF)
OPTION(BUILD_MA57 "Build the PyNumero ma57 interface" OFF)

# Dependencies that we manage / can install
SET(AMPLMP_TAG "3.1.0" CACHE STRING
  "AMPL/MP git tag/branch to checkout and build")
OPTION(BUILD_AMPLMP
  "Download and build AMPL/MP ${AMPLMP_TAG} from GitHub" OFF)

# Other build / environment options
OPTION(BUILD_AMPLMP_IF_NEEDED
  "Automatically enable AMPLMP build if ASL not found" OFF)
MARK_AS_ADVANCED(BUILD_AMPLMP_IF_NEEDED)

#OPTION(STATIC_LINK "STATIC_LINK" OFF)

# If we build AMPLMP, then we will get a dependency on dlopen
FIND_LIBRARY(DL_LIBRARY dl)

# We need the ASL and HSL libraries.  We can get them from Ipopt,
# AMPL/MP, or ASL (netlib)
SET(IPOPT_DIR CACHE PATH "Path to compiled Ipopt installation")
SET(AMPLMP_DIR "" CACHE PATH "Path to compiled AMPL/MP installation")
#SET(ASL_NETLIB_DIR "" CACHE PATH "Path to compiled ASL (netlib) installation")

# Use pkg-config to get the ASL/HSL directories from the Ipopt/COIN-OR build
FIND_PACKAGE(PkgConfig)
IF( PKG_CONFIG_FOUND )
  SET(_TMP "$ENV{PKG_CONFIG_PATH}")
  SET(ENV{PKG_CONFIG_PATH} "${IPOPT_DIR}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  pkg_check_modules(PC_COINASL QUIET coinasl)
  pkg_check_modules(PC_COINHSL QUIET coinhsl)
  SET(ENV{PKG_CONFIG_PATH} "${_TMP}")
ENDIF()

# Note: the directory search order is intentional: first the modules we
# are creating, then directories specifically set by the user, and
# finally automatically located installations (e.g., from pkg-config)
FIND_PATH(ASL_INCLUDE_DIR asl_pfgh.h
  HINTS "${CMAKE_INSTALL_PREFIX}/include"
        "${IPOPT_DIR}/include/coin/ThirdParty"
        "${AMPLMP_DIR}/include"
        "${PC_COINASL_INCLUDEDIR}"
        "${PC_COINASL_INCLUDE_DIRS}"
  PATH_SUFFIXES asl
)
FIND_LIBRARY(ASL_LIBRARY NAMES coinasl asl
  HINTS "${CMAKE_INSTALL_PREFIX}/lib"
        "${IPOPT_DIR}/lib"
        "${AMPLMP_DIR}/lib"
        "${PC_COINASL_LIBDIR}"
        "${PC_COINASL_LIBRARY_DIRS}"
)
FIND_LIBRARY(HSL_LIBRARY NAMES coinhsl libcoinhsl
  HINTS "${CMAKE_INSTALL_PREFIX}/lib"
        "${IPOPT_DIR}/lib"
        "${PC_COINHSL_LIBDIR}"
        "${PC_COINHSL_LIBRARY_DIRS}"
)

IF( BUILD_AMPLMP_IF_NEEDED AND (NOT ASL_LIBRARY OR NOT ASL_INCLUDE_DIR) )
    set_property(CACHE BUILD_AMPLMP PROPERTY VALUE ON)
ENDIF()

IF( BUILD_AMPLMP )
  get_filename_component(ABS_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
  ExternalProject_Add(amplmp
    GIT_TAG ${AMPLMP_TAG}
    GIT_REPOSITORY https://github.com/ampl/mp.git
    GIT_SHALLOW 1
    # We don't need *any* submodules, but leaving it as an empty string
    # doesn't disable it as suggested by the documentation
    GIT_SUBMODULES thirdparty/gsl
    #GIT_SUBMODULES_RECURSE 0
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${ABS_INSTALL_PREFIX}
    # 3.1.0 needs to be patched to compile with recent compilers
    # (notably ubuntu 18.04; see
    # https://github.com/easybuilders/easybuild-easyconfigs/issues/9380
    PATCH_COMMAND git apply
       ${CMAKE_CURRENT_SOURCE_DIR}/amplmp-${AMPLMP_TAG}.patch
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

set(PYNUMERO_ASL_SOURCES
  "AmplInterface.cpp"
  "AmplInterface.hpp"
  "AssertUtils.hpp"
)

IF( BUILD_ASL )
  ADD_LIBRARY( pynumero_ASL SHARED ${PYNUMERO_ASL_SOURCES} )
  TARGET_LINK_LIBRARIES( pynumero_ASL PUBLIC ${ASL_LIBRARY} )
  if ( DL_LIBRARY )
    TARGET_LINK_LIBRARIES( pynumero_ASL PUBLIC ${DL_LIBRARY} )
  ENDIF()
  TARGET_INCLUDE_DIRECTORIES( pynumero_ASL
    PUBLIC ${ASL_INCLUDE_DIR}
    INTERFACE . )
  SET_TARGET_PROPERTIES( pynumero_ASL PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL(TARGETS pynumero_ASL LIBRARY DESTINATION lib )
  IF( BUILD_AMPLMP )
    # If we are building AMPL/MP, it is possible that we are linking
    # against it, so we will add the appropriate dependency
    add_dependencies(pynumero_ASL amplmp)
  ENDIF()
ENDIF()

set(PYNUMERO_MA27_SOURCES
  "ma27Interface.cpp"
  "ma27Interface.hpp"
)

IF( BUILD_MA27 )
  ADD_LIBRARY( pynumero_MA27 SHARED ${PYNUMERO_MA27_SOURCES} )
  TARGET_LINK_LIBRARIES( pynumero_MA27 ${HSL_LIBRARY} )
  SET_TARGET_PROPERTIES( pynumero_MA27 PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL(TARGETS pynumero_MA27 LIBRARY DESTINATION lib )
ENDIF()

set(PYNUMERO_MA57_SOURCES
  "ma57Interface.cpp"
  "ma57Interface.hpp"
)

IF( BUILD_MA57 )
  ADD_LIBRARY( pynumero_MA57 SHARED ${PYNUMERO_MA57_SOURCES} )
  TARGET_LINK_LIBRARIES( pynumero_MA57 ${HSL_LIBRARY} )
  SET_TARGET_PROPERTIES( pynumero_MA57 PROPERTIES ENABLE_EXPORTS 1 )
  INSTALL(TARGETS pynumero_MA57 LIBRARY DESTINATION lib )
ENDIF()

#
# build the tests for the interfaces
#
add_subdirectory(tests)
