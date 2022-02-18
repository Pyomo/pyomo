cmake_minimum_required(VERSION 3.0)
# CMake 3.0 added GIT_SUBMODULES to ExternalProject_ADD, and without it
# the Ampl/MP checkout fails because one of the submodules (gecode) is a
# private repository.

include(ExternalProject)

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

# We need the ASL. We can get it from Ipopt, AMPL/MP, or ASL (netlib)
SET(IPOPT_DIR "" CACHE PATH "Path to compiled Ipopt installation")
SET(AMPLMP_DIR "" CACHE PATH "Path to compiled AMPL/MP installation")
#SET(ASL_NETLIB_DIR "" CACHE PATH "Path to compiled ASL (netlib) installation")

# Use pkg-config to get the ASL directories from the Ipopt/COIN-OR build
FIND_PACKAGE(PkgConfig)
IF( PKG_CONFIG_FOUND )
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
  HINTS "${CMAKE_INSTALL_PREFIX}/include"
        "${IPOPT_DIR}/include/coin-or/asl"
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
        ${LD_LIBRARY_DIR_LIST}
)

# If BUILD_AMPLMP_IF_NEEDED is set and we couldn't find / weren't
# pointed to an ASL build, then we will forcibly enable the AMPLMP build
# to provide the ASL.
IF( BUILD_AMPLMP_IF_NEEDED AND (NOT ASL_LIBRARY OR NOT ASL_INCLUDE_DIR) )
    set_property(CACHE BUILD_AMPLMP PROPERTY VALUE ON)
ENDIF()

IF( BUILD_AMPLMP )
  get_filename_component(ABS_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
  ExternalProject_Add(amplmp
    GIT_TAG ${AMPLMP_TAG}
    GIT_REPOSITORY https://github.com/ampl/mp.git
    # We don't need *any* submodules, but leaving it as an empty string
    # doesn't disable it as suggested by the documentation.  A
    # "workaround" from the web is to specify an existing directory that
    # is *not* a submodule
    GIT_SUBMODULES test
    CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:STRING=${ABS_INSTALL_PREFIX}
    UPDATE_DISCONNECTED TRUE
    # 3.1.0 needs to be patched to compile with recent compilers,
    # notably ubuntu 18.04.  The patch applies a backport of fmtlib/fmt
    # abbefd7; see https://github.com/fmtlib/fmt/issues/398
    # The patch also disables AMPL/MP tests to speed up compilation.
    PATCH_COMMAND git apply
       ${CMAKE_CURRENT_LIST_DIR}/amplmp-${AMPLMP_TAG}.patch
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
