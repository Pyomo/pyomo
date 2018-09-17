
cmake_minimum_required(VERSION 3.2)

#set(USE_ASL_PATH "" CACHE FILEPATH "Set the path to the ASL solvers directory containing a compiled amplsolver.a library.")
#message( STATUS "USE_ASL_PATH: " ${USE_ASL_PATH})

set(MP_PATH "" CACHE FILEPATH "Set the path to the ampl-mp package to link against ampl-mp asl library")
#message( STATUS "MP_PATH: " ${MP_PATH})

# set the default directory for ASL
set(DEFAULT_ASL_PATH "${PROJECT_SOURCE_DIR}/third_party/ASL/solvers")

if(APPLE)
    set(ASL_SOURCES
    src/AmplInterface.cpp
    src/AssertUtils.hpp
    src/fmemopen.c
    )
else()
    set(ASL_SOURCES
    src/AmplInterface.cpp
    src/AssertUtils.hpp
    )
endif()

if (MP_PATH)
   ADD_LIBRARY( pynumero_ASL SHARED ${ASL_SOURCES})
   set(ASL_INCLUDE_DIRS "${MP_PATH}/include/asl")
   set( ASL_LIBRARIES "${MP_PATH}/lib/libasl.a" )
   set( WITH_AMPL true)
elseif(DEFAULT_ASL_PATH)
   ADD_LIBRARY( pynumero_ASL SHARED ${ASL_SOURCES})
   set(ASL_INCLUDE_DIRS "${DEFAULT_ASL_PATH}")
   set( ASL_LIBRARIES "${DEFAULT_ASL_PATH}/amplsolver.a" )
   set( WITH_AMPL true)
else()
   MESSAGE( STATUS "*** ASL (AMPL) support not included - set USE_ASL_PATH to compile in ASL support." )
   set( ASL_INCLUDE_DIRS "" )
   set( ASL_LIBRARIES "" )
   set( WITH_AMPL false)
endif()


if (${WITH_AMPL})
   MESSAGE("-- ASL_INCLUDE_DIRS ${ASL_INCLUDE_DIRS}")
   MESSAGE("-- ASL_LIBRARIES ${ASL_LIBRARIES}")
   TARGET_INCLUDE_DIRECTORIES( pynumero_ASL PUBLIC ${ASL_INCLUDE_DIRS} )
   TARGET_LINK_LIBRARIES( pynumero_ASL ${ASL_LIBRARIES} ${CMAKE_DL_LIBS})
endif()
