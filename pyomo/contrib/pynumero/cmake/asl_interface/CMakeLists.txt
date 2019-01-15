#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

cmake_minimum_required(VERSION 3.2)

#set(USE_ASL_PATH "" CACHE FILEPATH "Set the path to the ASL solvers directory containing a compiled amplsolver.a library.")
set(MP_PATH "" CACHE FILEPATH "Set the path to the ampl-mp package to link against ampl-mp asl library")

# set the default directory for ASL
set(DEFAULT_ASL_PATH "${PROJECT_SOURCE_DIR}/third_party/ASL/solvers")

if(APPLE)
    set(ASL_SOURCES
    src/AmplInterface.cpp
    src/AssertUtils.hpp
    )
else()
    set(ASL_SOURCES
    src/AmplInterface.cpp
    src/AssertUtils.hpp
    )
endif()

if (MP_PATH)
   ADD_LIBRARY( pynumero_ASL SHARED ${ASL_SOURCES})

   FIND_LIBRARY(ASL_LIB
	        NAMES asl libdasl
                HINTS "${MP_PATH}/lib/"
		)
		
   set(ASL_INCLUDE_DIRS "${MP_PATH}/include/asl")
   set( ASL_LIBRARIES "${ASL_LIB}" )
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
   INSTALL(TARGETS pynumero_ASL
           DESTINATION lib
          )
   INSTALL(DIRECTORY src
           DESTINATION include
	   )
endif()
