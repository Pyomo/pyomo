#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

set(ASL_HEADERS "${PROJECT_SOURCE_DIR}/asl_interface/src" )
ADD_EXECUTABLE(asl_test src/simple_test.cpp)
TARGET_INCLUDE_DIRECTORIES( asl_test PUBLIC ${ASL_HEADERS} )
#SET_PROPERTY(TARGET asl_test PROPERTY ENABLE_EXPORTS)
TARGET_LINK_LIBRARIES( asl_test pynumero_ASL)

INSTALL(TARGETS asl_test
        DESTINATION bin
       )
