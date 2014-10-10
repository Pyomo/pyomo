#  _________________________________________________________________________
#
#  Coopr: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Coopr README.txt file.
#  _________________________________________________________________________

import coopr.solvers.plugins.solvers.ps
import coopr.solvers.plugins.solvers.PICO
import coopr.solvers.plugins.solvers.CBCplugin
import coopr.solvers.plugins.solvers.GLPK
import coopr.solvers.plugins.solvers.GLPK_old
import coopr.solvers.plugins.solvers.glpk_direct
import coopr.solvers.plugins.solvers.CPLEX
import coopr.solvers.plugins.solvers.CPLEXDirect
import coopr.solvers.plugins.solvers.CPLEXPersistent
import coopr.solvers.plugins.solvers.GUROBI
import coopr.solvers.plugins.solvers.BARON
import coopr.solvers.plugins.solvers.gurobi_direct
import coopr.solvers.plugins.solvers.ASL
import coopr.solvers.plugins.solvers.pywrapper
import coopr.solvers.plugins.solvers.SCIPAMPL
import coopr.solvers.plugins.solvers.XPRESS

#
# Interrogate the CBC executable to see if it recognizes the -AMPL flag
#
from coopr.solvers.plugins.solvers.CBCplugin import configure_cbc
configure_cbc()
del configure_cbc

#
# Interrogate the glpsol executable to see if it is new enough to allow the new parser logic
#
from coopr.solvers.plugins.solvers.GLPK import configure_glpk
configure_glpk()
del configure_glpk
