#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# TODO: Disabled until we can confirm application to Pyomo models
import pyomo.solvers.plugins.solvers.ps
import pyomo.solvers.plugins.solvers.PICO
import pyomo.solvers.plugins.solvers.CBCplugin
import pyomo.solvers.plugins.solvers.GLPK
import pyomo.solvers.plugins.solvers.GLPK_old
import pyomo.solvers.plugins.solvers.glpk_direct
import pyomo.solvers.plugins.solvers.CPLEX
import pyomo.solvers.plugins.solvers.CPLEXDirect
import pyomo.solvers.plugins.solvers.CPLEXPersistent
import pyomo.solvers.plugins.solvers.GUROBI
import pyomo.solvers.plugins.solvers.BARON
import pyomo.solvers.plugins.solvers.gurobi_direct
import pyomo.solvers.plugins.solvers.ASL
import pyomo.solvers.plugins.solvers.pywrapper
import pyomo.solvers.plugins.solvers.SCIPAMPL
import pyomo.solvers.plugins.solvers.CONOPT
import pyomo.solvers.plugins.solvers.XPRESS

#
# Interrogate the CBC executable to see if it recognizes the -AMPL flag
#
from pyomo.solvers.plugins.solvers.CBCplugin import configure_cbc
configure_cbc()
del configure_cbc

#
# Interrogate the glpsol executable to see if it is new enough to allow the new parser logic
#
from pyomo.solvers.plugins.solvers.GLPK import configure_glpk
configure_glpk()
del configure_glpk
