#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

from pyomo.util.plugin import PluginGlobals
PluginGlobals.add_env("pyomo")

from pyomo.gdp.disjunct import GDP_Error, Disjunct, Disjunction

# Do not import these files: importing them registers the transformation
# plugins with the pyomo script so that they get automatically invoked.
#import pyomo.gdp.bigm
#import pyomo.gdp.chull

PluginGlobals.pop_env()
