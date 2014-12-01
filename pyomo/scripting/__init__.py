#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

from pyomo.util.plugin import PluginGlobals
PluginGlobals.add_env("pyomo")

import pyomo.scripting.pyomo_command
import pyomo.scripting.util

PluginGlobals.pop_env()
