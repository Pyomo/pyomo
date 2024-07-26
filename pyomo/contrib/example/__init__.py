#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# import symbols and sub-packages
#
from pyomo.contrib.example.foo import *
import pyomo.contrib.example.bar

#
# import the plugins directory
#
# The pyomo.environ package normally calls the load() function in
# the pyomo.*.plugins subdirectories.  However, pyomo.contrib packages
# are not loaded by pyomo.environ, so we need to call this function
# when we import the rest of this package.
#
from pyomo.contrib.example.plugins import load

load()
