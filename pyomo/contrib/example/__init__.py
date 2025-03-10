#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

#
# Import "public" symbols and sub-packages.
#
from pyomo.contrib.example.foo import *
from pyomo.contrib.example import bar

#
# Register plugins from this sub-package.
#
# The pyomo.environ package normally calls the load() function in a
# hard-coded list of pyomo.*.plugins and pyomo.contrib.*.plugins
# modules.  However, This example is not included in that list, so we
# will load (and register) the plugins when this module (or any
# submodule) is imported.
#
from pyomo.contrib.example.plugins import load

load()
