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
