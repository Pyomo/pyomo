from pyomo.util.plugin import PluginGlobals
PluginGlobals.add_env("pyomo")

from pyomo.openopt.func_designer import *

PluginGlobals.pop_env()
