#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['Plugin', 'implements', 'Interface', 'CreatePluginFactory', 'PluginMeta', 'alias', 'ExtensionPoint', 'SingletonPlugin', 'PluginFactory', 'PluginError', 'PluginGlobals']

#
# Plugins defined with pyutilib.component.core
#
from pyutilib.component.core import Plugin, implements, Interface, CreatePluginFactory, PluginMeta, alias, ExtensionPoint, SingletonPlugin, PluginFactory, PluginError, PluginGlobals

def push(name):
    import pyutilib.component.core
    pyutilib.component.core.PluginGlobals.add_env(name)

def pop():
    import pyutilib.component.core
    pyutilib.component.core.PluginGlobals.pop_env()

def clear():                        #pragma:nocover
    import pyutilib.component.core
    pyutilib.component.core.PluginGlobals.clear()

