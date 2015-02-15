#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

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

