#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['Plugin', 'implements', 'Interface', 'CreatePluginFactory', 'PluginMeta', 'alias', 'ExtensionPoint', 'SingletonPlugin', 'PluginFactory', 'PluginError', 'PluginGlobals']

import sys
import weakref


if True:
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

    def clear():
        import pyutilib.component.core
        pyutilib.component.core.PluginGlobals.clear()

else:

    from pyomo.util._plugin import *

