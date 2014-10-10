
__all__ = ['Plugin', 'implements', 'Interface', 'CreatePluginFactory', 'PluginMeta', 'alias', 'ExtensionPoint', 'SingletonPlugin', 'PluginFactory', 'PluginError']

import sys
import weakref


if False:
    #
    # Plugins defined with pyutilib.component.core
    #
    from pyutilib.component.core import Plugin, implements, Interface, CreatePluginFactory, PluginMeta, alias, ExtensionPoint, SingletonPlugin, PluginFactory

    def push(name):
        import pyutilib.component.core
        pyutilib.component.core.PluginGlobals.push_env(name)

    def pop():
        import pyutilib.component.core
        pyutilib.component.core.PluginGlobals.pop_env()

    def clear():
        import pyutilib.component.core
        pyutilib.component.core.PluginGlobals.clear()

else:

    from pyomo.misc._plugin import *

