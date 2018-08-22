#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [
        'DataManagerFactory',
        'UnknownDataManager'
        ]

import logging
import pyutilib.misc
from pyomo.common.plugin import (
    implements, Interface, Plugin, PluginFactory, CreatePluginFactory,
    PluginError, ExtensionPoint )

logger = logging.getLogger('pyomo.core')


class IDataManager(Interface):

    def available(self):
        """ Returns True if the data manager can be executed """
        pass

    def requirements(self):
        """ Return a string describing the packages that need to be installed for this plugin to be available """
        pass

    def initialize(self, filename, **kwds):
        """ Prepare to read a data file. """
        pass

    def add_options(self, **kwds):
        """ Add options """
        pass

    def open(self):
        """ Open the data file. """
        pass

    def close(self):
        """ Close the data file. """
        pass

    def read(self):
        """ Read the data file. """
        pass

    def process(self, model, data, default):
        """ Process the data. """
        pass

    def clear(self):
        """ Reset Plugin. """
        pass


class UnknownDataManager(Plugin):

    implements(IDataManager)

    def __init__(self, *args, **kwds):
        Plugin.__init__(self, **kwds)
        #
        # The 'type' is the class type of the solver instance
        #
        self.type = kwds["type"]

    def available(self):
        return False


#
# A DataManagerFactory is an instance of a plugin factory that is
# customized with a custom __call__ method
#
DataManagerFactory = CreatePluginFactory(IDataManager)
#
# This is the custom __call__ method
#
def __datamanager_call__(self, _name=None, args=[], **kwds):
    if _name is None:
        return self
    _name=str(_name)
    if _name in IDataManager._factory_active:
        dm = PluginFactory(IDataManager._factory_cls[_name], args, **kwds)
        if not dm.available():
            raise PluginError("Cannot process data in %s files.  The following python packages need to be installed: %s" % (_name, dm.requirements()))
    else:
        dm = UnknownDataManager(type=_name)
    return dm
#
# Adding the the custom __call__ method to DataManagerFactory
#
pyutilib.misc.add_method(DataManagerFactory, __datamanager_call__, name='__call__')

