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
from pyomo.common import Factory

logger = logging.getLogger('pyomo.core')


class UnknownDataManager(object):

    def __init__(self, *args, **kwds):
        #
        # The 'type' is the class type of the solver instance
        #
        self.type = kwds["type"]

    def available(self):
        return False


class DataManagerFactoryClass(Factory):

    #
    # This is the custom __call__ method
    #
    def __call__(self, _name=None, args=[], **kwds):
        if _name is None:
            return self
        _name=str(_name)
        if _name in self._cls:
            dm = self._cls[_name](**kwds)
            if not dm.available():
                raise PluginError("Cannot process data in %s files.  The following python packages need to be installed: %s" % (_name, dm.requirements()))
        else:
            dm = UnknownDataManager(type=_name)
        return dm

DataManagerFactory = DataManagerFactoryClass('data file')

