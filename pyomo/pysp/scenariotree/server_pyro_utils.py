#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("WorkerInitType",
           "WorkerInit",
           "ScenarioWorkerInit",
           "BundleWorkerInit")

import logging
from collections import namedtuple

import pyutilib.enum

from six import string_types

logger = logging.getLogger('pyomo.pysp')

#
# Controls what scenario tree objects are created when instantiating
# a new worker on a scenario tree server:
#  - Scenarios: Worker manages one or more scenarios.
#  - Bundles: Worker manages one or more scenario bundles.
#
WorkerInitType = pyutilib.enum.Enum('Scenarios',
                                    'Bundles')

#
# A named tuple that groups together the information required to
# initialize a new worker on a scenario tree server:
#  - type_: a selection from the WorkerInitType enumeration
#  - names: A list of names for the scenario tree objects
#           that will be initialized on the worker. The names
#           should represent scenarios or bundles depending on
#           the choice of WorkerInitType.
#  - data: The data associated with choice of WorkerInitType.
#          For 'Scenarios', this should be None. For 'Bundles'
#          this should be a dictionary mapping bundle name to
#          a list of scenario names.
#
WorkerInit = namedtuple('WorkerInit',
                        ['type_', 'names', 'data'])

#
# A convenience function for populating a WorkerInit tuple
# for scenario worker initializations. If initializing a single
# scenario, arg should be a scenario name. If initializing a list
# of scenarios, arg should be a list or tuple of scenario names.
#
def ScenarioWorkerInit(arg):
    if isinstance(arg, string_types):
        return WorkerInit(type_=WorkerInitType.Scenarios,
                          names=(arg,),
                          data=None)
    else:
        assert type(arg) in (list, tuple)
        for name in arg:
            assert isinstance(name, string_types)
        return WorkerInit(type_=WorkerInitType.Scenarios,
                          names=arg,
                          data=None)

#
# A convenience function for populating a WorkerInit tuple
# for bundle worker initializations. If initializing a single
# bundle, arg should be the bundle name and data should be a
# list or tuple. If initializing a list of bundles, arg should
# a list or tuple of bundle names, and data should be a dict
# mapping bundle name to a list or tuple of scenarios.
#
def BundleWorkerInit(arg, data):
    if isinstance(arg, string_types):
        assert type(data) in (list, tuple)
        assert len(data) > 0
        return WorkerInit(type_=WorkerInitType.Bundles,
                          names=(arg,),
                          data={arg: data})
    else:
        assert type(arg) in (list, tuple)
        assert type(data) is dict
        for name in arg:
            assert isinstance(name, string_types)
            assert type(data[name]) in (list, tuple)
            assert len(data[name]) > 0
        return WorkerInit(type_=WorkerInitType.Bundles,
                          names=arg,
                          data=data)

