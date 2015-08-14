#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
from pyutilib.pyro import using_pyro4
from pyutilib.enum import Enum

import six

SPPyroScenarioTreeServer_ProcessTaskError = \
    "SPPyroScenarioTreeServer_ProcessTaskError"

WorkerInitType = Enum('Scenario',
                      'ScenarioList',
                      'ScenarioBundle',
                      'ScenarioBundleList')

InvocationType = Enum('SingleInvocation',
                      'PerBundleInvocation',
                      'PerBundleChainedInvocation',
                      'PerScenarioInvocation',
                      'PerScenarioChainedInvocation',
                      'PerNodeInvocation',
                      'PerNodeChainedInvocation')
