#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['IPreCheckHook', 'IPostCheckHook']

from pyomo.util.plugin import *

class IPreCheckHook(Interface):

    def precheck(self, runner, script, info):
        pass


class IPostCheckHook(Interface):

    def postcheck(self, runner, script, info):
        pass
