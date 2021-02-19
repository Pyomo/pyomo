#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['IPreCheckHook', 'IPostCheckHook']

from pyomo.common.plugin import Interface

class IPreCheckHook(Interface):

    def precheck(self, runner, script, info):
        pass


class IPostCheckHook(Interface):

    def postcheck(self, runner, script, info):
        pass
