#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# the purpose of this file is to collect all utility methods that compute
# attributes of blocks, based on their contents. 

__all__ = ['has_discrete_variables']

from pyomo.core.base import Var

def has_discrete_variables(block):
    for vardata in block.component_data_objects(Var, active=True):
        if not vardata.is_continuous():
            return True
    return False
