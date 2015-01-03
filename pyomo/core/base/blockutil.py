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

from six import itervalues, iteritems
from pyomo.core.base import Var
from pyomo.core.base.block import active_components_data


def has_discrete_variables(block):
    # Return True if there is a discrete variable in this block or any
    # sub-block.
    for block in block.all_blocks(active=True):
        for vardata in active_components_data(block, Var):
            if not var.is_continuous():
                return True
    return False
