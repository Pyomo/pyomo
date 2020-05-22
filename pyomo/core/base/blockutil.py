#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# the purpose of this file is to collect all utility methods that compute
# attributes of blocks, based on their contents. 

__all__ = ['has_discrete_variables']

from pyomo.common import deprecated
from pyomo.core.base import Var


@deprecated("This function has been moved to `pyomo.util.blockutil`",
            version='5.6.9')
def has_discrete_variables(block):
    from pyomo.util.blockutil import has_discrete_variables
    return has_discrete_variables(block)
