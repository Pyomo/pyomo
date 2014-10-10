#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

from pyutilib.enum import Enum

from pyomo.core.base.plugin import *

def predefined_sets():
    from pyomo.core.base.set_types import _virtual_sets
    ans = []
    for item in _virtual_sets:
        ans.append( (item.name,item.doc) )
    return ans


def model_components():
    return [(name,ModelComponentFactory.doc(name)) for name in ModelComponentFactory.services()]
