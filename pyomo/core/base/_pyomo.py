#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base.plugin import *

def predefined_sets():
    from pyomo.core.base.set_types import _virtual_sets
    ans = []
    for item in _virtual_sets:
        ans.append( (item.name, item.doc) )
    return ans


def model_components():
    return [(name,ModelComponentFactory.doc(name)) for name in ModelComponentFactory]
