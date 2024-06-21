#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.collections import ComponentSet
from pyomo.core import Var

def var_component_set(x):
    """
    For domain validation in ConfigDicts: Takes singletone or iterable argument 'x'
    of Vars and converts it to a ComponentSet of Vars.
    """
    if hasattr(x, 'ctype') and x.ctype is Var:
        if not x.is_indexed():
            return ComponentSet([x])
        ans = ComponentSet()
        for j in x.index_set():
            ans.add(x[j])
        return ans
    elif hasattr(x, '__iter__'):
        ans = ComponentSet()
        for i in x:
            ans.update(var_component_set(i))
        return ans
    else:
        raise ValueError("Expected Var or iterable of Vars.\n\tReceived %s" % type(x))
