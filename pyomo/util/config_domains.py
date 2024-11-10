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


class ComponentDataList:
    """ComponentDataList(ctype)
    Domain validation class that accepts singleton or iterable arguments and
    compiles them into a ComponentSet, verifying that they are all ComponentDatas
    of type 'ctype.'

    Parameters
    ----------
        ctype: The component type of the list

    Raises
    ------
        ValueError if all of the arguments are not of type 'ctype'
    """

    def __init__(self, ctype):
        self._ctype = ctype

    def __call__(self, x):
        if hasattr(x, 'ctype') and x.ctype is self._ctype:
            if not x.is_indexed():
                return ComponentSet([x])
            ans = ComponentSet()
            for j in x.index_set():
                ans.add(x[j])
            return ans
        elif hasattr(x, '__iter__'):
            ans = ComponentSet()
            for i in x:
                ans.update(self(i))
            return ans
        else:
            _ctype_name = str(self._ctype)
            raise ValueError(
                f"Expected {_ctype_name} or iterable of "
                f"{_ctype_name}s.\n\tReceived {type(x)}"
            )

    def domain_name(self):
        _ctype_name = str(self._ctype)
        return f'ComponentDataList({_ctype_name})'
