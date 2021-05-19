#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import re

literals = '()[],.'

re_number = re.compile(
    r'(?:[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|-?inf|nan)')

def name_repr(x, unknown_handler=str):
    if not isinstance(x, str):
        return _repr_map.get(x.__class__, unknown_handler)(x)
    else:
        x = repr(x)
        if x[1] == '|':
            return x
        if any(_ in x for _ in ('\\' + literals)):
            return x
        if re_number.fullmatch(x[1:-1]):
            return x
        return x[1:-1]

def tuple_repr(x, unknown_handler=str):
    return '(' + ','.join(name_repr(_, unknown_handler) for _ in x) \
        + (',)' if len(x) == 1 else ')')

def index_repr(idx, unknown_handler=str):
    """
    Return a string representation of an index.
    """
    if idx.__class__ is tuple and len(idx) > 1:
        idx_str = ",".join(name_repr(i, unknown_handler) for i in idx)
    else:
        idx_str = name_repr(idx, unknown_handler)
    return "[" + idx_str + "]"

_repr_map = {
    slice: lambda x: '*',
    Ellipsis.__class__: lambda x: '**',
    int: repr,
    float: repr,
    str: repr,
    # Note: the function is unbound at this point; extract with __func__
    tuple: tuple_repr,
}
