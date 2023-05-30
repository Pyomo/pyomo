#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import re

# Literals are used in parsing string names (and indicate tuples,
# indexing, and token separators)
literals = '()[],.'
# Special characters are additional characters that if they appear in
# the string force us to quote the string.  This includes the obvious
# things like single and double quote characters, but also backslash
# (indicates that the string contains escaped - possibly unicode -
# characters), and the colon (used as a token separator in the old
# ComponentUID "v1" format).
special_chars = literals + '\'":\\'

re_number = re.compile(
    r'(?:[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|-?inf|nan)'
)
re_special_char = re.compile(r'[' + re.escape(special_chars) + ']')


def name_repr(x, unknown_handler=str):
    if not isinstance(x, str):
        return _repr_map.get(x.__class__, unknown_handler)(x)
    else:
        x = repr(x)
        if x[1] == '|':
            return x
        unquoted = x[1:-1]
        if re_special_char.search(unquoted):
            return x
        if re_number.fullmatch(unquoted):
            return x
        return unquoted


def tuple_repr(x, unknown_handler=str):
    return (
        '('
        + ','.join(name_repr(_, unknown_handler) for _ in x)
        + (',)' if len(x) == 1 else ')')
    )


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
