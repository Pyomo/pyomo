#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

from pyomo.core.kernel.expr_pyomo4 import *
from pyomo.core.base.var import Var, _VarData

def identify_variables( expr,
                        include_fixed=True,
                        allow_duplicates=False,
                        include_potentially_variable=False ):
    if not allow_duplicates:
        _seen = set()
    _stack = [ ([expr], 0, 1) ]
    while _stack:
        _argList, _idx, _len = _stack.pop()
        while _idx < _len:
            _sub = _argList[_idx]
            _idx += 1
            if _sub.__class__ in native_types:
                pass
            elif _sub.is_expression():
                _stack.append(( _argList, _idx, _len ))
                _argList = _sub._args
                _idx = 0
                _len = len(_argList)
            elif isinstance(_sub, _VarData):
                if ( include_fixed
                     or not _sub.is_fixed()
                     or include_potentially_variable ):
                    if not allow_duplicates:
                        if id(_sub) in _seen:
                            continue
                        _seen.add(id(_sub))
                    yield _sub
            elif include_potentially_variable and _sub._potentially_variable():
                if not allow_duplicates:
                    if id(_sub) in _seen:
                        continue
                    _seen.add(id(_sub))
                yield _sub
