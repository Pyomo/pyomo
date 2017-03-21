#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core.kernel.expr_coopr3 import *
from pyomo.core.base.var import _VarData, Var

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
            if type(_sub) in native_types:
                pass
            elif _sub.is_expression():
                _stack.append(( _argList, _idx, _len ))
                if type(_sub) is _ProductExpression:
                    if _sub._denominator:
                        _stack.append(
                            (_sub._denominator, 0, len(_sub._denominator)) )
                    _argList = _sub._numerator
                else:
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

class _GetItemExpression(_ExpressionBase):
    __slots__ = ('_base',)

    def __init__(self, base, args):
        """Construct a call to an external function"""
        _ExpressionBase.__init__(self, args)
        self._base = base

    def __getstate__(self):
        result = _ExpressionBase.__getstate__(self)
        for i in _GetItemExpression.__slots__:
            result[i] = getattr(self, i)
        return result

    def getname(self, *args, **kwds):
        return self._base.getname(*args, **kwds)

    def polynomial_degree(self):
        return 0 if self.is_fixed() else 1

    def is_constant(self):
        return False

    def is_fixed(self):
        return not isinstance(self._base, Var)

    def _apply_operation(self, values):
        return value(self._base[values])

    def resolve_template(self):
        return self._base.__getitem__(tuple(value(i) for i in self._args))
