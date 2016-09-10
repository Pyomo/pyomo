#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core.base.numvalue import NumericValue, native_numeric_types
import pyomo.core.base

class TemplateExpressionError(ValueError):
    pass


class IndexTemplate(NumericValue):
    """This class can be used to greate "template expressions"

    Constructor Arguments:
    """

    __slots__ = ('_set', '_value')

    def __init__(self, _set):
        self._set = _set
        self._value = None

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(IndexTemplate, self).__getstate__()
        for i in IndexTemplate.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is None:
            raise TemplateExpressionError()
        else:
            return self._value

    def is_fixed(self):
        """
        Returns True because this value is fixed.
        """
        return True

    def is_constant(self):
        """
        Returns False because this cannot immediately be simplified.
        """
        return False

    def to_string(self, ostream=None, verbose=None, precedence=0):
        if ostream is None:
            ostream = sys.stdout
        ostream.write("{"+self._set.name(True)+"}")


def substitute_template_expression(expr, substituter, *args):
    # Again, due to circular imports, we cannot import expr at the
    # module scope because this module gets imported by expr
    from pyomo.core.base import expr as EXPR
    from pyomo.core.base import expr_common as common

    _stack = [ [[expr.clone()], 0, 1, None] ]
    _stack_idx = 0
    while _stack_idx >= 0:
        _ptr = _stack[_stack_idx]
        while _ptr[1] < _ptr[2]:
            _obj = _ptr[0][_ptr[1]]
            _ptr[1] += 1            
            _subType = type(_obj)
            if _subType in native_numeric_types or not _obj.is_expression():
                continue
            if _subType is EXPR._GetItemExpression:
                if type(_ptr[0]) is tuple:
                    _list = list(_ptr[0])
                    _list[_ptr[1]-1] = substituter(_obj, *args)
                    _ptr[0] = tuple(_list)
                    _ptr[3]._args = _list
                else:
                    _ptr[0][_ptr[1]-1] = substituter(_obj, *args)
            elif _subType is EXPR._ProductExpression:
                # _ProductExpression is fundamentally different in
                # Coopr3 / Pyomo4 expression systems and must be handled
                # specially.
                if common.mode is common.Mode.coopr3_trees:
                    _lists = (_obj._numerator, _obj._denominator)
                else:
                    _lists = (_obj._args,)
                for _list in _lists:
                    if not _list:
                        continue
                    _stack_idx += 1
                    _ptr = [_list, 0, len(_list), _obj]
                    if _stack_idx < len(_stack):
                        _stack[_stack_idx] = _ptr
                    else:
                        _stack.append( _ptr )
            else:
                if not _obj._args:
                    continue
                _stack_idx += 1
                _ptr = [_obj._args, 0, len(_obj._args), _obj]
                if _stack_idx < len(_stack):
                    _stack[_stack_idx] = _ptr
                else:
                    _stack.append( _ptr )
        _stack_idx -= 1
    return _stack[0][0][0]

def substitute_template_with_param(expr, _map):
    _id = id(expr._base)
    if _id not in _map:
        _map[_id] = pyomo.core.base.param.Param(mutable=True)
        _map[_id].construct()
        _map[_id]._name = expr._base.name()
    return _map[_id]


def substitute_template_with_index(expr, _map):
    return expr.resolve_template()
    
