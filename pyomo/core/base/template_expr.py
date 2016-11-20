#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core.base.numvalue import (
    NumericValue, native_numeric_types, as_numeric, value )
import pyomo.core.base
import logging

class TemplateExpressionError(ValueError):
    def __init__(self, template, *args, **kwds):
        self.template = template
        super(TemplateExpressionError, self).__init__(*args, **kwds)


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

    def __deepcopy__(self, memo):
        # Because we leverage deepcopy for expression cloning, we need
        # to see if this is a clone operation and *not* copy the
        # template.
        #
        # TODO: JDS: We should consider converting the IndexTemplate to
        # a proper Component: that way it could leverage the normal
        # logic of using the parent_block scope to dictate the behavior
        # of deepcopy.
        if '__block_scope__' in memo:
            memo[id(self)] = self
            return self
        #
        # "Normal" deepcopying outside the context of pyomo.
        #
        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        ans.__setstate__(deepcopy(self.__getstate__(), memo))
        return ans

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is None:
            raise TemplateExpressionError(self)
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

    def __str__(self):
        return self.getname()

    def getname(self, fully_qualified=False, name_buffer=None):
        return "{"+self._set.getname(fully_qualified, name_buffer)+"}"

    def to_string(self, ostream=None, verbose=None, precedence=0):
        if ostream is None:
            ostream = sys.stdout
        ostream.write( self.name )

    def set_value(self, value):
        # It might be nice to check if the value is valid for the base
        # set, but things are tricky when the base set is not dimention
        # 1.  So, for the time being, we will just "trust" the user.
        self._value = value


def substitute_template_expression(expr, substituter, *args):
    """Substitute IndexTemplates in an expression tree.

    This is a general utility function for walking the expression tree
    ans subtituting all occurances of IndexTemplate and
    _GetItemExpression nodes.  The routine is a general expression
    walker for both Coopr3 / Pyomo4 expressions.  This borrows from
    pseudo-visitor pattern to defer the actual substitution to the
    substituter function / arguments passed to this method.

    Args:
        substituter: method taking (expression, *args) and returning 
           the new object
        *args: these are passed directly to the substituter

    Returns:
        a new expression tree with all substitutions done
    """
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
            if _subType is EXPR._GetItemExpression or _subType is IndexTemplate:
                if type(_ptr[0]) is tuple:
                    _list = list(_ptr[0])
                    _list[_ptr[1]-1] = substituter(_obj, *args)
                    _ptr[0] = tuple(_list)
                    _ptr[3]._args = _list
                else:
                    _ptr[0][_ptr[1]-1] = substituter(_obj, *args)
            elif _subType in native_numeric_types or not _obj.is_expression():
                continue
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


class _GetItemIndexer(object):
    # Note that this class makes the assumption that only one template
    # ever appears in an expression for a single index
    def __init__(self, expr):
        self._base = expr._base
        self._args = []
        _hash = [ id(self._base) ]
        for x in expr._args:
            try:
                active_level = logging.root.manager.disable
                logging.disable(logging.CRITICAL)
                val = value(x)
                self._args.append(val)
                _hash.append(val)
            except TemplateExpressionError as e:
                if x is not e.template:
                    raise TypeError(
                        "Cannot use the param substituter with expression "
                        "templates\nwhere the component index has the "
                        "IndexTemplate in an expression.\n\tFound in %s"
                        % ( expr, ))
                self._args.append(e.template)
                _hash.append(id(e.template._set))
            finally:
                logging.disable(active_level)

        self._hash = tuple(_hash)

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if type(other) is _GetItemIndexer:
            return self._hash == other._hash
        else:
            return False

def substitute_getitem_with_param(expr, _map):
    """A simple substituter to replace _GetItem nodes with mutable Params.

    This substituter will replace all _GetItemExpression nodes with a
    new Param.  For example, this method will create expressions
    suitable for passing to DAE integrators
    """

    if type(expr) is IndexTemplate:
        return expr

    _id = _GetItemIndexer(expr)
    if _id not in _map:
        _map[_id] = pyomo.core.base.param.Param(mutable=True)
        _map[_id].construct()
        _args = []
        _map[_id]._name = "%s[%s]" % (
            expr._base.name, ','.join(str(x) for x in _id._args) )
    return _map[_id]


def substitute_template_with_value(expr):
    """A simple substituter to expand expression for current template

    This substituter will replace all _GetItemExpression / IndexTemplate
    nodes with the actual _ComponentData based on the current value of
    the IndexTamplate(s)

    """

    if type(expr) is IndexTemplate:
        return as_numeric(expr())
    else:
        return expr.resolve_template()
