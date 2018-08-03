#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import copy
import logging
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import (
    NumericValue, native_numeric_types, as_numeric, value )
import pyomo.core.base
from pyomo.core.expr.expr_errors import TemplateExpressionError

class IndexTemplate(NumericValue):
    """A "placeholder" for an index value in template expressions.

    This class is a placeholder for an index value within a template
    expression.  That is, given the expression template for "m.x[i]",
    where `m.z` is indexed by `m.I`, the expression tree becomes:

    _GetItem:
       - m.x
       - IndexTemplate(_set=m.I, _value=None)

    Constructor Arguments:
       _set: the Set from which this IndexTemplate can take values
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
        ans.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return ans

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is None:
            if exception:
                raise TemplateExpressionError(self)
            return None
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

    def is_potentially_variable(self):
        """Returns False because index values cannot be variables.

        The IndexTemplate represents a placeholder for an index value
        for an IndexedComponent, and at the moment, Pyomo does not
        support variable indirection.
        """
        return False

    def __str__(self):
        return self.getname()

    def getname(self, fully_qualified=False, name_buffer=None, relative_to=None):
        return "{"+self._set.getname(fully_qualified, name_buffer, relative_to)+"}"

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        return self.name

    def set_value(self, value):
        # It might be nice to check if the value is valid for the base
        # set, but things are tricky when the base set is not dimention
        # 1.  So, for the time being, we will just "trust" the user.
        self._value = value


class ReplaceTemplateExpression(EXPR.ExpressionReplacementVisitor):

    def __init__(self, substituter, *args):
        super(ReplaceTemplateExpression, self).__init__()
        self.substituter = substituter
        self.substituter_args = args

    def visiting_potential_leaf(self, node):
        if type(node) is EXPR.GetItemExpression or type(node) is IndexTemplate:
            return True, self.substituter(node, *self.substituter_args)

        return super(
            ReplaceTemplateExpression, self).visiting_potential_leaf(node)


def substitute_template_expression(expr, substituter, *args):
    """Substitute IndexTemplates in an expression tree.

    This is a general utility function for walking the expression tree
    and subtituting all occurances of IndexTemplate and
    _GetItemExpression nodes.

    Args:
        substituter: method taking (expression, *args) and returning 
           the new object
        *args: these are passed directly to the substituter

    Returns:
        a new expression tree with all substitutions done
    """
    visitor = ReplaceTemplateExpression(substituter, *args)
    return visitor.dfs_postorder_stack(expr)


class _GetItemIndexer(object):
    # Note that this class makes the assumption that only one template
    # ever appears in an expression for a single index

    def __init__(self, expr):
        self._base = expr._base
        self._args = []
        _hash = [ id(self._base) ]
        for x in expr.args:
            try:
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
                logging.disable(logging.NOTSET)

        self._hash = tuple(_hash)

    def nargs(self):
        return len(self._args)

    def arg(self, i):
        return self._args[i]

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        if type(other) is _GetItemIndexer:
            return self._hash == other._hash
        else:
            return False

    def __str__(self):
        return "%s[%s]" % (
            self._base.name, ','.join(str(x) for x in self._args) )


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
