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
from six import iteritems, itervalues

from pyomo.core.expr.expr_errors import TemplateExpressionError
from pyomo.core.expr.numvalue import (
    NumericValue, native_numeric_types, native_types, nonpyomo_leaf_types,
    as_numeric, value,
)
from pyomo.core.expr.numeric_expr import ExpressionBase
from pyomo.core.expr.visitor import (
    ExpressionReplacementVisitor, StreamBasedExpressionVisitor
)

class _NotSpecified(object): pass

class GetItemExpression(ExpressionBase):
    """
    Expression to call :func:`__getitem__` on the base object.
    """
    __slots__ = ('_base',)
    PRECEDENCE = 1

    def _precedence(self):
        return GetItemExpression.PRECEDENCE

    def __init__(self, args, base=None):
        """Construct an expression with an operation and a set of arguments"""
        self._args_ = args
        self._base = base

    def nargs(self):
        return len(self._args_)

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._base)

    def __getstate__(self):
        state = super(GetItemExpression, self).__getstate__()
        for i in GetItemExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError()
        return GetAttrExpression((self, attr))

    def getname(self, *args, **kwds):
        return self._base.getname(*args, **kwds)

    def is_potentially_variable(self):
        if any(arg.is_potentially_variable() for arg in self._args_
               if arg.__class__ not in nonpyomo_leaf_types):
            return True
        for x in itervalues(self._base._data):
            if hasattr(x, 'is_potentially_variable') and \
               x.is_potentially_variable():
                return True
        return False

    def _is_fixed(self, values):
        if not all(values):
            return False
        for x in itervalues(self._base):
            if hasattr(x, 'is_fixed') and not x.is_fixed():
                return False
        return True

    def _compute_polynomial_degree(self, result):
        if any(x != 0 for x in result):
            return None
        ans = 0
        for x in itervalues(self._base):
            if x.__class__ in nonpyomo_leaf_types:
                continue
            tmp = x.polynomial_degree()
            if tmp is None:
                return None
            elif tmp > ans:
                ans = tmp
        return ans

    def _apply_operation(self, result):
        obj = self._base.__getitem__( tuple(result) )
        if isinstance(obj, NumericValue):
            obj = value(obj)
        return obj

    def _to_string(self, values, verbose, smap, compute_values):
        values = tuple(_[1:-1] if _[0]=='(' and _[-1]==')' else _
                       for _ in values)
        if verbose:
            return "getitem(%s, %s)" % (self.getname(), ', '.join(values))
        return "%s[%s]" % (self.getname(), ','.join(values))

    def _resolve_template(self, args):
        return self._base.__getitem__(args)


class GetAttrExpression(ExpressionBase):
    """
    Expression to call :func:`__getattr__` on the base object.
    """
    __slots__ = ()
    PRECEDENCE = 1

    def _precedence(self):
        return GetAttrExpression.PRECEDENCE

    def nargs(self):
        return len(self._args_)

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError()
        return GetAttrExpression((self, attr))

    def __getitem__(self, *idx):
        return GetItemExpression(idx, base=self)

    def getname(self, *args, **kwds):
        return 'getattr'

    def _compute_polynomial_degree(self, result):
        if result[1] != 0:
            return None
        return result[0]

    def _apply_operation(self, result):
        assert len(result) == 2
        obj = getattr(result[0], result[1])
        if isinstance(obj, NumericValue):
            obj = value(obj)
        return obj

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "getitem(%s, %s)" % values
        return "%s.%s" % values

    def _resolve_template(self, args):
        return getattr(*args)


class TemplateSumExpression(ExpressionBase):
    """
    Expression to represent an unexpanded sum over one or more sets.
    """
    __slots__ = ('_iters',)
    PRECEDENCE = 1

    def _precedence(self):
        return TemplateSumExpression.PRECEDENCE

    def __init__(self, args, _iters):
        assert len(args) == 1
        self._args_ = args
        self._iters = _iters

    def nargs(self):
        return 1

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._iters)

    def __getstate__(self):
        state = super(TemplateSumExpression, self).__getstate__()
        for i in GetItemExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return "SUM"

    def is_potentially_variable(self):
        if any(arg.is_potentially_variable() for arg in self._args_
               if arg.__class__ not in nonpyomo_leaf_types):
            return True
        return False

    def _is_fixed(self, values):
        return all(values)

    def _compute_polynomial_degree(self, result):
        if None in result:
            return None
        return result[0]

    def _set_iter_vals(vals):
        for i, iterGroup in enumerate(self._iters):
            if len(iterGroup) == 1:
                iterGroup[0].set_value(val[i])
            else:
                for j, v in enumerate(val):
                    iterGroup[j].set_value(v)

    def _get_iter_vals(vals):
        return tuple(tuple(x._value for x in ig) for ig in self._iters)

    def _apply_operation(self, result):
        ans = 0
        _init_vals = self._get_iter_vals()
        _sets = tuple(iterGroup[0]._set for iterGroup in self._iters)
        for val in itertools.product(*_sets):
            self._set_iter_vals(val)
            ans += value(self._args_[0])
        self._set_iter_vals(_init_vals)
        return ans

    def _to_string(self, values, verbose, smap, compute_values):
        ans = ''
        for iterGroup in self._iters:
            ans += ' for %s in %s' % (','.join(str(i) for i in iterGroup),
                                      iterGroup[0]._set)
        val = values[0]
        if val[0]=='(' and val[-1]==')':
            val = val[1:-1]
        return "SUM(%s%s)" % (val, ans)

    def _resolve_template(self, args):
        _init_vals = self._get_iter_vals()
        _sets = tuple(iterGroup[0]._set for iterGroup in self._iters)
        ans = []
        for val in itertools.product(*_sets):
            self._set_iter_vals(val)
            ans.append(resolve_template(self._args_[0]))
        self._set_iter_vals(_init_vals)
        return SumExpression(ans)


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

    __slots__ = ('_set', '_value', '_index', '_id')

    def __init__(self, _set, index=0, _id=None):
        self._set = _set
        self._value = _NotSpecified
        self._index = index
        self._id = _id

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
        if self._value is _NotSpecified:
            if exception:
                raise TemplateExpressionError(
                    self, "Evaluating uninitialized IndexTemplate")
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
        if self._id is not None:
            return "_%s" % (self._id,)

        _set_name = self._set.getname(fully_qualified, name_buffer, relative_to)
        if self._index is not None and self._set.dimen != 1:
            _set_name += "(%s)" % (self._index,)
        return "{"+_set_name+"}"

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        return self.name

    def set_value(self, *values):
        # It might be nice to check if the value is valid for the base
        # set, but things are tricky when the base set is not dimention
        # 1.  So, for the time being, we will just "trust" the user.
        # After all, the actual Set will raise exceptions if the value
        # is not present.
        if not values:
            self._value = _NotSpecified
        elif self._index is not None:
            if len(values) == 1:
                self._value = values[0]
            else:
                raise ValueError("Passed multiple values %s to a scalar "
                                 "IndexTemplate %s" % (values, self))
        else:
            self._value = values


def resolve_template(expr):
    """Resolve a template into a concrete expression

    This takes a template expression and returns the concrete equivalent
    by substituting the current values of all IndexTemplate objects and
    resolving (evaluating and removing) all GetItemExpression,
    GetAttrExpression, and TemplateSumExpression expression nodes.

    """
    def beforeChild(node, child):
        # Efficiency: do not decend into leaf nodes.
        if type(child) in native_types or not child.is_expression_type():
            return False, child
        else:
            return True, None

    def exitNode(node, args):
        if hasattr(node, '_resolve_template'):
            return node._resolve_template(args)
        if len(args) == node.nargs() and all(
                a is b for a,b in zip(node.args, args)):
            return node
        return node.create_node_with_local_data(args)

    return StreamBasedExpressionVisitor(
        beforeChild=beforeChild,
        exitNode=exitNode,
    ).walk_expression(expr)


class ReplaceTemplateExpression(ExpressionReplacementVisitor):

    def __init__(self, substituter, *args):
        super(ReplaceTemplateExpression, self).__init__()
        self.substituter = substituter
        self.substituter_args = args

    def visiting_potential_leaf(self, node):
        if type(node) is GetItemExpression or type(node) is IndexTemplate:
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
    import pyomo.core.base.param
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
    the IndexTemplate(s)

    """

    if type(expr) is IndexTemplate:
        return as_numeric(expr())
    else:
        return resolve_template(expr)



class mock_globals(object):
    """Implement custom context for a user-specified function.

    This class implements a custom context that injects user-specified
    attributes into the globals() context before calling a function (and
    then cleans up the global context after the function returns).

    Parameters
    ----------
        fcn : function
            The function whose globals context will be overridden
        overrides : dict
            A dict mapping {name: object} that will be injected into the
            `fcn` globals() context.
    """
    __slots__ = ('_data',)

    def __init__(self, fcn, overrides):
        self._data = fcn, overrides

    def __call__(self, *args, **kwds):
        fcn, overrides = self._data
        _old = {}
        try:
            for name, val in iteritems(overrides):
                if name in fcn.__globals__:
                    _old[name] = fcn.__globals__[name]
            fcn.__globals__[name] = val

            return fcn(*args, **kwds)
        finally:
            for name, val in iteritems(overrides):
                if name in _old:
                    fcn.__globals__[name] = _old[name]
                else:
                    del fcn.__globals__[name]


class _set_iterator_template_generator(object):
    """Replacement iterator that returns IndexTemplates

    In order to generate template expressions, we hijack the normal Set
    iteration mechanisms so that this iterator is returned instead of
    the usual iterator.  This iterator will return IndexTemplate
    object(s) instead of the actual Set items the first time next() is
    called.
    """
    def __init__(self, _set, context):
        self._set = _set
        self.context = context

    def __iter__(self):
        return self

    def __next__(self):
        # Prevent context from ever being called more than once
        if self.context is None:
            raise StopIteration()
        context, self.context = self.context, None

        _set = self._set
        d = _set.dimen
        if d is None:
            idx = (IndexTemplate(_set, None, context.next_id()),)
        else:
            idx = tuple(
                IndexTemplate(_set, i, context.next_id()) for i in range(d)
            )
        context.cache.append(idx)
        if len(idx) == 1:
            return idx[0]
        else:
            return idx

    next = __next__

class _template_iter_context(object):
    """Manage the iteration context when generating templatized rules

    This class manages the context tracking when generating templatized
    rules.  It has two methods (`sum_template` and `get_iter`) that
    replace standard functions / methods (`sum` and
    :py:meth:`_FiniteSetMixin.__iter__`, respectively).  It also tracks
    unique identifiers for IndexTemplate objects and their groupings
    within `sum()` generators.
    """
    def __init__(self):
        self.cache = []
        self._id = 0

    def get_iter(self, _set):
        return _set_iterator_template_generator(_set, self)

    def npop_cache(self, n):
        result = self.cache[-n:]
        self.cache[-n:] = []
        return result

    def next_id(self):
        self._id += 1
        return self._id

    def sum_template(self, generator):
        init_cache = len(self.cache)
        expr = next(generator)
        final_cache = len(self.cache)
        return TemplateSumExpression(
            (expr,), self.npop_cache(final_cache-init_cache)
        )

def templatize_rule(block, rule, index_set):
    import pyomo.core.base.set
    context = _template_iter_context()
    try:
        # Override Set iteration to return IndexTemplates
        _old_iter = pyomo.core.base.set._FiniteSetMixin.__iter__
        pyomo.core.base.set._FiniteSetMixin.__iter__ = \
            lambda x: context.get_iter(x)
        # Override sum with our sum
        _old_sum = __builtins__['sum']
        __builtins__['sum'] = context.sum_template
        # Get the index templates needed for calling the rule
        if index_set is not None:
            if not index_set.isfinite():
                raise TemplateExpressionError(
                    None,
                    "Cannot templatize rule with non-finite indexing set")
            indices = iter(index_set).next()
            context.cache.pop()
        else:
            indices = ()
        if type(indices) is not tuple:
            indices = (indices,)
        # Call the rule, returning the template expression and the
        # top-level IndexTemplaed generated when calling the rule.
        #
        # TBD: Should this just return a "FORALL()" expression node that
        # behaves similarly to the GetItemExpression node?
        return rule(block, *indices), indices
    finally:
        pyomo.core.base.set._FiniteSetMixin.__iter__ = _old_iter
        __builtins__['sum'] = _old_sum
        if len(context.cache):
            raise TemplateExpressionError(
                None,
                "Explicit iteration (for loops) over Sets is not supported by "
                "template expressions.  Encountered loop over %s"
                % (context.cache[-1][0]._set,))
