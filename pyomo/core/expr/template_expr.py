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
import itertools
import logging
import sys
from six import itervalues
from six.moves import builtins

from pyomo.core.expr.expr_errors import TemplateExpressionError
from pyomo.core.expr.numvalue import (
    NumericValue, native_types, nonpyomo_leaf_types,
    as_numeric, value,
)
from pyomo.core.expr.numeric_expr import ExpressionBase, SumExpression
from pyomo.core.expr.visitor import (
    ExpressionReplacementVisitor, StreamBasedExpressionVisitor
)

logger = logging.getLogger(__name__)

class _NotSpecified(object): pass

class GetItemExpression(ExpressionBase):
    """
    Expression to call :func:`__getitem__` on the base object.
    """
    PRECEDENCE = 1

    def _precedence(self):
        return GetItemExpression.PRECEDENCE

    def __init__(self, args):
        """Construct an expression with an operation and a set of arguments"""
        self._args_ = args

    def nargs(self):
        return len(self._args_)

    def __getattr__(self, attr):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError()
        return GetAttrExpression((self, attr))

    def __iter__(self):
        return iter(value(self))

    def __len__(self):
        return len(value(self))

    def getname(self, *args, **kwds):
        return self._args_[0].getname(*args, **kwds)

    def is_potentially_variable(self):
        _false = lambda: False
        if any( getattr(arg, 'is_potentially_variable', _false)()
                for arg in self._args_ ):
            return True
        base = self._args_[0]
        if base.is_expression_type():
            base = value(base)
        # TODO: fix value iteration when generating templates
        #
        # There is a nasty problem here: we want to iterate over all the
        # members of the base and see if *any* of them are potentially
        # variable.  Unfortunately, this method is called during
        # expression generation, and we *could* be generating a
        # template.  When that occurs, iterating over the base will
        # yield a new IndexTemplate (which will in turn raise an
        # exception because IndexTemplates are not constant).  The real
        # solution is probably to re-think how we define
        # is_potentially_variable, but for now we will only handle
        # members that are explicitly stored in the _data dict.  Not
        # general (because a Component could implement a non-standard
        # storage scheme), but as of now [30 Apr 20], there are no known
        # Components where this assumption will cause problems.
        return any( getattr(x, 'is_potentially_variable', _false)()
                    for x in itervalues(getattr(base, '_data', {})) )

    def _is_fixed(self, values):
        if not all(values[1:]):
            return False
        _true = lambda: True
        return all( getattr(x, 'is_fixed', _true)()
                    for x in itervalues(values[0]) )

    def _compute_polynomial_degree(self, result):
        if any(x != 0 for x in result[1:]):
            return None
        ans = 0
        for x in itervalues(result[0]):
            if x.__class__ in nonpyomo_leaf_types \
               or not hasattr(x, 'polynomial_degree'):
                continue
            tmp = x.polynomial_degree()
            if tmp is None:
                return None
            elif tmp > ans:
                ans = tmp
        return ans

    def _apply_operation(self, result):
        obj = result[0].__getitem__( tuple(result[1:]) )
        if obj.__class__ in nonpyomo_leaf_types:
            return obj
        # Note that because it is possible (likely) that the result
        # could be an IndexedComponent_slice object, must test "is
        # True", as the slice will return a list of values.
        if obj.is_numeric_type() is True:
            obj = value(obj)
        return obj

    def _to_string(self, values, verbose, smap, compute_values):
        values = tuple(_[1:-1] if _[0]=='(' and _[-1]==')' else _
                       for _ in values)
        if verbose:
            return "getitem(%s, %s)" % (values[0], ', '.join(values[1:]))
        return "%s[%s]" % (values[0], ','.join(values[1:]))

    def _resolve_template(self, args):
        return args[0].__getitem__(tuple(args[1:]))


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
        return GetItemExpression((self,) + idx)

    def __iter__(self):
        return iter(value(self))

    def __len__(self):
        return len(value(self))

    def getname(self, *args, **kwds):
        return 'getattr'

    def _compute_polynomial_degree(self, result):
        if result[1] != 0:
            return None
        return result[0]

    def _apply_operation(self, result):
        assert len(result) == 2
        obj = getattr(result[0], result[1])
        if obj.__class__ in nonpyomo_leaf_types:
            return obj
        # Note that because it is possible (likely) that the result
        # could be an IndexedComponent_slice object, must test "is
        # True", as the slice will return a list of values.
        if obj.is_numeric_type() is True:
            obj = value(obj)
        return obj

    def _to_string(self, values, verbose, smap, compute_values):
        assert len(values) == 2
        if verbose:
            return "getattr(%s, %s)" % tuple(values)
        # Note that the string argument for getattr comes quoted, so we
        # need to remove the quotes.
        attr = values[1]
        if attr[0] in '\"\'' and attr[0] == attr[-1]:
            attr = attr[1:-1]
        return "%s.%s" % (values[0], attr)

    def _resolve_template(self, args):
        return getattr(*tuple(args))


class _TemplateSumExpression_argList(object):
    """A virtual list to represent the expanded SumExpression args

    This class implements a "virtual args list" for
    TemplateSumExpressions without actually generating the expanded
    expression.  It can be accessed either in "one-pass" without
    generating a list of template argument values (more efficient), or
    as a random-access list (where it will have to create the full list
    of argument values (less efficient).

    The instance can be used as a context manager to both lock the
    IndexTemplate values within this context and to restore their original
    values upon exit.

    It is (intentionally) not iterable.

    """
    def __init__(self, TSE):
        self._tse = TSE
        self._i = 0
        self._init_vals = None
        self._iter = self._get_iter()
        self._lock = None

    def __len__(self):
        return self._tse.nargs()

    def __getitem__(self, i):
        if self._i == i:
            self._set_iter_vals(next(self._iter))
            self._i += 1
        elif self._i is not None:
            # Switch to random-access mode.  If we have already
            # retrieved one of the indices, then we need to regenerate
            # the iterator from scratch.
            self._iter = list(self._get_iter() if self._i else self._iter)
            self._set_iter_vals(self._iter[i])
        else:
            self._set_iter_vals(self._iter[i])
        return self._tse._local_args_[0]

    def __enter__(self):
        self._lock = self
        self._lock_iters()

    def __exit__(self, exc_type, exc_value, tb):
        self._unlock_iters()
        self._lock = None

    def _get_iter(self):
        # Note: by definition, all _set pointers within an itergroup
        # point to the same Set
        _sets = tuple(iterGroup[0]._set for iterGroup in self._tse._iters)
        return itertools.product(*_sets)

    def _lock_iters(self):
        self._init_vals = tuple(
            tuple(
                it.lock(self._lock) for it in iterGroup
            ) for iterGroup in self._tse._iters )

    def _unlock_iters(self):
        self._set_iter_vals(self._init_vals)
        for iterGroup in self._tse._iters:
            for it in iterGroup:
                it.unlock(self._lock)

    def _set_iter_vals(self, val):
        for i, iterGroup in enumerate(self._tse._iters):
            if len(iterGroup) == 1:
                iterGroup[0].set_value(val[i], self._lock)
            else:
                for j, v in enumerate(val[i]):
                    iterGroup[j].set_value(v, self._lock)


class TemplateSumExpression(ExpressionBase):
    """
    Expression to represent an unexpanded sum over one or more sets.
    """
    __slots__ = ('_iters', '_local_args_')
    PRECEDENCE = 1

    def _precedence(self):
        return TemplateSumExpression.PRECEDENCE

    def __init__(self, args, _iters):
        assert len(args) == 1
        self._args_ = args
        self._iters = _iters

    def nargs(self):
        # Note: by definition, all _set pointers within an itergroup
        # point to the same Set
        ans = 1
        for iterGroup in self._iters:
            ans *= len(iterGroup[0]._set)
        return ans

    @property
    def args(self):
        return _TemplateSumExpression_argList(self)

    @property
    def _args_(self):
        return _TemplateSumExpression_argList(self)

    @_args_.setter
    def _args_(self, args):
        self._local_args_ = args

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._iters)

    def __getstate__(self):
        state = super(TemplateSumExpression, self).__getstate__()
        for i in TemplateSumExpression.__slots__:
            state[i] = getattr(self, i)
        return state

    def getname(self, *args, **kwds):
        return "SUM"

    def is_potentially_variable(self):
        if any(arg.is_potentially_variable() for arg in self._local_args_
               if arg.__class__ not in nonpyomo_leaf_types):
            return True
        return False

    def _is_fixed(self, values):
        return all(values)

    def _compute_polynomial_degree(self, result):
        if None in result:
            return None
        return result[0]

    def _apply_operation(self, result):
        return sum(result)

    def _to_string(self, values, verbose, smap, compute_values):
        ans = ''
        val = values[0]
        if val[0]=='(' and val[-1]==')' and _balanced_parens(val[1:-1]):
            val = val[1:-1]
        iterStrGenerator = (
            ( ', '.join(str(i) for i in iterGroup),
              iterGroup[0]._set.to_string(verbose=verbose) )
            for iterGroup in self._iters
        )
        if verbose:
            iterStr = ', '.join('iter(%s, %s)' % x for x in iterStrGenerator)
            return 'templatesum(%s, %s)' % (val, iterStr)
        else:
            iterStr = ' '.join('for %s in %s' % x for x in iterStrGenerator)
            return 'SUM(%s %s)' % (val, iterStr)

    def _resolve_template(self, args):
        return SumExpression(args)


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

    __slots__ = ('_set', '_value', '_index', '_id', '_lock')

    def __init__(self, _set, index=0, _id=None):
        self._set = _set
        self._value = _NotSpecified
        self._index = index
        self._id = _id
        self._lock = None

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
                    self, "Evaluating uninitialized IndexTemplate (%s)"
                    % (self,))
            return None
        else:
            return self._value

    def _resolve_template(self, args):
        assert not args
        return self()

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

    def set_value(self, values=_NotSpecified, lock=None):
        # It might be nice to check if the value is valid for the base
        # set, but things are tricky when the base set is not dimention
        # 1.  So, for the time being, we will just "trust" the user.
        # After all, the actual Set will raise exceptions if the value
        # is not present.
        if lock is not self._lock:
            raise RuntimeError(
                "The TemplateIndex %s is currently locked by %s and "
                "cannot be set through lock %s" % (self, self._lock, lock))
        if values is _NotSpecified:
            self._value = _NotSpecified
            return
        if type(values) is not tuple:
            values = (values,)
        if self._index is not None:
            if len(values) == 1:
                self._value = values[0]
            else:
                raise ValueError("Passed multiple values %s to a scalar "
                                 "IndexTemplate %s" % (values, self))
        else:
            self._value = values

    def lock(self, lock):
        assert self._lock is None
        self._lock = lock
        return self._value

    def unlock(self, lock):
        assert self._lock is lock
        self._lock = None


def resolve_template(expr):
    """Resolve a template into a concrete expression

    This takes a template expression and returns the concrete equivalent
    by substituting the current values of all IndexTemplate objects and
    resolving (evaluating and removing) all GetItemExpression,
    GetAttrExpression, and TemplateSumExpression expression nodes.

    """
    def beforeChild(node, child, child_idx):
        # Efficiency: do not decend into leaf nodes.
        if type(child) in native_types or not child.is_expression_type():
            if hasattr(child, '_resolve_template'):
                return False, child._resolve_template(())
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
        initializeWalker=lambda x: beforeChild(None, x, None),
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
        self._base = expr.arg(0)
        self._args = []
        _hash = [ id(self._base) ]
        for x in expr.args[1:]:
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

    @property
    def base(self):
        return self._base

    @property
    def args(self):
        return self._args

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
        _map[_id]._name = "%s[%s]" % (
            _id.base.name, ','.join(str(x) for x in _id.args) )
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
        if d is None or type(d) is not int:
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
    internal_error = None
    _old_iters = (
            pyomo.core.base.set._FiniteSetMixin.__iter__,
            GetItemExpression.__iter__,
            GetAttrExpression.__iter__,
        )
    _old_sum = builtins.sum
    try:
        # Override Set iteration to return IndexTemplates
        pyomo.core.base.set._FiniteSetMixin.__iter__ \
            = GetItemExpression.__iter__ \
            = GetAttrExpression.__iter__ \
            = lambda x: context.get_iter(x).__iter__()
        # Override sum with our sum
        builtins.sum = context.sum_template
        # Get the index templates needed for calling the rule
        if index_set is not None:
            # Note, do not rely on the __iter__ overload, as non-finite
            # Sets don't have an __iter__.
            indices = next(iter(context.get_iter(index_set)))
            try:
                context.cache.pop()
            except IndexError:
                assert indices is None
                indices = ()
        else:
            indices = ()
        if type(indices) is not tuple:
            indices = (indices,)
        # Call the rule, returning the template expression and the
        # top-level IndexTemplate(s) generated when calling the rule.
        #
        # TBD: Should this just return a "FORALL()" expression node that
        # behaves similarly to the GetItemExpression node?
        return rule(block, indices), indices
    except:
        internal_error = sys.exc_info()
        raise
    finally:
        pyomo.core.base.set._FiniteSetMixin.__iter__, \
            GetItemExpression.__iter__, \
            GetAttrExpression.__iter__ = _old_iters
        builtins.sum = _old_sum
        if len(context.cache):
            if internal_error is not None:
                logger.error("The following exception was raised when "
                             "templatizing the rule '%s':\n\t%s"
                             % (rule.__name__, internal_error[1]))
            raise TemplateExpressionError(
                None,
                "Explicit iteration (for loops) over Sets is not supported "
                "by template expressions.  Encountered loop over %s"
                % (context.cache[-1][0]._set,))
    return None, indices


def templatize_constraint(con):
    return templatize_rule(con.parent_block(), con.rule, con.index_set())
