#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
import logging
import sys
import builtins
from contextlib import nullcontext

from pyomo.common.collections import MutableMapping
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.gc_manager import PauseGC
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
    ARG_TYPE,
    NumericExpression,
    Numeric_NPV_Mixin,
    SumExpression,
    mutable_expression,
    register_arg_type,
    _balanced_parens,
)
from pyomo.core.expr.numvalue import (
    NumericValue,
    native_types,
    nonpyomo_leaf_types,
    as_numeric,
    value,
    is_constant,
)
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
    ExpressionReplacementVisitor,
    StreamBasedExpressionVisitor,
    expression_to_string,
    _ToStringVisitor,
)

logger = logging.getLogger(__name__)


class _NotSpecified(object):
    pass


class GetItemExpression(ExpressionBase):
    """
    Expression to call :func:`__getitem__` on the base object.
    """

    __slots__ = ()
    PRECEDENCE = 1

    def __new__(cls, args=()):
        if cls is not GetItemExpression:
            return super().__new__(cls)
        npv_args = not any(
            hasattr(arg, 'is_potentially_variable') and arg.is_potentially_variable()
            for arg in args
        )
        try:
            component = _reduce_template_to_component(args[0])
            cdata = component._ComponentDataClass(component)
            if cdata.is_numeric_type():
                if npv_args and not cdata.is_potentially_variable():
                    return super().__new__(NPV_Numeric_GetItemExpression)
                else:
                    return super().__new__(Numeric_GetItemExpression)
            if cdata.is_logical_type():
                if npv_args and not cdata.is_potentially_variable():
                    return super().__new__(NPV_Boolean_GetItemExpression)
                else:
                    return super().__new__(Boolean_GetItemExpression)
        except (AttributeError, TypeError):
            # TypeError: error reducing to a component (usually due to
            #     unbounded domain on a Var used in a GetItemExpression)
            # AttributeError: resolved component did not support the
            #     PyomoObject API
            pass
        if npv_args:
            return super().__new__(NPV_Structural_GetItemExpression)
        else:
            return super().__new__(Structural_GetItemExpression)

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

    def nargs(self):
        return len(self._args_)

    def _is_fixed(self, values):
        if not all(values[1:]):
            return False
        _true = lambda: True
        return all(getattr(x, 'is_fixed', _true)() for x in values[0].values())

    def _to_string(self, values, verbose, smap):
        values = tuple(_[1:-1] if _[0] == '(' and _[-1] == ')' else _ for _ in values)
        if verbose:
            return "getitem(%s, %s)" % (values[0], ', '.join(values[1:]))
        return "%s[%s]" % (values[0], ','.join(values[1:]))

    def _resolve_template(self, args):
        return args[0].__getitem__(args[1:])

    def _apply_operation(self, result):
        return result[0].__getitem__(result[1:])


class Numeric_GetItemExpression(GetItemExpression, NumericExpression):
    __slots__ = ()

    def nargs(self):
        return len(self._args_)

    def _compute_polynomial_degree(self, result):
        if any(x != 0 for x in result[1:]):
            return None
        ans = 0
        for x in result[0].values():
            if x.__class__ in nonpyomo_leaf_types or not hasattr(
                x, 'polynomial_degree'
            ):
                continue
            tmp = x.polynomial_degree()
            if tmp is None:
                return None
            elif tmp > ans:
                ans = tmp
        return ans


class NPV_Numeric_GetItemExpression(Numeric_NPV_Mixin, Numeric_GetItemExpression):
    __slots__ = ()


class Boolean_GetItemExpression(GetItemExpression, BooleanExpression):
    __slots__ = ()


class NPV_Boolean_GetItemExpression(NPV_Mixin, Boolean_GetItemExpression):
    __slots__ = ()


class Structural_GetItemExpression(ExpressionArgs_Mixin, GetItemExpression):
    __slots__ = ()


class NPV_Structural_GetItemExpression(NPV_Mixin, Structural_GetItemExpression):
    __slots__ = ()


class GetAttrExpression(ExpressionBase):
    """
    Expression to call :func:`__getattr__` on the base object.
    """

    __slots__ = ()
    PRECEDENCE = 1

    def __new__(cls, args=()):
        if cls is not GetAttrExpression:
            return super().__new__(cls)
        # Ironically, we need to actually create this object in order to
        # determine what the class for this object should be.
        if args[0].is_potentially_variable():
            self = Structural_GetAttrExpression(args)
        else:
            self = NPV_Structural_GetAttrExpression(args)
        try:
            attr = _reduce_template_to_component(self)
            if attr.is_numeric_type():
                if attr.is_potentially_variable() or self.is_potentially_variable():
                    return super().__new__(Numeric_GetAttrExpression)
                else:
                    return super().__new__(NPV_Numeric_GetAttrExpression)
            elif attr.is_logical_type():
                if attr.is_potentially_variable() or self.is_potentially_variable():
                    return super().__new__(Boolean_GetAttrExpression)
                else:
                    return super().__new__(NPV_Boolean_GetAttrExpression)
        except (AttributeError, TypeError):
            # TypeError: error reducing to a component (usually due to
            #     unbounded domain on a Var used in a GetItemExpression)
            # AttributeError: resolved component did not support the
            #     PyomoObject API
            pass
        return self

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

    def __call__(self, *args, **kwargs):
        """
        Return the value of this object.
        """
        # Backwards compatibility with __call__(exception):
        #
        # TODO: deprecate (then remove) evaluating expressions by
        # "calling" them.
        #
        # [ESJ 3/25/25]: Note that since this always calls the ExpressionBase
        # implementation of __call__ if 'exception' is specified, we need not
        # check the type of the exception arg here--it will get checked in the
        # base class.
        try:
            if not args:
                if not kwargs:
                    return super().__call__()
                elif len(kwargs) == 1 and 'exception' in kwargs:
                    return super().__call__(**kwargs)
            elif (
                not kwargs and len(args) == 1 and (args[0] is True or args[0] is False)
            ):
                return super().__call__(*args)
        except TemplateExpressionError:
            pass
        # Note: the only time we will implicitly create a CallExpression
        # node is directly after a GetAttrExpression: that is, someone
        # got the attribute (method) and is now calling it.
        # Implementing the auto-generation of CallExpression in other
        # contexts is likely to be confounded with evaluating expressions.
        return CallExpression((self,) + args, kwargs)

    def getname(self, *args, **kwds):
        return 'getattr'

    def nargs(self):
        return 2

    def _apply_operation(self, result):
        obj, attr = result
        return getattr(obj, attr)

    def _to_string(self, values, verbose, smap):
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
        return getattr(*args)


class Numeric_GetAttrExpression(GetAttrExpression, NumericExpression):
    __slots__ = ()

    def _compute_polynomial_degree(self, result):
        if result[1] != 0:
            return None
        return result[0]


class NPV_Numeric_GetAttrExpression(Numeric_NPV_Mixin, Numeric_GetAttrExpression):
    __slots__ = ()


class Boolean_GetAttrExpression(GetAttrExpression, BooleanExpression):
    __slots__ = ()


class NPV_Boolean_GetAttrExpression(NPV_Mixin, Boolean_GetAttrExpression):
    __slots__ = ()


class Structural_GetAttrExpression(ExpressionArgs_Mixin, GetAttrExpression):
    __slots__ = ()


class NPV_Structural_GetAttrExpression(NPV_Mixin, Structural_GetAttrExpression):
    __slots__ = ()


class CallExpression(NumericExpression):
    """
    Expression to call :func:`__call__` on the base object.
    """

    __slots__ = ('_kwds',)
    PRECEDENCE = None

    def __init__(self, args, kwargs):
        self._args_ = tuple(args) + tuple(kwargs.values())
        self._kwds = tuple(kwargs.keys())

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
        return 'call'

    def _compute_polynomial_degree(self, result):
        return None

    def _apply_operation(self, result):
        na = len(self._args_) - len(self._kwds)
        return result[0](*result[1:na], **dict(zip(self._kwds, result[na:])))

    def _to_string(self, values, verbose, smap):
        na = len(self._args_) - len(self._kwds)
        args = ', '.join(values[1:na])
        if self._kwds:
            if na > 1:
                args += ', '
            args += ', '.join(
                f'{key}={val}' for key, val in zip(self._kwds, values[na:])
            )
        if verbose:
            return f"call({values[0]}, {args})"
        return f"{values[0]}({args})"

    def _resolve_template(self, args):
        return self._apply_operation(args)


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
            tuple(it.lock(self._lock) for it in iterGroup)
            for iterGroup in self._tse._iters
        )

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


class TemplateSumExpression(NumericExpression):
    """
    Expression to represent an unexpanded sum over one or more sets.
    """

    __slots__ = ('_iters', '_local_args_')
    PRECEDENCE = 1

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

    def template_args(self):
        ans = list(self._local_args_)
        for itergroup in self._iters:
            ans.append(itergroup[0]._set)
        return tuple(ans)

    def template_iters(self):
        return self._iters

    def create_node_with_local_data(self, args):
        return self.__class__(args, self._iters)

    def getname(self, *args, **kwds):
        return "SUM"

    def is_potentially_variable(self):
        if any(
            arg.is_potentially_variable()
            for arg in self._local_args_
            if arg.__class__ not in nonpyomo_leaf_types
        ):
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

    def to_string(self, verbose=None, smap=None):
        ans = ''
        assert len(self._local_args_) == 1
        val = expression_to_string(self._local_args_[0], verbose=verbose, smap=smap)
        if val[0] == '(' and val[-1] == ')' and _balanced_parens(val[1:-1]):
            val = val[1:-1]
        iterStrGenerator = (
            (
                ', '.join(
                    (smap.getSymbol(i) if smap is not None else str(i))
                    for i in iterGroup
                ),
                (
                    iterGroup[0]._set.to_string(verbose=verbose, smap=smap)
                    if hasattr(iterGroup[0]._set, 'to_string')
                    else (
                        smap.getSymbol(iterGroup[0]._set)
                        if smap is not None
                        else str(iterGroup[0]._set)
                    )
                ),
            )
            for iterGroup in self._iters
        )
        if verbose:
            iterStr = ', '.join('iter(%s, %s)' % x for x in iterStrGenerator)
            return 'templatesum(%s, %s)' % (val, iterStr)
        else:
            iterStr = ' '.join('for %s in %s' % x for x in iterStrGenerator)
            return 'SUM(%s %s)' % (val, iterStr)

    def _resolve_template(self, args):
        with mutable_expression() as e:
            for arg in args:
                e += arg
        if e.nargs() > 1:
            return e
        elif not e.nargs():
            return 0
        else:
            return e.arg(0)


# FIXME: This is a hack to get certain complex cases to print without error
_ToStringVisitor._leaf_node_types.add(TemplateSumExpression)


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

    __slots__ = ('_set', '_value', '_index', '_id', '_group', '_lock')

    def __init__(self, _set, index=0, _id=None, _group=None):
        self._set = _set
        self._value = _NotSpecified
        self._index = index
        self._id = _id
        self._group = _group
        self._lock = None

    def __deepcopy__(self, memo):
        # Because we leverage deepcopy for expression/component cloning,
        # we need to see if this is a Component.clone() operation and
        # *not* copy the template.
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
        return super().__deepcopy__(memo)

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is _NotSpecified:
            if exception:
                raise TemplateExpressionError(
                    self, "Evaluating uninitialized IndexTemplate (%s)" % (self,)
                )
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
        return "{" + _set_name + "}"

    def set_value(self, values=_NotSpecified, lock=None):
        # It might be nice to check if the value is valid for the base
        # set, but things are tricky when the base set is not dimension
        # 1.  So, for the time being, we will just "trust" the user.
        # After all, the actual Set will raise exceptions if the value
        # is not present.
        if lock is not self._lock:
            raise RuntimeError(
                "The IndexTemplate %s is currently locked by %s and "
                "cannot be set through lock %s" % (self, self._lock, lock)
            )
        if values is _NotSpecified:
            self._value = _NotSpecified
            return
        if type(values) is not tuple:
            values = (values,)
        if self._index is not None:
            if len(values) == 1:
                self._value = values[0]
            else:
                self._value = values[self._index]
        else:
            self._value = values

    def lock(self, lock):
        assert self._lock is None
        self._lock = lock
        return self._value

    def unlock(self, lock):
        assert self._lock is lock
        self._lock = None


# Instead of special-casing _categorize_arg_type for this class, we
# will directly register that it should be treated as an NPV arg
register_arg_type(IndexTemplate, ARG_TYPE.NPV)


class _TemplateResolver(StreamBasedExpressionVisitor):
    def beforeChild(self, node, child, child_idx):
        # Efficiency: do not descend into leaf nodes.
        if type(child) in native_types:
            return False, child
        elif not child.is_expression_type():
            if hasattr(child, '_resolve_template'):
                return False, child._resolve_template(())
            return False, child
        else:
            return True, None

    def exitNode(self, node, args):
        if hasattr(node, '_resolve_template'):
            return node._resolve_template(args)
        if len(args) == node.nargs() and all(a is b for a, b in zip(node.args, args)):
            return node
        if all(map(is_constant, args)):
            return node._apply_operation(args)
        else:
            return node.create_node_with_local_data(args)

    def initializeWalker(self, expr):
        return self.beforeChild(None, expr, None)


def resolve_template(expr):
    """Resolve a template into a concrete expression

    This takes a template expression and returns the concrete equivalent
    by substituting the current values of all IndexTemplate objects and
    resolving (evaluating and removing) all GetItemExpression,
    GetAttrExpression, and TemplateSumExpression expression nodes.

    """
    if resolve_template.visitor is None:
        resolve_template.visitor = _TemplateResolver()
    return resolve_template.visitor.walk_expression(expr)


resolve_template.visitor = None


class _wildcard_info(object):
    __slots__ = ('iter', 'source', 'value', 'original_value', 'objects')

    def __init__(self, src, obj):
        self.source = src
        self.original_value = obj._value
        self.objects = [obj]
        self.reset()
        if self.original_value in (None, _NotSpecified):
            self.advance()

    def advance(self):
        with _TemplateIterManager.pause():
            self.value = next(self.iter)
        for obj in self.objects:
            obj.set_value(self.value)

    def reset(self):
        # Because we want to actually iterate over the underlying
        # template expression, we will temporarily pause our overrides
        # of sum() and the set iters
        with _TemplateIterManager.pause():
            self.iter = iter(self.source)

    def restore(self):
        for obj in self.objects:
            obj.set_value(self.original_value)


def _reduce_template_to_component(expr):
    """Resolve a template into a concrete component

    This takes a template expression and returns the concrete equivalent
    by substituting the current values of all IndexTemplate objects and
    resolving (evaluating and removing) all GetItemExpression,
    GetAttrExpression, and TemplateSumExpression expression nodes.

    """
    import pyomo.core.base.set

    # wildcards holds lists of
    #   [iterator, source, value, orig_value, object0, ...]
    # 'iterator' iterates over 'source' to provide 'value's for each of
    # the 1 or more 'objects'.  Objects can be IndexTemplate objects or
    # (discrete) Variables
    wildcards = []
    wildcard_groups = {}
    level = -1

    def beforeChild(node, child, child_idx):
        # Efficiency: do not descend into leaf nodes.
        if type(child) in native_types:
            return False, child
        elif not child.is_expression_type():
            if hasattr(child, '_resolve_template'):
                try:
                    ans = child._resolve_template(())
                except TemplateExpressionError:
                    # We are attempting "loose" template resolution: for
                    # every unset IndexTemplate, search the underlying
                    # set to find *any* valid match.
                    if child._group not in wildcard_groups:
                        wildcard_groups[child._group] = len(wildcards)
                        info = _wildcard_info(child._set, child)
                        wildcards.append(info)
                    else:
                        info = wildcards[wildcard_groups[child._group]]
                        info.objects.append(child)
                        child.set_value(info.value)
                    ans = child._resolve_template(())
                return False, ans
            if child.is_variable_type():
                from pyomo.core.base.set import RangeSet

                if child.domain.isdiscrete():
                    domain = child.domain
                    bounds = child.bounds
                    if bounds != (None, None):
                        try:
                            bounds = pyomo.core.base.set.RangeSet(*bounds, 0)
                            domain = domain & bounds
                        except:
                            pass
                    info = _wildcard_info(domain, child)
                    wildcards.append(info)
                return False, value(child)
            return False, child
        else:
            return True, None

    def exitNode(node, args):
        if hasattr(node, '_resolve_template'):
            return node._resolve_template(args)
        if len(args) == node.nargs() and all(a is b for a, b in zip(node.args, args)):
            return node
        if all(map(is_constant, args)):
            return node._apply_operation(args)
        else:
            return node.create_node_with_local_data(args)

    walker = StreamBasedExpressionVisitor(
        initializeWalker=lambda x: beforeChild(None, x, None),
        beforeChild=beforeChild,
        exitNode=exitNode,
    )
    while 1:
        try:
            with _TemplateIterManager.pause():
                ans = walker.walk_expression(expr)
            break
        except (KeyError, AttributeError):
            # We are attempting "loose" template resolution: for every
            # unset IndexTemplate, search the underlying set to find
            # *any* valid match.
            level = len(wildcards) - 1
            while level >= 0:
                info = wildcards[level]
                try:
                    info.advance()
                    break
                except StopIteration:
                    # Because we want to actually iterate over the
                    # underlying template expression, we will
                    # temporarily pause our overrides of sum() and the
                    # set iters
                    info.reset()
                    info.advance()
                    level -= 1
            if level < 0:
                for info in wildcards:
                    info.restore()
                raise
    for info in wildcards:
        info.restore()
    return ans


class ReplaceTemplateExpression(ExpressionReplacementVisitor):
    template_types = {
        IndexTemplate,
        GetItemExpression,
        Numeric_GetItemExpression,
        NPV_Numeric_GetItemExpression,
        Boolean_GetItemExpression,
        NPV_Boolean_GetItemExpression,
    }

    def __init__(self, substituter, *args, **kwargs):
        kwargs.setdefault('remove_named_expressions', True)
        super().__init__(**kwargs)
        self.substituter = substituter
        self.substituter_args = args

    def beforeChild(self, node, child, child_idx):
        if type(child) in ReplaceTemplateExpression.template_types:
            return False, self.substituter(child, *self.substituter_args)
        return super().beforeChild(node, child, child_idx)


def substitute_template_expression(expr, substituter, *args, **kwargs):
    r"""Substitute IndexTemplates in an expression tree.

    This is a general utility function for walking the expression tree
    and substituting all occurrences of IndexTemplate and
    GetItemExpression nodes.

    Parameters
    ----------
    expr : NumericExpression
        the source template expression

    substituter: Callable
        method taking ``(expression, *args)`` and returning the new object

    \*args:
        positional arguments passed directly to the substituter

    Returns
    -------
    NumericExpression :
        a new expression tree with all substitutions done

    """
    visitor = ReplaceTemplateExpression(substituter, *args, **kwargs)
    return visitor.walk_expression(expr)


class _GetItemIndexer(object):
    # Note that this class makes the assumption that only one template
    # ever appears in an expression for a single index

    def __init__(self, expr):
        self._base = expr.arg(0)
        self._args = []
        _hash = [id(self._base)]
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
                        "IndexTemplate in an expression.\n\tFound in %s" % (expr,)
                    )
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
        return "%s[%s]" % (self._base.name, ','.join(str(x) for x in self._args))


def substitute_getitem_with_param(expr, _map):
    """A simple substituter to replace _GetItem nodes with mutable Params.

    This substituter will replace all GetItemExpression nodes with a
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
        _map[_id]._name = "%s[%s]" % (_id.base.name, ','.join(str(x) for x in _id.args))
    return _map[_id]


def substitute_template_with_value(expr):
    """A simple substituter to expand expression for current template

    This substituter will replace all GetItemExpression / IndexTemplate
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
        if _set.is_expression_type():
            d = _reduce_template_to_component(_set).dimen
        else:
            d = _set.dimen
        grp = context.next_group()
        if d is None or type(d) is not int:
            idx = (IndexTemplate(_set, None, context.next_id(), grp),)
        else:
            idx = tuple(
                IndexTemplate(_set, i, context.next_id(), grp) for i in range(d)
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
        self._group = 0

    def get_iter(self, _set):
        return _set_iterator_template_generator(_set, self)

    def npop_cache(self, n):
        result = self.cache[-n:]
        self.cache[-n:] = []
        return result

    def next_id(self):
        self._id += 1
        return self._id

    def next_group(self):
        self._group += 1
        return self._group

    def sum_template(self, generator):
        init_cache = len(self.cache)
        expr = next(generator)
        final_cache = len(self.cache)
        return TemplateSumExpression((expr,), self.npop_cache(final_cache - init_cache))


class _template_iter_manager(object):
    class _iter_wrapper(object):
        __slots__ = ('_class', '_iter', '_old_iter')

        def __init__(self, cls, context):
            def _iter_fcn(obj):
                return context.get_iter(obj)

            self._class = cls
            self._old_iter = cls.__iter__
            self._iter = _iter_fcn

        def acquire(self):
            self._class.__iter__ = self._iter

        def release(self):
            self._class.__iter__ = self._old_iter

    class _pause_template_iter_manager(object):
        __slots__ = ('iter_manager',)

        def __init__(self, iter_manager):
            self.iter_manager = iter_manager

        def __enter__(self):
            self.iter_manager.release()
            return self

        def __exit__(self, et, ev, tb):
            self.iter_manager.acquire()

    def __init__(self):
        self.paused = True
        self.context = None
        self.iters = None
        self.builtin_sum = builtins.sum

    def init(self, context, *iter_fcns):
        assert self.context is None
        self.context = context
        self.iters = [self._iter_wrapper(it, context) for it in iter_fcns]
        return self

    def acquire(self):
        assert self.paused
        self.paused = False
        builtins.sum = self.context.sum_template
        for it in self.iters:
            it.acquire()

    def release(self):
        assert not self.paused
        self.paused = True
        builtins.sum = self.builtin_sum
        for it in self.iters:
            it.release()

    def __enter__(self):
        assert self.context
        self.acquire()
        return self

    def __exit__(self, et, ev, tb):
        self.release()
        self.context = None
        self.iters = None

    def pause(self):
        if self.paused:
            return nullcontext()
        else:
            return self._pause_template_iter_manager(self)


# Global manager for coordinating overriding set iteration
_TemplateIterManager = _template_iter_manager()


def templatize_rule(block, rule, index_set):
    import pyomo.core.base.set

    context = _template_iter_context()
    internal_error = None
    try:
        # Override Set iteration to return IndexTemplates
        with _TemplateIterManager.init(
            context,
            pyomo.core.base.set._FiniteSetMixin,
            GetItemExpression,
            GetAttrExpression,
        ):
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
        if len(context.cache):
            if internal_error is not None:
                logger.error(
                    "The following exception was raised when "
                    "templatizing the rule '%s':\n\t%s" % (rule.name, internal_error[1])
                )
            raise TemplateExpressionError(
                None,
                "Explicit iteration (for loops) over Sets is not supported "
                "by template expressions.  Encountered loop over %s"
                % (context.cache[-1][0]._set,),
            )
    return None, indices


def templatize_constraint(con):
    expr, indices = templatize_rule(con.parent_block(), con.rule, con.index_set())
    if expr.__class__ is tuple:
        expr = tuple_to_relational_expr(expr)
    return expr, indices
