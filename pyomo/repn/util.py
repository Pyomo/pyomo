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

import collections
import functools
import itertools
import logging
import operator
import sys

from pyomo.common import enums
from pyomo.common.collections import Sequence, ComponentMap, ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.numeric_types import (
    check_if_numeric_type,
    native_types,
    native_numeric_types,
    native_complex_types,
    native_logical_types,
)
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base import (
    Var,
    Param,
    Expression,
    Objective,
    Block,
    Constraint,
    Expression,
    NumericLabeler,
    Suffix,
    SortComponents,
)
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import NamedExpressionData
from pyomo.core.expr.numvalue import is_fixed, value
import pyomo.core.expr as EXPR
import pyomo.core.kernel as kernel

logger = logging.getLogger(__name__)

valid_expr_ctypes_minlp = {Var, Param, Expression, Objective}
valid_active_ctypes_minlp = {Block, Constraint, Objective, Suffix}
sum_like_expression_types = {
    EXPR.SumExpression,
    EXPR.LinearExpression,
    EXPR.NPV_SumExpression,
}
_named_subexpression_types = (
    NamedExpressionData,
    kernel.expression.expression,
    kernel.objective.objective,
)

HALT_ON_EVALUATION_ERROR = False
nan = float('nan')
int_float = {int, float}


class ExprType(enums.IntEnum):
    # Note that the ordering is meaningful, and we will compare
    # instances using relational operators.  In particular, we assume
    # that the Enum values increase in polynomial degree, starting at
    # CONSTANT and endine at GENERAL (nonlinear).
    CONSTANT = 0
    FIXED = 3
    VARIABLE = 5
    MONOMIAL = 10
    LINEAR = 20
    QUADRATIC = 30
    GENERAL = 40


_FileDeterminism_deprecation = {
    1: 20,
    2: 30,
    'DEPRECATED_KEYS': 20,
    'DEPRECATED_KEYS_AND_NAMES': 30,
}


class FileDeterminism(enums.IntEnum):
    NONE = 0
    # DEPRECATED_KEYS = 1
    # DEPRECATED_KEYS_AND_NAMES = 2
    ORDERED = 10
    SORT_INDICES = 20
    SORT_SYMBOLS = 30

    # We will define __str__ and __format__ so that behavior in python
    # 3.11 is consistent with 3.7 - 3.10.

    def __str__(self):
        return enums.Enum.__str__(self)

    def __format__(self, spec):
        # Removal of Python 3.7 support allows us to use Enum.__format__
        return enums.Enum.__format__(self, spec)

    @classmethod
    def _missing_(cls, value):
        # This is not a perfect deprecation path, as the old attributes
        # are no longer valid.  However, as the previous implementation
        # was a pure int and not an Enum, this is sufficient for our
        # needs.
        if value in _FileDeterminism_deprecation:
            new = FileDeterminism(_FileDeterminism_deprecation[value])
            deprecation_warning(
                f'FileDeterminism({value}) is deprecated.  '
                f'Please use {str(new)} ({int(new)})',
                version='6.5.0',
            )
            return new
        return super()._missing_(value)


def val2str(val):
    """Converts an object to str with special handling for InvalidNumber

    This will convert the ``val`` to a string.  If ``val`` is an
    InvalidNumber, the conversion to string will bypass the exception
    raised by :py:meth:`InvalidNumber.__str__`.

    """
    if hasattr(val, '_str'):
        return val._str()
    if hasattr(val, 'to_string'):
        return val.to_string()
    return repr(val)


class InvalidNumber(PyomoObject):
    def __init__(self, value, cause=""):
        self.value = value
        if cause.__class__ is list:
            self.causes = list(cause)
        else:
            self.causes = [cause]

    @staticmethod
    def parse_args(*args):
        causes = []
        real_args = []
        for arg in args:
            if arg.__class__ is InvalidNumber:
                causes.extend(arg.causes)
                real_args.append(arg.value)
            else:
                real_args.append(arg)
        return real_args, causes

    def _cmp(self, op, other):
        args, causes = InvalidNumber.parse_args(self, other)
        try:
            return op(*args)
        except TypeError:
            # TypeError will be raised when comparing incompatible types
            # (e.g., int <= None)
            return False

    def _op(self, op, *args):
        args, causes = InvalidNumber.parse_args(*args)
        try:
            return InvalidNumber(op(*args), causes)
        except TypeError:
            # TypeError will be raised when operating on incompatible
            # types (e.g., int + None);
            return InvalidNumber(self.value, causes)
        except (ArithmeticError, ValueError):
            # ArithmeticError and ValueError can be raised by invalid
            # operations (like divide by zero or log of a negative number)
            return InvalidNumber(float('nan'), causes)

    def __eq__(self, other):
        ans = self._cmp(operator.eq, other)
        try:
            return bool(ans)
        except ValueError:
            # ValueError can be raised by numpy.ndarray when two arrays
            # are returned.  In that case, ndarray returns a new ndarray
            # of bool values.  We will fall back on using `all` to
            # reduce it to a single bool.
            try:
                return all(ans)
            except:
                pass
            raise

    def __lt__(self, other):
        # Note that as < is ambiguous for arrays, we will attempt to
        # cast the result to bool and if it was an array, allow the
        # exception to propagate
        return bool(self._cmp(operator.lt, other))

    def __gt__(self, other):
        # See the comment in __lt__() on the use of bool()
        return bool(self._cmp(operator.gt, other))

    def __le__(self, other):
        # See the comment in __lt__() on the use of bool()
        return bool(self._cmp(operator.le, other))

    def __ge__(self, other):
        # See the comment in __lt__() on the use of bool()
        return bool(self._cmp(operator.ge, other))

    def _error(self, msg):
        causes = list(filter(None, self.causes))
        if causes:
            msg += "\nThe InvalidNumber was generated by:\n\t"
            msg += "\n\t".join(causes)
        raise InvalidValueError(msg)

    def __str__(self):
        # We want attempts to convert InvalidNumber to a string
        # representation to raise a InvalidValueError, unless we are in
        # the middle of processing an exception.  In that case, it is
        # very likely that an exception handler is generating an error
        # message.  We will play nice and return a reasonable string.
        if sys.exc_info()[1] is None:
            self._error(f'Cannot emit {self._str()} in compiled representation')
        else:
            return self._str()

    def _str(self):
        return f'InvalidNumber({self.value!r})'

    def __repr__(self):
        return str(self)

    def __format__(self, format_spec):
        # FIXME: We want to move to where converting InvalidNumber to
        # string (with either repr() or f"") should raise a
        # InvalidValueError.  However, at the moment, this breaks some
        # tests in PyROS.
        # return self.value.__format__(format_spec)
        return self._error(f'Cannot emit {str(self)} in compiled representation')

    def __float__(self):
        # We want attempts to convert InvalidNumber to a float
        # representation to raise a InvalidValueError.
        return self._error(f'Cannot convert {str(self)} to float')

    def __neg__(self):
        return self._op(operator.neg, self)

    def __abs__(self):
        return self._op(operator.abs, self)

    def __add__(self, other):
        return self._op(operator.add, self, other)

    def __sub__(self, other):
        return self._op(operator.sub, self, other)

    def __mul__(self, other):
        return self._op(operator.mul, self, other)

    def __truediv__(self, other):
        return self._op(operator.truediv, self, other)

    def __pow__(self, other):
        return self._op(operator.pow, self, other)

    def __radd__(self, other):
        return self._op(operator.add, other, self)

    def __rsub__(self, other):
        return self._op(operator.sub, other, self)

    def __rmul__(self, other):
        return self._op(operator.mul, other, self)

    def __rtruediv__(self, other):
        return self._op(operator.truediv, other, self)

    def __rpow__(self, other):
        return self._op(operator.pow, other, self)


_CONSTANT = ExprType.CONSTANT


class BeforeChildDispatcher(collections.defaultdict):
    """Dispatcher for handling the :class:`StreamBasedExpressionVisitor`
    `beforeChild` callback

    This dispatcher implements a specialization of :class:`defaultdict`
    that supports automatic type registration.  Any missing types will
    return the :meth:`register_dispatcher` method, which (when called
    as a callback) will interrogate the type, identify the appropriate
    callback, add the callback to the dict, and return the result of
    calling the callback.  As the callback is added to the dict, no type
    will incur the overhead of `register_dispatcher` more than once.

    Note that all dispatchers are implemented as `staticmethod`
    functions to avoid the (unnecessary) overhead of binding to the
    dispatcher object.

    """

    __slots__ = ()

    def __missing__(self, key):
        return self.register_dispatcher

    def register_dispatcher(self, visitor, child):
        child_type = type(child)
        if child_type in native_numeric_types:
            self[child_type] = self._before_native_numeric
        elif child_type in native_logical_types:
            self[child_type] = self._before_native_logical
        elif issubclass(child_type, str):
            self[child_type] = self._before_string
        elif child_type in native_types:
            if issubclass(child_type, tuple(native_complex_types)):
                self[child_type] = self._before_complex
            else:
                self[child_type] = self._before_invalid
        elif not hasattr(child, 'is_expression_type'):
            if check_if_numeric_type(child):
                self[child_type] = self._before_native_numeric
            else:
                self[child_type] = self._before_invalid
        elif not child.is_expression_type():
            if child.is_indexed():
                cdata = child._ComponentDataClass(child)
                if cdata.is_expression_type():
                    self[child_type] = self._before_indexed_expr
                elif cdata.is_numeric_type() or child.is_logical_type():
                    if cdata.is_potentially_variable():
                        self[child_type] = self._before_indexed_var
                    else:
                        self[child_type] = self._before_indexed_param
                elif child.is_component_type():
                    self[child_type] = self._before_indexed_component
            elif child.is_numeric_type() or child.is_logical_type():
                if child.is_potentially_variable():
                    self[child_type] = self._before_var
                elif isinstance(child, EXPR.IndexTemplate):
                    self[child_type] = self._before_index_template
                else:
                    self[child_type] = self._before_param
            elif child.is_component_type():
                self[child_type] = self._before_component
        elif not child.is_potentially_variable():
            self[child_type] = self._before_npv
            pv_base_type = child.potentially_variable_base_class()
            if pv_base_type not in self:
                try:
                    child.__class__ = pv_base_type
                    self.register_dispatcher(visitor, child)
                finally:
                    child.__class__ = child_type
        elif (
            issubclass(child_type, _named_subexpression_types)
            or child_type is kernel.expression.noclone
        ):
            self[child_type] = self._before_named_expression
        else:
            self[child_type] = self._before_general_expression
        return self[child_type](visitor, child)

    @staticmethod
    def _before_general_expression(visitor, child):
        return True, None

    @staticmethod
    def _before_native_numeric(visitor, child):
        return False, (_CONSTANT, child)

    @staticmethod
    def _before_native_logical(visitor, child):
        return False, (
            _CONSTANT,
            InvalidNumber(
                child, f"{child!r} ({type(child).__name__}) is not a valid numeric type"
            ),
        )

    @staticmethod
    def _before_complex(visitor, child):
        return False, (_CONSTANT, complex_number_error(child, visitor, child))

    @staticmethod
    def _before_invalid(visitor, child):
        return False, (
            _CONSTANT,
            InvalidNumber(
                child, f"{child!r} ({type(child).__name__}) is not a valid numeric type"
            ),
        )

    @staticmethod
    def _before_string(visitor, child):
        return False, (
            _CONSTANT,
            InvalidNumber(
                child, f"{child!r} ({type(child).__name__}) is not a valid numeric type"
            ),
        )

    @staticmethod
    def _before_npv(visitor, child):
        try:
            return False, (
                _CONSTANT,
                visitor.check_constant(visitor.evaluate(child), child),
            )
        except (ValueError, ArithmeticError):
            return True, None

    @staticmethod
    def _before_param(visitor, child):
        return False, (_CONSTANT, visitor.check_constant(child.value, child))

    @staticmethod
    def _before_index_template(visitor, child):
        raise NotImplementedError(
            f"{visitor.__class__.__name__} can not handle expressions "
            f"containing {child.__class__} nodes"
        )

    @staticmethod
    def _before_indexed_expr(visitor, child):
        raise NotImplementedError(
            f"{visitor.__class__.__name__} can not handle expressions "
            f"containing {child.__class__} nodes"
        )

    @staticmethod
    def _before_indexed_param(visitor, child):
        raise NotImplementedError(
            f"{visitor.__class__.__name__} can not handle expressions "
            f"containing {child.__class__} nodes"
        )

    @staticmethod
    def _before_indexed_var(visitor, child):
        raise NotImplementedError(
            f"{visitor.__class__.__name__} can not handle expressions "
            f"containing {child.__class__} nodes"
        )

    @staticmethod
    def _before_component(visitor, child):
        raise NotImplementedError(
            f"{visitor.__class__.__name__} can not handle expressions "
            f"containing {child.__class__} nodes"
        )

    #
    # The following methods must be defined by derivative classes (along
    # with any other special-case handling they want to implement;
    # usually including handling for Monomial, Linear, and
    # ExternalFunction
    #

    # @staticmethod
    # def _before_var(visitor, child):
    #     pass

    # @staticmethod
    # def _before_named_expression(visitor, child):
    #     pass


class ExitNodeDispatcher(collections.defaultdict):
    """Dispatcher for handling the :class:`StreamBasedExpressionVisitor`
    `exitNode` callback

    This dispatcher implements a specialization of :class:`defaultdict`
    that supports automatic type registration.  As the identified
    callback is added to the dict, no type will incur the overhead of
    `register_dispatcher` more than once.

    Note that in this case, the client is expected to register all
    non-NPV expression types.  The auto-registration is designed to only
    handle two cases:
    - Auto-detection of user-defined Named Expression types
    - Automatic mappimg of NPV expressions to their equivalent non-NPV handlers
    - Automatic registration of derived expression types

    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

    def __missing__(self, key):
        if type(key) is tuple:
            # Only lookup/cache argument-specific handlers for unary or
            # binary operators
            if len(key) <= 3:
                node_class = key[0]
                node_args = key[1:]
            else:
                node_class = key = key[0]
                if node_class in self:
                    return self[node_class]
        else:
            node_class = key
        bases = node_class.__mro__
        # Note: if we add an `etype`, then this special-case can be removed
        if (
            issubclass(node_class, _named_subexpression_types)
            or node_class is kernel.expression.noclone
        ):
            bases = [Expression]
        fcn = None
        for base_type in bases:
            if key is not node_class:
                if (base_type,) + node_args in self:
                    fcn = self[(base_type,) + node_args]
                    break
            if base_type in self:
                fcn = self[base_type]
                break
        if fcn is None:
            partial_matches = set(
                k[0] for k in self if type(k) is tuple and issubclass(node_class, k[0])
            )
            for base_type in node_class.__mro__:
                if node_class is not key:
                    key = (base_type,) + node_args
                if base_type in partial_matches:
                    raise DeveloperError(
                        f"Base expression key '{key}' not found when inserting "
                        f"dispatcher for node '{node_class.__name__}' while walking "
                        "expression tree."
                    )
            return self.unexpected_expression_type
        self[key] = fcn
        return fcn

    def unexpected_expression_type(self, visitor, node, *args):
        raise DeveloperError(
            f"Unexpected expression node type '{type(node).__name__}' "
            f"found while walking expression tree in {type(visitor).__name__}."
        )


def initialize_exit_node_dispatcher(exit_handlers):
    exit_dispatcher = {}
    for cls, handlers in exit_handlers.items():
        for args, fcn in handlers.items():
            if args is None:
                exit_dispatcher[cls] = fcn
            else:
                exit_dispatcher[(cls, *args)] = fcn
    return exit_dispatcher


def apply_node_operation(node, args):
    try:
        ans = node._apply_operation(args)
        if ans != ans and ans.__class__ is not InvalidNumber:
            ans = InvalidNumber(ans, "Evaluating '{node}' returned NaN")
        return ans
    except:
        exc_msg = str(sys.exc_info()[1])
        logger.warning(
            "Exception encountered evaluating expression "
            "'%s(%s)'\n\tmessage: %s\n\texpression: %s"
            % (node.name, ", ".join(map(str, args)), exc_msg, node)
        )
        if HALT_ON_EVALUATION_ERROR:
            raise
        return InvalidNumber(nan, exc_msg)


def complex_number_error(value, visitor, expr, node=""):
    msg = f'Pyomo {visitor.__class__.__name__} does not support complex numbers'
    cause = ' '.join(filter(None, ("Complex number returned from expression", node)))
    logger.warning(f"{cause}\n\tmessage: {msg}\n\texpression: {expr}")
    if HALT_ON_EVALUATION_ERROR:
        raise InvalidValueError(
            f'Pyomo {visitor.__class__.__name__} does not support complex numbers'
        )
    return InvalidNumber(value, cause)


def categorize_valid_components(
    model, active=True, sort=None, valid=set(), targets=set()
):
    """Walk model and check for valid component types

    This routine will walk the model and check all component types.
    Components types in the `valid` set are ignored, blocks with
    components in the `targets` set are collected, and all other
    component types are added to a dictionary of `unrecognized`
    components.

    A Component type may not appear in both `valid` and `targets` sets.

    Parameters
    ----------
    model: BlockData
        The model tree to walk

    active: True or None
        If True, only unrecognized active components are returned in the
        `uncategorized` dictionary.  Also, if True, only active Blocks
        are descended into.

    sort: bool or SortComponents
        The sorting flag to pass to the block walkers

    valid: Set[type]
        The set of "valid" component types.  These are ignored by the
        categorizer.

    targets: Set[type]
        The set of component types to "collect".  Blocks with components
        in the `targets` set will be returned in the `component_map`

    Returns
    -------
    component_map: Dict[type, List[BlockData]]
        A dict mapping component type to a list of block data
        objects that contain declared component of that type.

    unrecognized: Dict[type, List[ComponentData]]
        A dict mapping unrecognized component types to a (non-empty)
        list of component data objects found on the model.

    """
    assert active in (True, None)
    # Note: we assume every target component is valid but that we expect
    # there to be far mode valid components than target components.
    # Generate an error if a target is in the valid set (because the
    # valid set will preclude recording the block in the component_map)
    if any(ctype in valid for ctype in targets):
        ctypes = list(filter(valid.__contains__, targets))
        raise DeveloperError(
            f"categorize_valid_components: Cannot have component type {ctypes} in "
            "both the `valid` and `targets` sets"
        )
    unrecognized = {}
    component_map = {k: [] for k in targets}
    for block in model.block_data_objects(active=active, descend_into=True, sort=sort):
        local_ctypes = block.collect_ctypes(active=None, descend_into=False)
        for ctype in local_ctypes:
            if ctype in kernel.base._kernel_ctype_backmap:
                ctype = kernel.base._kernel_ctype_backmap[ctype]
            if ctype in valid:
                continue
            if ctype in targets:
                component_map[ctype].append(block)
                continue
            # TODO: we should rethink the definition of "active" for
            # Components that are not subclasses of ActiveComponent
            if (
                active
                and not issubclass(ctype, ActiveComponent)
                and not issubclass(ctype, kernel.base.ICategorizedObject)
            ):
                continue
            if ctype not in unrecognized:
                unrecognized[ctype] = []
            unrecognized[ctype].extend(
                block.component_data_objects(
                    ctype=ctype,
                    active=active,
                    descend_into=False,
                    sort=SortComponents.unsorted,
                )
            )
    return component_map, {k: v for k, v in unrecognized.items() if v}


def FileDeterminism_to_SortComponents(file_determinism):
    if file_determinism >= FileDeterminism.SORT_SYMBOLS:
        return SortComponents.ALPHABETICAL | SortComponents.SORTED_INDICES
    if file_determinism >= FileDeterminism.SORT_INDICES:
        return SortComponents.SORTED_INDICES
    if file_determinism >= FileDeterminism.ORDERED:
        return SortComponents.ORDERED_INDICES
    return SortComponents.UNSORTED


def initialize_var_map_from_column_order(model, config, var_map):
    column_order = config.column_order
    sorter = FileDeterminism_to_SortComponents(config.file_determinism)

    if column_order is None or column_order.__class__ is bool:
        if not column_order:
            column_order = None
    elif isinstance(column_order, ComponentMap):
        # The column order has historically has supported a ComponentMap of
        # component to position in addition to the simple list of
        # components.  Convert it to the simple list
        column_order = sorted(column_order, key=column_order.__getitem__)

    if column_order == True:
        column_order = model.component_data_objects(Var, descend_into=True, sort=sorter)
    elif config.file_determinism > FileDeterminism.ORDERED:
        # We will pre-gather the variables so that their order
        # matches the file_determinism flag.  This is a little
        # cumbersome, but is implemented this way for consistency
        # with the original NL writer.
        var_objs = model.component_data_objects(Var, descend_into=True, sort=sorter)
        if column_order is None:
            column_order = var_objs
        else:
            column_order = itertools.chain(column_order, var_objs)

    if column_order is not None:
        # Note that Vars that appear twice (e.g., through a
        # Reference) will be sorted with the FIRST occurrence.
        fill_in = ComponentSet()
        for var in column_order:
            if var.is_indexed():
                for _v in var.values(sorter):
                    if not _v.fixed:
                        var_map[id(_v)] = _v
            elif not var.fixed:
                pc = var.parent_component()
                if pc is not var and pc not in fill_in:
                    # For any VarData in an IndexedVar, remember the
                    # IndexedVar so that after all the VarData that the
                    # user has specified in the column ordering have
                    # been processed (and added to the var_map) we can
                    # go back and fill in any missing VarData from that
                    # IndexedVar.  This is needed because later when
                    # walking expressions we assume that any VarData
                    # that is not in the var_map will be the first
                    # VarData from its Var container (indexed or scalar).
                    fill_in.add(pc)
                var_map[id(var)] = var
        # Note that ComponentSet iteration is deterministic, and
        # re-inserting _v into the var_map will not change the ordering
        # for any pre-existing variables
        for pc in fill_in:
            for _v in pc.values(sorter):
                if not _v.fixed:
                    var_map[id(_v)] = _v
    return var_map


def row_order2row_map(config):
    """Convert a row_order (list or ComponentMap) into a dict mapping
    constraint id -> row index. Returns an empty dict if no ordering."""
    row_order = config.row_order
    if row_order is None or isinstance(row_order, bool):
        return {}

    sorter = FileDeterminism_to_SortComponents(config.file_determinism)

    if isinstance(row_order, ComponentMap):
        # Convert ComponentMap to sorted list based on its values
        row_order = sorted(row_order, key=row_order.__getitem__)

    row_map = {}
    for con in row_order:
        if con.is_indexed():
            for c in con.values(sorter):
                row_map[id(c)] = c
        else:
            row_map[id(con)] = con

    # Map the implicit dict ordering to an explicit 0..n ordering
    return {_id: i for i, _id in enumerate(row_map)}


def ordered_active_constraints(model, config):
    sorter = FileDeterminism_to_SortComponents(config.file_determinism)
    constraints = model.component_data_objects(Constraint, active=True, sort=sorter)

    row_map = row_order2row_map(config)
    if not row_map:
        return constraints
    # sorted() is stable (per Python docs), so we can let all
    # unspecified rows have a row number one bigger than the
    # number of rows specified by the user ordering.
    _n = len(row_map)
    _row_getter = row_map.get
    return sorted(constraints, key=lambda x: _row_getter(id(x), _n))


class VarRecorder(object):
    def __init__(self, var_map, sorter):
        self.var_map = var_map
        self.sorter = sorter

    def add(self, var):
        # We always add all indices to the var_map at once so that
        # we can honor deterministic ordering of unordered sets
        # (because the user could have iterated over an unordered
        # set when constructing an expression, thereby altering the
        # order in which we would see the variables)
        vm = self.var_map
        try:
            _iter = var.parent_component().values(self.sorter)
        except AttributeError:
            # Note that this only works for the AML, as kernel does not
            # provide a parent_component()
            _iter = (var,)
        for v in _iter:
            if not v.fixed:
                vm[id(v)] = v


class OrderedVarRecorder(object):
    def __init__(self, var_map, var_order, sorter):
        self.var_map = var_map
        self.var_order = var_order
        self.sorter = sorter

    def add(self, var):
        # We always add all indices to the var_map at once so that
        # we can honor deterministic ordering of unordered sets
        # (because the user could have iterated over an unordered
        # set when constructing an expression, thereby altering the
        # order in which we would see the variables)
        vm = self.var_map
        vo = self.var_order
        try:
            _iter = var.parent_component().values(self.sorter)
        except AttributeError:
            # Note that this only works for the AML, as kernel does not
            # provide a parent_component()
            _iter = (var,)
        for i, v in enumerate(_iter, start=len(vo)):
            vid = id(v)
            vo[vid] = i
            if not v.fixed:
                vm[vid] = v


class TemplateVarRecorder(object):
    def __init__(self, var_map, sorter):
        self.var_map = var_map
        self._var_order = None
        self.sorter = sorter
        self.env = {None: 0}
        self.symbolmap = EXPR.SymbolMap(NumericLabeler('x'))
        if var_map:
            # If the user provided an initial var_map, we want to honor
            # that ordering.  This means we need to both initialize the
            # env dict with all the Vars referenced, PLUS fill in any
            # additional vars that we would have indexed/recorded in
            # add()
            next_i = len(var_map)
            for i, v in enumerate(list(var_map.values())):
                var_comp = v.parent_component()
                name = self.symbolmap.getSymbol(var_comp)
                ve = self.env.get(name, None)
                if ve is None:
                    ve = self.env[name] = {}
                    for idx, vdata in var_comp.items():
                        vid = id(vdata)
                        if vid not in var_map:
                            var_map[vid] = v
                            ve[idx] = next_i
                            next_i += 1
                ve[v.index()] = i

    @property
    def var_order(self):
        if self._var_order is None:
            self._var_order = {vid: i for i, vid in enumerate(self.var_map)}
        return self._var_order

    def add(self, var):
        # Note: the following is mostly a copy of
        # LinearBeforeChildDispatcher.record_var, but with extra
        # handling to update the env in the same loop
        var_comp = var.parent_component()
        # Double-check that the component has not already been processed
        # (through an individual var data)
        name = self.symbolmap.getSymbol(var_comp)
        if name in self.env:
            return

        # We always add all indices to the var_map at once so that
        # we can honor deterministic ordering of unordered sets
        # (because the user could have iterated over an unordered
        # set when constructing an expression, thereby altering the
        # order in which we would see the variables)
        vm = self.var_map
        ve = self.env[name] = {}
        try:
            _iter = var_comp.items(self.sorter)
        except AttributeError:
            # Note that this only works for the AML, as kernel does not
            # provide a parent_component()
            _iter = (None, var)
        if self._var_order is None:
            for i, (idx, v) in enumerate(_iter, start=len(vm)):
                vm[id(v)] = v
                ve[idx] = i
        else:
            vo = self._var_order
            for i, (idx, v) in enumerate(_iter, start=len(vm)):
                vid = id(v)
                vm[vid] = v
                ve[idx] = i
                vo[vid] = i


# Copied from cpxlp.py:
# Keven Hunter made a nice point about using %.16g in his attachment
# to ticket #4319. I am adjusting this to %.17g as this mocks the
# behavior of using %r (i.e., float('%r'%<number>) == <number>) with
# the added benefit of outputting (+/-). The only case where this
# fails to mock the behavior of %r is for large (long) integers (L),
# which is a rare case to run into and is probably indicative of
# other issues with the model.
# *** NOTE ***: If you use 'r' or 's' here, it will break code that
#               relies on using '%+' before the formatting character
#               and you will need to go add extra logic to output
#               the number's sign.
_ftoa_precision_str = '%.17g'


def ftoa(val, parenthesize_negative_values=False):
    if val is None:
        return val
    #
    # Basic checking, including conversion of *fixed* Pyomo types to floats
    if type(val) in native_numeric_types:
        _val = val
    else:
        if is_fixed(val):
            _val = value(val)
        else:
            raise ValueError(
                "Converting non-fixed bound or value to string: %s" % (val,)
            )
    #
    # Convert to string
    a = _ftoa_precision_str % _val
    #
    # Remove unnecessary least significant digits.  While not strictly
    # necessary, this helps keep the emitted string consistent between
    # python versions by simplifying things like "1.0000000000001" to
    # "1".
    i = len(a) - 1
    if i:
        try:
            while float(a[:i]) == _val:
                i -= 1
        except:
            pass
    i += 1
    #
    # It is important to issue a warning if the conversion loses
    # precision (as the emitted model is not exactly what the user
    # specified)
    if i == len(a) and float(a) != _val:
        logger.warning("Converting %s to string resulted in loss of precision" % val)
    #
    if parenthesize_negative_values and a[0] == '-':
        return '(' + a[:i] + ')'
    else:
        return a[:i]
