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

import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload

from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import RenamedClass
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer
from pyomo.common.numeric_types import (
    native_types,
    native_numeric_types,
    check_if_numeric_type,
)

import pyomo.core.expr as EXPR
from pyomo.core.expr.expr_common import _type_check_exception_arg
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
    IndexedComponent,
    UnindexedComponent_set,
    IndexedComponent_NDArrayMixin,
)
from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.base.initializer import Initializer

logger = logging.getLogger('pyomo.core')


class NamedExpressionData(numeric_expr.NumericValue):
    """An object that defines a generic "named expression".

    This is the base class for both :class:`ExpressionData` and
    :class:`ObjectiveData`.
    """

    # Note: derived classes are expected to declare the _args_ slot
    __slots__ = ()

    EXPRESSION_SYSTEM = EXPR.ExpressionType.NUMERIC
    PRECEDENCE = 0
    ASSOCIATIVITY = EXPR.OperatorAssociativity.NON_ASSOCIATIVE

    def __call__(self, exception=NOTSET):
        """Compute the value of this expression."""
        exception = _type_check_exception_arg(self, exception)
        (arg,) = self.args
        if arg.__class__ in native_types:
            # Note: native_types includes NoneType
            return arg
        return arg(exception=exception)

    def create_node_with_local_data(self, values, classtype=None):
        """
        Construct a simple expression after constructing the
        contained expression.

        This class provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.
        """
        if classtype is None:
            classtype = self.parent_component()._ComponentDataClass
        obj = classtype()
        obj._args_ = values
        return obj

    def is_named_expression_type(self):
        """A boolean indicating whether this in a named expression."""
        return True

    def is_expression_type(self, expression_system=None):
        """A boolean indicating whether this in an expression."""
        return expression_system is None or expression_system == self.EXPRESSION_SYSTEM

    def arg(self, index):
        if index != 0:
            raise KeyError("Invalid index for expression argument: %d" % index)
        return self.args[0]

    @property
    def args(self):
        return self._args_

    def nargs(self):
        return 1

    def _to_string(self, values, verbose, smap):
        if verbose:
            return "%s{%s}" % (str(self), values[0])
        if self.args[0] is None:
            return "%s{None}" % str(self)
        return values[0]

    def clone(self):
        """Return a clone of this expression (no-op)."""
        return self

    def _apply_operation(self, result):
        # This "expression" is a no-op wrapper, so just return the inner
        # result
        return result[0]

    def polynomial_degree(self):
        """A tuple of subexpressions involved in this expressions operation."""
        if self.args[0] is None:
            return None
        return self.expr.polynomial_degree()

    def _compute_polynomial_degree(self, result):
        return result[0]

    def _is_fixed(self, values):
        return values[0]

    # NamedExpressionData should never return False because
    # they can store subexpressions that contain variables
    def is_potentially_variable(self):
        return True

    @property
    def expr(self):
        (arg,) = self.args
        if arg is None:
            return None
        return as_numeric(arg)

    @expr.setter
    def expr(self, value):
        self.set_value(value)

    def set_value(self, expr):
        """Set the expression on this expression."""
        if expr is None or expr.__class__ in native_numeric_types:
            self._args_ = (expr,)
            return
        try:
            if expr.is_numeric_type():
                self._args_ = (expr,)
                return
        except AttributeError:
            if check_if_numeric_type(expr):
                self._args_ = (expr,)
                return
        raise ValueError(
            f"Cannot assign {expr.__class__.__name__} to "
            f"'{self.name}': {self.__class__.__name__} components only "
            "allow numeric expression types."
        )

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        # The underlying expression can always be changed
        # so this should never evaluate as constant
        return False

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        (e,) = self.args
        return e.__class__ in native_types or e.is_fixed()

    # Override the in-place operators here so that we can redirect the
    # dispatcher based on the current contained expression type and not
    # this Expression object (which would map to "other")

    def __iadd__(self, other):
        (e,) = self.args
        return numeric_expr._add_dispatcher[e.__class__, other.__class__](e, other)

    # Note: the default implementation of __isub__ leverages __iadd__
    # and doesn't need to be reimplemented here

    def __imul__(self, other):
        (e,) = self.args
        return numeric_expr._mul_dispatcher[e.__class__, other.__class__](e, other)

    def __idiv__(self, other):
        (e,) = self.args
        return numeric_expr._div_dispatcher[e.__class__, other.__class__](e, other)

    def __itruediv__(self, other):
        (e,) = self.args
        return numeric_expr._div_dispatcher[e.__class__, other.__class__](e, other)

    def __ipow__(self, other):
        (e,) = self.args
        return numeric_expr._pow_dispatcher[e.__class__, other.__class__](e, other)


class _ExpressionData(metaclass=RenamedClass):
    __renamed__new_class__ = NamedExpressionData
    __renamed__version__ = '6.7.2'


class _GeneralExpressionDataImpl(metaclass=RenamedClass):
    __renamed__new_class__ = NamedExpressionData
    __renamed__version__ = '6.7.2'


class ExpressionData(NamedExpressionData, ComponentData):
    """An object that defines an expression that is never cloned

    Parameters
    ----------
    expr : NumericValue
        The Pyomo expression stored in this expression.

    component : Expression
        The Expression object that owns this data.

    """

    __slots__ = ('_args_',)

    def __init__(self, expr=None, component=None):
        self._args_ = (expr,)
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET


class _GeneralExpressionData(metaclass=RenamedClass):
    __renamed__new_class__ = ExpressionData
    __renamed__version__ = '6.7.2'


@ModelComponentFactory.register(
    "Named expressions that can be used in other expressions."
)
class Expression(IndexedComponent, IndexedComponent_NDArrayMixin):
    """A shared expression container, which may be defined over an index.

    Parameters
    ----------
    rule : ~.Initializer

        The source to use to initialize the expression(s) in this
        component.  See :func:`.Initializer` for accepted argument types.

    initialize :
        A synonym for `rule`

    expr :
        A synonym for `rule`

    name : str
        Name of this component; will be overridden if this is assigned
        to a Block.

    doc : str
        Text describing this component.

    """

    _ComponentDataClass = ExpressionData
    # This seems like a copy-paste error, and should be renamed/removed
    NoConstraint = IndexedComponent.Skip

    def __new__(cls, *args, **kwds):
        if cls != Expression:
            return super(Expression, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return ScalarExpression.__new__(ScalarExpression)
        else:
            return IndexedExpression.__new__(IndexedExpression)

    @overload
    def __init__(
        self, *indexes, rule=None, expr=None, initialize=None, name=None, doc=None
    ): ...

    def __init__(self, *args, **kwds):
        _init = self._pop_from_kwargs(
            'Expression', kwds, ('rule', 'expr', 'initialize'), None
        )
        # Historically, Expression objects were dense (but None):
        # setting arg_not_specified causes Initializer to recognize
        # _init==None as a constant initializer returning None
        #
        # To initialize a completely empty Expression, pass either
        # initialize={} (to require explicit setitem before a getitem),
        # or initialize=NOTSET (to allow getitem before setitem)
        self._rule = Initializer(_init, arg_not_specified=NOTSET)

        kwds.setdefault('ctype', Expression)
        IndexedComponent.__init__(self, *args, **kwds)

    def _pprint(self):
        return (
            [
                ('Size', len(self)),
                ('Index', None if (not self.is_indexed()) else self._index_set),
            ],
            self.items(),
            ("Expression",),
            lambda k, v: ["Undefined" if v.expr is None else v.expr],
        )

    def display(self, prefix="", ostream=None):
        """TODO"""
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab = "    "
        ostream.write(prefix + self.local_name + " : ")
        ostream.write("Size=" + str(len(self)))

        ostream.write("\n")
        tabular_writer(
            ostream,
            prefix + tab,
            ((k, v) for k, v in self._data.items()),
            ("Value",),
            lambda k, v: ["Undefined" if v.expr is None else v()],
        )

    #
    # A utility to extract all index-value pairs defining this
    # expression, returned as a dictionary. useful in many contexts,
    # in which key iteration and repeated __getitem__ calls are too
    # expensive to extract the contents of an expression.
    #
    def extract_values(self):
        return {key: expression_data.expr for key, expression_data in self.items()}

    #
    # takes as input a (index, value) dictionary for updating this
    # Expression.  if check=True, then both the index and value are
    # checked through the __getitem__ method of this class.
    #
    def store_values(self, new_values):
        if (self.is_indexed() is False) and (not None in new_values):
            raise KeyError(
                "Cannot store value for scalar Expression"
                "=" + self.name + "; no value with index "
                "None in input new values map."
            )

        for index, new_value in new_values.items():
            self._data[index].set_value(new_value)

    def _getitem_when_not_present(self, idx):
        if self._rule is None:
            _init = None
            # TBD: Is this desired behavior?  I can see implicitly setting
            # an Expression if it was not originally defined, but I am less
            # convinced that implicitly creating an Expression (like what
            # works with a Var) makes sense.  [JDS 25 Nov 17]
            # raise KeyError(idx)
        else:
            _init = self._rule(self.parent_block(), idx)
            if _init is Expression.Skip:
                raise KeyError(idx)
        return self._setitem_when_not_present(idx, _init)

    def construct(self, data=None):
        """Apply the rule to construct values in this set"""
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug(
                "Constructing Expression, name=%s, from data=%s"
                % (self.name, str(data))
            )

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        try:
            # We do not (currently) accept data for constructing Constraints
            assert data is None
            self._construct_from_rule_using_setitem()
        finally:
            timer.report()


class ScalarExpression(ExpressionData, Expression):
    def __init__(self, *args, **kwds):
        ExpressionData.__init__(self, expr=None, component=self)
        Expression.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    #
    # Override abstract interface methods to first check for
    # construction
    #

    def __call__(self, exception=NOTSET):
        """Return expression on this expression."""
        exception = _type_check_exception_arg(self, exception)
        if self._constructed:
            return super().__call__(exception)
        raise ValueError(
            "Evaluating the expression of Expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    @property
    def expr(self):
        """Return expression on this expression."""
        if self._constructed:
            return ExpressionData.expr.fget(self)
        raise ValueError(
            "Accessing the expression of Expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    @expr.setter
    def expr(self, expr):
        """Set the expression on this expression."""
        self.set_value(expr)

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression on this expression."""
        if self._constructed:
            return ExpressionData.set_value(self, expr)
        raise ValueError(
            "Setting the expression of Expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no object to set)." % (self.name)
        )

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        if self._constructed:
            return ExpressionData.is_constant(self)
        raise ValueError(
            "Accessing the is_constant flag of Expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        if self._constructed:
            return ExpressionData.is_fixed(self)
        raise ValueError(
            "Accessing the is_fixed flag of Expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)." % (self.name)
        )

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise KeyError(
                "ScalarExpression object '%s' does not accept "
                "index values other than None. Invalid value: %s" % (self.name, index)
            )
        if (type(expr) is tuple) and (expr == Expression.Skip):
            raise ValueError(
                "Expression.Skip can not be assigned "
                "to an Expression that is not indexed: %s" % (self.name)
            )
        self.set_value(expr)
        return self


class SimpleExpression(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarExpression
    __renamed__version__ = '6.0'


class IndexedExpression(Expression):
    #
    # Leaving this method for backward compatibility reasons
    # Note: It allows adding members outside of self._index_set.
    #       This has always been the case. Not sure there is
    #       any reason to maintain a reference to a separate
    #       index set if we allow this.
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if (type(expr) is tuple) and (expr == Expression.Skip):
            return None
        cdata = ExpressionData(expr, component=self)
        self._data[index] = cdata
        return cdata
