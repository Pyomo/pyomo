#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['Expression', '_ExpressionData']

import sys
import logging
from weakref import ref as weakref_ref

from pyomo.core.base.component import (ComponentData,
                                       register_component)
from pyomo.core.base.indexed_component import (IndexedComponent,
                                               normalize_index)
from pyomo.core.base.misc import (apply_indexed_rule,
                                  tabular_writer)
from pyomo.core.base.numvalue import (NumericValue,
                                      as_numeric)
import pyomo.core.base.expr
from pyomo.core.base.util import is_functor

from six import iteritems

logger = logging.getLogger('pyomo.core')

#
# This class is a pure interface
#

class _ExpressionData(ComponentData, NumericValue):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        owner       The Expression that owns this data.

    Public Class Attributes
        expr       The expression owned by this data.
    """

    __slots__ = ()

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None

    #
    # Interface
    #

    def __call__(self, exception=True):
        """Compute the value of this expression."""
        if self.expr is None:
            return None
        return self.expr(exception=exception)

    #
    # Ducktyping _ExpressionBase functionality
    #

    def is_expression(self):
        """A boolean indicating whether this in an expression."""
        return True

    @property
    def _args(self):
        """A tuple of subexpressions involved in this expressions operation."""
        return (self.expr,)

    def _arguments(self):
        """A generator of subexpressions involved in this expressions operation."""
        yield self.expr

    def clone(self):
        """Return a clone of this expression (no-op)."""
        return self

    def polynomial_degree(self):
        """A tuple of subexpressions involved in this expressions operation."""
        return self.expr.polynomial_degree()

    def _polynomial_degree(self, result):
        return result.pop()

    def to_string(self, ostream=None, verbose=None, precedence=0):
        if ostream is None:
            ostream = sys.stdout
        _verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE if \
            verbose is None else verbose
        if _verbose:
            ostream.write(str(self))
            ostream.write("{")
        if self.expr is None:
            ostream.write("Undefined")
        else:
            self.expr.to_string( ostream=ostream, verbose=verbose,
                                   precedence=precedence )
        if _verbose:
            ostream.write("}")

    @property
    def _parent_expr(self):
        return None
    @_parent_expr.setter
    def _parent_expr(self, value):
        raise NotImplementedError

    #
    # Abstract Interface
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        raise NotImplementedError

    def set_value(self, expr):
        """Set the expression on this expression."""
        raise NotImplementedError

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        raise NotImplementedError

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        raise NotImplementedError

class _GeneralExpressionData(_ExpressionData, NumericValue):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        owner       The Expression that owns this data.

    Public Class Attributes
        expr       The expression owned by this data.
    """

    __slots__ = ('_expr',)

    def __init__(self, expr, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ExpressionData
        #   - ComponentData
        #   - NumericValue
        self._component = weakref_ref(component) if (component is not None) \
                          else None

        self._expr = as_numeric(expr) if (expr is not None) else None

    def __getstate__(self):
        state = super(_GeneralExpressionData, self).__getstate__()
        for i in _GeneralExpressionData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: None of the slots on this class need to be edited, so we
    # don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

    #
    # Abstract Interface
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        return self._expr

    # for backwards compatibility reasons
    @property
    def value(self):
        return self._expr
    @value.setter
    def value(self, expr):
        logger.warn("DEPRECATED: The .value setter method on "
                    "_GeneralExpressionData is deprecated. Use "
                    "the set_value(expr) method instead")
        self.set_value(expr)

    def set_value(self, expr):
        """Set the expression on this expression."""
        self._expr = as_numeric(expr) if (expr is not None) else None

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        # The underlying expression can always be changed
        # so this should never evaluate as constant
        return False

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        return self._expr.is_fixed()

class Expression(IndexedComponent):
    """
    A shared expression container, which may be defined over a index.

    Constructor Arguments:
        initialize  A Pyomo expression or dictionary of expressions
                        used to initialize this object.
        expr        A synonym for initialize.
        rule        A rule function used to initialize this object.
    """

    def __new__(cls, *args, **kwds):
        if cls != Expression:
            return super(Expression, cls).__new__(cls)
        if args == ():
            return SimpleExpression.__new__(SimpleExpression)
        else:
            return IndexedExpression.__new__(IndexedExpression)

    def __init__(self, *args, **kwds):
        self._init_rule = kwds.pop('rule', None)
        self._init_expr = kwds.pop('initialize', None)
        self._init_expr = kwds.pop('expr', self._init_expr)
        if is_functor(self._init_expr) and \
           (not isinstance(self._init_expr, NumericValue)):
            raise TypeError(
                "A callable type that is not a Pyomo "
                "expression can not be used to initialize "
                "an Expression object. Use 'rule' to initalize "
                "with function types.")
        if (self._init_rule is not None) and \
           (self._init_expr is not None):
            raise ValueError(
                "Both a rule and an expression can not be "
                "used to initialized an Expression object")

        kwds.setdefault('ctype', Expression)
        IndexedComponent.__init__(self, *args, **kwds)

    def _pprint(self):
        return (
            [('Size', len(self)),
             ('Index', None if (not self.is_indexed())
                  else self._index)
             ],
            self.iteritems(),
            ("Key","Expression"),
            lambda k,v: \
               [k, "Undefined" if v.expr is None else v.expr]
            )

    def display(self, prefix="", ostream=None):
        """TODO"""
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab="    "
        ostream.write(prefix+self.cname()+" : ")
        ostream.write("Size="+str(len(self)))

        ostream.write("\n")
        tabular_writer(
            ostream,
            prefix+tab,
            ((k,v) for k,v in iteritems(self._data)),
            ( "Key","Value" ),
            lambda k, v: \
               [k, "Undefined" if v.expr is None else v()])

    #
    # A utility to extract all index-value pairs defining this
    # expression, returned as a dictionary. useful in many contexts,
    # in which key iteration and repeated __getitem__ calls are too
    # expensive to extract the contents of an expression.
    #
    def extract_values(self):
        return dict((key, expression_data.expr) \
                    for key, expression_data in iteritems(self))

    #
    # takes as input a (index, value) dictionary for updating this
    # Expression.  if check=True, then both the index and value are
    # checked through the __getitem__ method of this class.
    #
    def store_values(self, new_values):

        if (self.is_indexed() is False) and \
           (not None in new_values):
            raise RuntimeError(
                "Cannot store value for scalar Expression"
                "="+self.cname(True)+"; no value with index "
                "None in input new values map.")

        for index, new_value in iteritems(new_values):
            self._data[index].set_value(new_value)

    def __setitem__(self, ndx, val):
        #
        # Get the expression data object
        #
        if ndx in self._data:
            exprdata = self._data[ndx]
        else:
            _ndx = normalize_index(ndx)
            if _ndx in self._data:
                exprdata = self._data[_ndx]
            else:
                raise KeyError(
                    "Cannot set the value of Expression '%s' with "
                    "invalid index '%s'"
                    % (self.cname(True), str(ndx)))
        #
        # Set the value
        #
        exprdata.set_value(val)

    def construct(self, data=None):
        """ Apply the rule to construct values in this set """

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Constructing Expression, name=%s, from data=%s"
                % (self.cname(True), str(data)))

        if self._constructed:
            return
        self._constructed = True

        _init_expr = self._init_expr
        _init_rule = self._init_rule
        #
        # We no longer need these
        #
        self._init_expr = None
        self._init_rule = None

        #
        # Construct _GeneralExpressionData objects for all index values
        #
        if self.is_indexed():
            self._data.update(
                (key, _GeneralExpressionData(None, component=self))
                for key in self._index)
        else:
            self._data[None] = self

        #
        # Initialize members
        #
        if _init_expr is not None:
            #
            # Initialize values with a value
            #
            if _init_expr.__class__ is dict:
                for key in self._index:
                    # Skip indices that are not in the dictionary
                    if not key in _init_expr:
                        continue
                    self._data[key].set_value(_init_expr[key])
            else:
                for key in self._index:
                    self._data[key].set_value(_init_expr)

        elif _init_rule is not None:
            #
            # Initialize values with a rule
            #
            if self.is_indexed():
                for key in self._index:
                    self._data[key].set_value(
                        apply_indexed_rule(self,
                                           _init_rule,
                                           self._parent(),
                                           key))
            else:
                self.set_value(_init_rule(self._parent()))

class SimpleExpression(_GeneralExpressionData, Expression):

    def __init__(self, *args, **kwds):
        _GeneralExpressionData.__init__(self, None, component=self)
        Expression.__init__(self, *args, **kwds)

    #
    # Since this class derives from Component and
    # Component.__getstate__ just packs up the entire __dict__ into
    # the state dict, we do not need to define the __getstate__ or
    # __setstate__ methods.  We just defer to the super() get/set
    # state.  Since all of our get/set state methods rely on super()
    # to traverse the MRO, this will automatically pick up both the
    # Component and Data base classes.
    #

    #
    # Override abstract interface methods to first check for
    # construction
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        if self._constructed:
            return _GeneralExpressionData.expr.fget(self)
        raise ValueError(
            "Accessing the expression of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)."
            % (self.cname(True)))

    # for backwards compatibility reasons
    @property
    def value(self):
        return self.expr
    @value.setter
    def value(self, expr):
        logger.warn("DEPRECATED: The .value setter method on "
                    "SimpleExpression is deprecated. Use the "
                    "set_value(expr) method instead")
        self.set_value(expr)

    def set_value(self, expr):
        """Set the expression on this expression."""
        if self._constructed:
            return _GeneralExpressionData.set_value(self, expr)
        raise ValueError(
            "Setting the expression of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no object to set)."
            % (self.cname(True)))

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        if self._constructed:
            return _GeneralExpressionData.is_constant(self)
        raise ValueError(
            "Accessing the is_constant flag of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)."
            % (self.cname(True)))

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        if self._constructed:
            return _GeneralExpressionData.is_fixed(self)
        raise ValueError(
            "Accessing the is_fixed flag of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)."
            % (self.cname(True)))

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise ValueError(
                "SimpleExpression object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.cname(True), index))
        self.set_value(expr)
        return self

class IndexedExpression(Expression):

    #
    # Leaving this method for backward compatibility reasons
    # Note: It allows adding members outside of self._index.
    #       This has always been the case. Not sure there is
    #       any reason to maintain a reference to a separate
    #       index set if we allow this.
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        cdata = _GeneralExpressionData(expr, component=self)
        self._data[index] = cdata
        return cdata

register_component(
    Expression,
    "Named expressions that can be used in other expressions.")
