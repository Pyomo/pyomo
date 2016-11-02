#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("expression",
           "data_expression",
           "expression_list",
           "expression_dict")

import sys
import abc

import pyutilib.math

import pyomo.core.base.expr_common
from pyomo.core.base.component_interface import \
    (IComponent,
     _IActiveComponent,
     _IActiveComponentContainer,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.base.component_dict import ComponentDict
from pyomo.core.base.component_list import ComponentList
from pyomo.core.base.numvalue import (NumericValue,
                                      is_fixed,
                                      potentially_variable,
                                      as_numeric)

import six

class IExpression(IComponent, NumericValue):
    """
    The interface for reusable expressions.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    expr = _abstract_readwrite_property(
        doc="Access the stored expression")

    #
    # Implement the NumericValue abstract methods
    #

    def __call__(self, exception=True):
        """Compute the value of this expression."""
        if self.expr is None:
            return None
        return self.expr(exception=exception)

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        return False

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        return is_fixed(self.expr)

    def _potentially_variable(self):
        """A boolean indicating whether this expression can
        reference variables."""
        return True

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
        """The polynomial degree of the stored expression."""
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
            self.expr.to_string(ostream=ostream,
                                verbose=verbose,
                                precedence=precedence)
        if _verbose:
            ostream.write("}")

    @property
    def _parent_expr(self):
        return None
    @_parent_expr.setter
    def _parent_expr(self, value):
        raise NotImplementedError

class expression(IExpression):
    """A reusable expression."""
    # To avoid a circular import, for the time being, this
    # property will be set in expression.py
    _ctype = None
    __slots__ = ("_parent",
                 "_expr",
                 "__weakref__")
    def __init__(self, expr=None):
        self._parent = None
        self._expr = None

        # call the setter
        self.expr = expr

    #
    # Define the IExpression abstract methods
    #

    @property
    def expr(self):
        return self._expr
    @expr.setter
    def expr(self, expr):
        self._expr = as_numeric(expr) if (expr is not None) else None

class data_expression(expression):
    """A reusable expression that is restricted to storage
    of data expressions."""

    #
    # Overload a few methods
    #

    def _potentially_variable(self):
        """A boolean indicating whether this expression can
        reference variables."""
        return False

    def polynomial_degree(self):
        """Always return zero because we always validate
        that the stored expression can never reference
        variables."""
        return 0

    def _polynomial_degree(self, result):
        return results.pop()

    @property
    def expr(self):
        return self._expr
    @expr.setter
    def expr(self, expr):
        if potentially_variable(expr):
            raise ValueError("Expression is not restricted to data.")
        self._expr = as_numeric(expr) if (expr is not None) else None

class expression_list(ComponentList):
    """A list-style container for expressions."""
    # To avoid a circular import, for the time being, this
    # property will be set in expression.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        super(expression_list, self).__init__(*args, **kwds)

class expression_dict(ComponentDict):
    """A dict-style container for expressions."""
    # To avoid a circular import, for the time being, this
    # property will be set in expression.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        super(expression_dict, self).__init__(*args, **kwds)
