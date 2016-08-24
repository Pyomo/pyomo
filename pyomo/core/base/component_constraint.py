#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("IConstraint",
           "constraint",
           "constraint_list",
           "constraint_dict")

import abc

import pyutilib.math

from pyomo.core.base.component_interface import \
    (IComponent,
     _IActiveComponent,
     _IActiveComponentContainer,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.base.component_dict import ComponentDict
from pyomo.core.base.component_list import ComponentList

from pyomo.core.base.numvalue import (ZeroConstant,
                                      is_constant,
                                      as_numeric,
                                      value,
                                      _sub)
from pyomo.core.base import expr as EXPR

import six

class IConstraint(IComponent, _IActiveComponent):
    """
    The interface for optimization constraints.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    body = _abstract_readonly_property(
        doc=("Access the body of the "
             "constraint expression."))
    lower = _abstract_readonly_property(
        doc=("Access the lower bound of the "
             "constraint expression."))
    upper = _abstract_readonly_property(
        doc=("Access the upper bound of the "
             "constraint expression."))
    equality = _abstract_readonly_property(
        doc=("A boolean indicating whether this "
             "is an equality constraint."))
    strict_lower = _abstract_readonly_property(
        doc=("A boolean indicating whether this "
             "constraint has a strict lower bound."))
    strict_upper = _abstract_readonly_property(
        doc=("A boolean indicating whether this "
             "constraint has a strict upper bound."))

    #
    # Interface
    #

    def __call__(self, exception=True):
        """Compute the value of the body of this constraint."""
        if self.body is None:
            return None
        return self.body(exception=exception)

    def lslack(self):
        """
        Returns the value of L-f(x) for constraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        if self.body is None:
            return None
        elif self.lower is None:
            return float('-inf')
        else:
            return value(self.lower)-value(self.body)

    def uslack(self):
        """
        Returns the value of U-f(x) for constraints of the form:
            (L <=) f(x) <= U
            U >= f(x) (>= L)
        """
        if self.body is None:
            return None
        elif self.upper is None:
            return float('inf')
        else:
            return value(self.upper)-value(self.body)

class constraint(IConstraint):
    """An optimization constraint."""
    # To avoid a circular import, for the time being, this
    # property will be set in constraint.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_body",
                 "_lower",
                 "_upper",
                 "_equality",
                 "__weakref__")
    def __init__(self, expr=None):
        self._parent = None
        self._active = True
        self._body = None
        self._lower = None
        self._upper = None
        self._equality = False

        # call the setter
        self.set_expr(expr)

    def set_expr(self, expr):
        """Set the expression on this constraint."""
        if expr is None:
            self._body = None
            self._lower = None
            self._upper = None
            self._equality = False
            return

        _expr_type = expr.__class__
        if _expr_type is tuple: # or expr_type is list:
            #
            # Form equality expression
            #
            if len(expr) == 2:
                arg0 = expr[0]
                if arg0 is not None:
                    arg0 = as_numeric(arg0)
                arg1 = expr[1]
                if arg1 is not None:
                    arg1 = as_numeric(arg1)

                self._equality = True
                if arg1 is None or arg1.is_fixed():
                    self._lower = self._upper = arg1
                    self._body = arg0
                elif arg0 is None or arg0.is_fixed():
                    self._lower = self._upper = arg0
                    self._body = arg1
                else:
                    self._lower = self._upper = ZeroConstant
                    self._body = arg0 - arg1
            #
            # Form inequality expression
            #
            elif len(expr) == 3:
                arg0 = expr[0]
                if arg0 is not None:
                    arg0 = as_numeric(arg0)
                    if not arg0.is_fixed():
                        raise ValueError(
                            "Constraint expression is a 3-tuple (lower,"
                            " expression, upper) but the lower "
                            "value was non-constant.")

                arg1 = expr[1]
                if arg1 is not None:
                    arg1 = as_numeric(arg1)

                arg2 = expr[2]
                if arg2 is not None:
                    arg2 = as_numeric(arg2)
                    if not arg2.is_fixed():
                        raise ValueError(
                            "Constraint expression is a 3-tuple (lower,"
                            " expression, upper) but the upper "
                            "value was non-constant.")

                self._lower = arg0
                self._body  = arg1
                self._upper = arg2
            else:
                raise ValueError(
                    "Can not set constraint expression using "
                    "a tuple of length %d. Expecting a tuple of "
                    "length 2 or 3:\n"
                    "Equality:   (left, right)\n"
                    "Inequality: (lower, expression, upper)"
                    % (len(expr)))

            relational_expr = False
        else:
            try:
                relational_expr = expr.is_relational()
                if not relational_expr:
                    raise ValueError(
                        "Constraint expression does not have a proper "
                        "value. Found '%s'\nExpecting a tuple or "
                        "equation. Examples:"
                        "\n   summation(model.costs) == model.income"
                        "\n   (0, model.price[item], 50)"
                        % (str(expr)))
            except AttributeError:
                msg = ("Constraint expression does not have a proper "
                       "value. Found '%s'\nExpecting a tuple or "
                       "equation. Examples:"
                       "\n   summation(model.costs) == model.income"
                       "\n   (0, model.price[item], 50)"
                       % (str(expr)))
                if type(expr) is bool:
                    msg += ("\nNote: constant Boolean expressions "
                            "are not valid constraint expressions. "
                            "Some apparently non-constant compound "
                            "inequalities (e.g. 'expr >= 0 <= 1') "
                            "can return boolean values; the proper "
                            "form for compound inequalities is "
                            "always 'lb <= expr <= ub'.")
                raise ValueError(msg)
        #
        # Special check for chainedInequality errors like "if var <
        # 1:" within rules.  Catching them here allows us to provide
        # the user with better (and more immediate) debugging
        # information.  We don't want to check earlier because we
        # want to provide a specific debugging message if the
        # construction rule returned True/False; for example, if the
        # user did ( var < 1 > 0 ) (which also results in a non-None
        # chainedInequality value)
        #
        if EXPR.generate_relational_expression.chainedInequality is not None:
            raise TypeError(EXPR.chainedInequalityErrorMessage())
        #
        # Process relational expressions
        # (i.e. explicit '==', '<', and '<=')
        #
        if relational_expr:
            if _expr_type is EXPR._EqualityExpression:
                # Equality expression: only 2 arguments!
                self._equality = True
                try:
                    _args = (expr._lhs, expr._rhs)
                except AttributeError:
                    _args = expr._args
                if _args[1].is_fixed():
                    self._lower = self._upper = _args[1]
                    self._body = _args[0]
                elif _args[0].is_fixed():
                    self._lower = self._upper = _args[0]
                    self._body = _args[1]
                else:
                    self._lower = self._upper = ZeroConstant
                    self._body = \
                        EXPR.generate_expression_bypassCloneCheck(
                            _sub,
                            _args[0],
                            _args[1])
            else:
                # Inequality expression: 2 or 3 arguments
                if expr._strict:
                    try:
                        _strict = \
                            sum(1 if _s else 0
                                for _s in expr._strict) > 0
                    except:
                        _strict = True
                    if _strict:
                        #
                        # We can relax this when:
                        #   (a) we have a need for this
                        #   (b) we have problem writer that
                        #       explicitly handles this
                        #   (c) we make sure that all problem writers
                        #       that don't handle this make it known
                        #       to the user through an error or
                        #       warning
                        #
                        raise ValueError(
                            "Constraint expression is a strict "
                            "inequality expression ('>' or '<'). All"
                            " constraints must be formulated using "
                            "using '<=', '>=', or '=='.")

                try:
                    _args = (expr._lhs, expr._rhs)
                    if expr._lhs.__class__ is \
                       EXPR._InequalityExpression:
                        _args = (expr._lhs._lhs,
                                 expr._lhs._rhs,
                                 expr._rhs)
                    elif expr._lhs.__class__ is \
                         EXPR._InequalityExpression:
                        _args = (expr._lhs,
                                 expr._rhs._lhs,
                                 expr._rhs._rhs)
                except AttributeError:
                    _args = expr._args

                if len(_args) == 3:

                    if not _args[0].is_fixed():
                        raise ValueError(
                            "Constraint expression is a double-sided "
                            "inequality expression (lower <= "
                            "expression <= upper) but the lower "
                            "bound was non-constant.")
                    if not _args[2].is_fixed():
                        raise ValueError(
                            "Constraint expression is a double-sided "\
                            "inequality expression (lower <= "
                            "expression <= upper) but the upper "
                            "bound was non-constant.")
                    self._lower = _args[0]
                    self._body  = _args[1]
                    self._upper = _args[2]

                else:

                    if _args[1].is_fixed():
                        self._lower = None
                        self._body  = _args[0]
                        self._upper = _args[1]
                    elif _args[0].is_fixed():
                        self._lower = _args[0]
                        self._body  = _args[1]
                        self._upper = None
                    else:
                        self._lower = None
                        self._body  = \
                            EXPR.\
                            generate_expression_bypassCloneCheck(
                                _sub,
                                _args[0],
                                _args[1])
                        self._upper = ZeroConstant

        #
        # Replace numeric bound values with a NumericConstant object,
        # and reset the values to 'None' if they are 'infinite'
        #
        if (self._lower is not None) and \
           is_constant(self._lower):
            val = self._lower()
            if not pyutilib.math.is_finite(val):
                if val > 0:
                    raise ValueError(
                        "Constraint expression has a +Inf lower "
                        "bound.")
                self._lower = None
            elif bool(val > 0) == bool(val <= 0):
                raise ValueError(
                    "Constraint expression has a non-numeric "
                    "lower bound.")

        if (self._upper is not None) and \
           is_constant(self._upper):
            val = self._upper()
            if not pyutilib.math.is_finite(val):
                if val < 0:
                    raise ValueError(
                        "Constraint expression has a -Inf upper "
                        "bound.")
                self._upper = None
            elif bool(val > 0) == bool(val <= 0):
                raise ValueError(
                    "Constraint expression has a non-numeric "
                    "upper bound.")

        #
        # Error check, to ensure that we don't have a constraint that
        # doesn't depend on any variables / parameters.
        #
        # Error check, to ensure that we don't have an equality
        # constraint with 'infinite' RHS
        #
        if self._equality:
            if self._lower is None:
                raise ValueError(
                    "Equality constraint expression defined with "
                    "non-finite term.")
            assert self._lower is self._upper

    #
    # Define the IConstraint abstract methods
    #

    @property
    def body(self):
        return self._body
    @property
    def lower(self):
        return self._lower
    @property
    def upper(self):
        return self._upper
    @property
    def equality(self):
        return self._equality
    @property
    def strict_lower(self):
        return False
    @property
    def strict_upper(self):
        return False

class constraint_list(ComponentList,
                      _IActiveComponentContainer):
    """A list-style container for constraints."""
    # To avoid a circular import, for the time being, this
    # property will be set in constraint.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(constraint_list, self).__init__(*args, **kwds)

class constraint_dict(ComponentDict,
                      _IActiveComponentContainer):
    """A dict-style container for constraints."""
    # To avoid a circular import, for the time being, this
    # property will be set in constraint.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(constraint_dict, self).__init__(*args, **kwds)
