#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

# TODO: incorporate changes associated with the
#       inequality-clone-issue branch as this
#       constraint.expr setter is basically
#       a copy of what is in constraint.py
#       (so changes there on master need to be)
#        duplicated here)

__all__ = ("constraint",
           "linear_constraint",
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
                                      potentially_variable,
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
        doc=("The body of the "
             "constraint expression."))
    lb = _abstract_readonly_property(
        doc=("The lower bound of the "
             "constraint expression."))
    ub = _abstract_readonly_property(
        doc=("The upper bound of the "
             "constraint expression."))
    rhs = _abstract_readonly_property(
        doc=("The righthand side of the "
             "constraint expression."))
    equality = _abstract_readonly_property(
        doc=("A boolean indicating whether this "
             "is an equality constraint."))

    _linear_canonical_form = _abstract_readonly_property(
        doc=("Indicates whether or not the class or "
             "instance provides the properties that "
             "define the linear canonical form of a "
             "constraint"))

    # temporary (for backwards compatibility)
    @property
    def lower(self):
        return self.lb
    @property
    def upper(self):
        return self.ub

    #
    # Interface
    #

    def __call__(self, exception=True):
        """Compute the value of the body of this constraint."""
        if self.body is None:
            return None
        return self.body(exception=exception)

    @property
    def lslack(self):
        """Lower slack (body - lb)"""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self()
        if body is None:
            return None
        elif self.lb is None:
            return float('inf')
        else:
            return body - value(self.lb)

    @property
    def uslack(self):
        """Upper slack (ub - body)"""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self()
        if body is None:
            return None
        elif self.ub is None:
            return float('inf')
        else:
            return value(self.ub) - body

    @property
    def slack(self):
        """min(lslack, uslack)"""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self()
        if body is None:
            return None
        elif self.lb is None:
            return self.uslack
        elif self.ub is None:
            return self.lslack
        lslack = self.lslack
        uslack = self.uslack
        return min(lslack, uslack)

    @property
    def expr(self):
        """Get the expression on this constraint."""
        body_expr = self.body
        if body_expr is None:
            return None
        if self.equality:
            return body_expr == self.rhs
        else:
            if self.lb is None:
                return body_expr <= self.ub
            elif self.ub is None:
                return self.lb <= body_expr
            return self.lb <= body_expr <= self.ub

class constraint(IConstraint):
    """An algebraic constraint."""
    # To avoid a circular import, for the time being, this
    # property will be set in constraint.py
    _ctype = None
    _linear_canonical_form = False
    __slots__ = ("_parent",
                 "_active",
                 "_body",
                 "_lb",
                 "_ub",
                 "_equality",
                 "__weakref__")

    def __init__(self,
                 expr=None,
                 lb=None,
                 body=None,
                 ub=None,
                 rhs=None):
        self._parent = None
        self._active = True
        self._body = None
        self._lb = None
        self._ub = None
        self._equality = False

        if expr is not None:
            if body is not None:
                raise ValueError("Both the 'expr' and 'body' "
                                 "keywords can not be used to "
                                 "initialize a constraint.")
            if lb is not None:
                raise ValueError("Both the 'expr' and 'lb' "
                                 "keywords can not be used to "
                                 "initialize a constraint.")
            if ub is not None:
                raise ValueError("Both the 'expr' and 'ub' "
                                 "keywords can not be used to "
                                 "initialize a constraint.")
            if rhs is not None:
                raise ValueError("Both the 'expr' and 'rhs' "
                                 "keywords can not be used to "
                                 "initialize a constraint.")
            # call the setter
            self.expr = expr
        else:
            self.body = body
            if rhs is None:
                self.lb = lb
                self.ub = ub
            else:
                if ((lb is not None) or \
                    (ub is not None)):
                    raise ValueError("The 'rhs' keyword can not "
                                     "be used with the 'lb' or "
                                     "'ub' keywords to initialize"
                                     " a constraint.")
                self.rhs = rhs

    #
    # Define the IConstraint abstract methods
    #

    @property
    def body(self):
        return self._body
    @body.setter
    def body(self, body):
        if body is not None:
            body = as_numeric(body)
        self._body = body

    @property
    def lb(self):
        return self._lb
    @lb.setter
    def lb(self, lb):
        if self.equality:
            raise ValueError(
                "The lb property can not be set "
                "when the equality property is True.")
        if lb is not None:
            if potentially_variable(lb):
                raise ValueError(
                    "Constraint lower bounds must be "
                    "expressions restricted to data.")
        self._lb = lb

    @property
    def ub(self):
        return self._ub
    @ub.setter
    def ub(self, ub):
        if self.equality:
            raise ValueError(
                "The ub property can not be set "
                "when the equality property is True.")
        if ub is not None:
            if potentially_variable(ub):
                raise ValueError(
                    "Constraint lower bounds must be "
                    "expressions restricted to data.")
        self._ub = ub

    @property
    def rhs(self):
        if not self.equality:
            raise ValueError(
                "The rhs property can not be read "
                "when the equality property is False.")
        return self._lb
    @rhs.setter
    def rhs(self, rhs):
        if rhs is not None:
            if potentially_variable(rhs):
                raise ValueError(
                    "Constraint righthand must be "
                    "expressions restricted to data.")
        self._lb = rhs
        self._ub = rhs
        self._equality = True

    @property
    def equality(self):
        return self._equality
    @equality.setter
    def equality(self, equality):
        if equality:
            raise ValueError(
                "The constraint equality flag can "
                "only be set to True by assigning "
                "an expression to the rhs property "
                "(e.g., con.rhs = con.lb).")
        assert not equality
        self._equality = equality

    #
    # Extend the IConstraint interface to allow the
    # expression on this constraint to be changed
    # after construction.
    #

    @property
    def expr(self):
        """Get the expression on this constraint."""
        return super(constraint,self).expr

    @expr.setter
    def expr(self, expr):
        """Set the expression on this constraint."""
        if expr is None:
            self._equality = False
            self.body = None
            self.lb = None
            self.ub = None
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
                if arg1 is None or (not arg1._potentially_variable()):
                    self.rhs = arg1
                    self.body = arg0
                elif arg0 is None or (not arg0._potentially_variable()):
                    self.rhs = arg0
                    self.body = arg1
                else:
                    self.rhs = ZeroConstant
                    self.body = arg0 - arg1
            #
            # Form inequality expression
            #
            elif len(expr) == 3:
                arg0 = expr[0]
                if arg0 is not None:
                    arg0 = as_numeric(arg0)
                    if arg0._potentially_variable():
                        raise ValueError(
                            "Constraint '%s' found a 3-tuple (lower,"
                            " expression, upper) but the lower "
                            "value was not data or an expression "
                            "restricted to storage of data."
                            % (self.name))

                arg1 = expr[1]
                if arg1 is not None:
                    arg1 = as_numeric(arg1)

                arg2 = expr[2]
                if arg2 is not None:
                    arg2 = as_numeric(arg2)
                    if arg2._potentially_variable():
                        raise ValueError(
                            "Constraint '%s' found a 3-tuple (lower,"
                            " expression, upper) but the upper "
                            "value was not data or an expression "
                            "restricted to storage of data."
                            % (self.name))

                self.lb = arg0
                self.body  = arg1
                self.ub = arg2
            else:
                raise ValueError(
                    "Constructor rule for constraint '%s' returned "
                    "a tuple of length %d. Expecting a tuple of "
                    "length 2 or 3:\n"
                    "Equality:   (left, right)\n"
                    "Inequality: (lower, expression, upper)"
                    % (self.name, len(expr)))

            relational_expr = False
        else:
            try:
                relational_expr = expr.is_relational()
                if not relational_expr:
                    raise ValueError(
                        "Constraint '%s' does not have a proper "
                        "value. Found '%s'\nExpecting a tuple or "
                        "equation. Examples:"
                        "\n   summation(model.costs) == model.income"
                        "\n   (0, model.price[item], 50)"
                        % (self.name, str(expr)))
            except AttributeError:
                msg = ("Constraint '%s' does not have a proper "
                       "value. Found '%s'\nExpecting a tuple or "
                       "equation. Examples:"
                       "\n   summation(model.costs) == model.income"
                       "\n   (0, model.price[item], 50)"
                       % (self.name, str(expr)))
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
                if not _args[1]._potentially_variable():
                    self.rhs = _args[1]
                    self.body = _args[0]
                elif not _args[0]._potentially_variable():
                    self.rhs = _args[0]
                    self.body = _args[1]
                else:
                    self.rhs = ZeroConstant
                    self.body = \
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
                            "Constraint '%s' encountered a strict "
                            "inequality expression ('>' or '<'). All"
                            " constraints must be formulated using "
                            "using '<=', '>=', or '=='."
                            % (self.name))

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

                    if _args[0]._potentially_variable():
                        raise ValueError(
                            "Constraint '%s' found a double-sided "
                            "inequality expression (lower <= "
                            "expression <= upper) but the lower "
                            "bound was not data or an expression "
                            "restricted to storage of data."
                            % (self.name))
                    if _args[2]._potentially_variable():
                        raise ValueError(
                            "Constraint '%s' found a double-sided "\
                            "inequality expression (lower <= "
                            "expression <= upper) but the upper "
                            "bound was not data or an expression "
                            "restricted to storage of data."
                            % (self.name))

                    self.lb = _args[0]
                    self.body  = _args[1]
                    self.ub = _args[2]

                else:

                    if not _args[1]._potentially_variable():
                        self.lb = None
                        self.body  = _args[0]
                        self.ub = _args[1]
                    elif not _args[0]._potentially_variable():
                        self.lb = _args[0]
                        self.body  = _args[1]
                        self.ub = None
                    else:
                        self.lb = None
                        self.body = \
                            EXPR.\
                            generate_expression_bypassCloneCheck(
                                _sub,
                                _args[0],
                                _args[1])
                        self.ub = ZeroConstant

        #
        # Replace numeric bound values with a NumericConstant object,
        # and reset the values to 'None' if they are 'infinite'
        #
        if (self.lb is not None) and \
           is_constant(self.lb):
            val = self.lb()
            if not pyutilib.math.is_finite(val):
                if val > 0:
                    raise ValueError(
                        "Constraint '%s' created with a +Inf lower "
                        "bound." % (self.name))
                self.lb = None
            elif bool(val > 0) == bool(val <= 0):
                raise ValueError(
                    "Constraint '%s' created with a non-numeric "
                    "lower bound." % (self.name))

        if (self.ub is not None) and \
           is_constant(self.ub):
            val = self.ub()
            if not pyutilib.math.is_finite(val):
                if val < 0:
                    raise ValueError(
                        "Constraint '%s' created with a -Inf upper "
                        "bound." % (self.name))
                self.ub = None
            elif bool(val > 0) == bool(val <= 0):
                raise ValueError(
                    "Constraint '%s' created with a non-numeric "
                    "upper bound." % (self.name))

        #
        # Error check, to ensure that we don't have a constraint that
        # doesn't depend on any variables / parameters.
        #
        # Error check, to ensure that we don't have an equality
        # constraint with 'infinite' RHS
        #
        if self._equality:
            if self.lb is None:
                raise ValueError(
                    "Equality constraint '%s' defined with "
                    "non-finite term." % (self.name))
            assert self.lb is self.ub

class linear_constraint(IConstraint):
    """
    A linear constraint defined by a list of variables
    and coefficients
    """
    # To avoid a circular import, for the time being, this
    # property will be set in constraint.py
    _ctype = None
    _linear_canonical_form = True
    __slots__ = ("_parent",
                 "_active",
                 "_variables",
                 "_coefficients",
                 "_lb",
                 "_ub",
                 "_equality",
                 "__weakref__")

    def __init__(self,
                 variables,
                 coefficients,
                 lb=None,
                 ub=None,
                 rhs=None):
        self._parent = None
        self._active = True
        self._variables = variables
        self._coefficients = coefficients
        self._lb = None
        self._ub = None
        self._equality = False

        if type(self._variables) is not tuple:
            self._variables = tuple(self._variables)
        if type(self._coefficients) is not tuple:
            self._coefficients = tuple(self._coefficients)

        if rhs is None:
            self.lb = lb
            self.ub = ub
        else:
            if ((lb is not None) or \
                (ub is not None)):
                raise ValueError("The 'rhs' keyword can not "
                                 "be used with the 'lb' or "
                                 "'ub' keywords to initialize"
                                 " a constraint.")
            self.rhs = rhs

    @property
    def terms(self):
        """The linear terms in the body of this constraint
        as (variable, coefficient) tuples"""
        return tuple(zip(self._variables, self._coefficients))
    @terms.setter
    def terms(self, terms):
        variables = []
        coefficients = []
        for v, c in terms:
            variables.append(v)
            coefficients.append(c)
        self._coefficients = tuple(coefficients)
        self._variables = tuple(variables)

    #
    # Define the IConstraint abstract methods
    #

    @property
    def body(self):
        return sum(c * v for c,v in zip(self._coefficients,
                                        self._variables))

    @property
    def lb(self):
        return self._lb
    @lb.setter
    def lb(self, lb):
        if self.equality:
            raise ValueError(
                "The lb property can not be set "
                "when the equality property is True.")
        if lb is not None:
            if potentially_variable(lb):
                raise ValueError(
                    "Constraint lower bounds must be "
                    "expressions restricted to data.")
        self._lb = lb

    @property
    def ub(self):
        return self._ub
    @ub.setter
    def ub(self, ub):
        if self.equality:
            raise ValueError(
                "The ub property can not be set "
                "when the equality property is True.")
        if ub is not None:
            if potentially_variable(ub):
                raise ValueError(
                    "Constraint lower bounds must be "
                    "expressions restricted to data.")
        self._ub = ub

    @property
    def rhs(self):
        if not self.equality:
            raise ValueError(
                "The rhs property can not be read "
                "when the equality property is False.")
        return self._lb
    @rhs.setter
    def rhs(self, rhs):
        if rhs is not None:
            if potentially_variable(rhs):
                raise ValueError(
                    "Constraint righthand must be "
                    "expressions restricted to data.")
        self._lb = rhs
        self._ub = rhs
        self._equality = True

    @property
    def equality(self):
        return self._equality
    @equality.setter
    def equality(self, equality):
        if equality:
            raise ValueError(
                "The constraint equality flag can "
                "only be set to True by assigning "
                "an expression to the rhs property "
                "(e.g., con.rhs = con.lb).")
        assert not equality
        self._equality = equality

    #
    # Override a the default __call__ method on IConstraint
    # to avoid calling building the body expression
    #

    def __call__(self, exception=True):
        try:
            return sum(value(c) * v() for c,v in zip(self._coefficients,
                                                     self._variables))
        except (ValueError, TypeError):
            if exception:
                raise
            return None

    #
    # Define the LinearCanonicalRepn abstract methods
    #

    @property
    def variables(self):
        return tuple(v for v in self._variables
                     if not v.fixed)

    @property
    def coefficients(self):
        return tuple(c for c,v in zip(self._coefficients,
                                      self._variables)
                     if not v.fixed)

    # for backwards compatibility
    linear=coefficients

    @property
    def constant(self):
        terms = tuple(value(c) * v() for c,v in zip(self._coefficients,
                                                    self._variables)
                      if v.fixed)
        if len(terms) == 0:
            return None
        return sum(terms)

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
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
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
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        self._active = True
        super(constraint_dict, self).__init__(*args, **kwds)
