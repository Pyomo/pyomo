#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import (ZeroConstant,
                                      as_numeric,
                                      is_potentially_variable,
                                      is_numeric_data,
                                      value)
from pyomo.core.expr import logical_expr
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     _abstract_readonly_property)
from pyomo.core.kernel.container_utils import \
    define_simple_containers

from six.moves import zip

_pos_inf = float('inf')
_neg_inf = float('-inf')

class IConstraint(ICategorizedObject):
    """The interface for constraints"""
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    body = _abstract_readonly_property(
        doc="The body of the constraint")
    lb = _abstract_readonly_property(
        doc="The lower bound of the constraint")
    ub = _abstract_readonly_property(
        doc="The upper bound of the constraint")
    rhs = _abstract_readonly_property(
        doc="The right-hand side of the constraint")
    equality = _abstract_readonly_property(
        doc=("A boolean indicating whether this "
             "is an equality constraint"))

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
        if exception and (self.body is None):
            raise ValueError("constraint body is None")
        elif self.body is None:
            return None
        return self.body(exception=exception)

    @property
    def lslack(self):
        """Lower slack (body - lb). Returns :const:`None` if
        a value for the body can not be computed."""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self(exception=False)
        if body is None:
            return None
        lb = self.lb
        if lb is None:
            lb = _neg_inf
        else:
            lb = value(lb)
        return body - lb

    @property
    def uslack(self):
        """Upper slack (ub - body). Returns :const:`None` if
        a value for the body can not be computed."""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self(exception=False)
        if body is None:
            return None
        ub = self.ub
        if ub is None:
            ub = _pos_inf
        else:
            ub = value(ub)
        return ub - body

    @property
    def slack(self):
        """min(lslack, uslack). Returns :const:`None` if a
        value for the body can not be computed."""
        # this method is written so that constraint
        # types that build the body expression on the
        # fly do not have to here
        body = self(exception=False)
        if body is None:
            return None
        return min(self.lslack, self.uslack)

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
            return logical_expr.RangedExpression((self.lb, body_expr, self.ub), (False, False))

    @property
    def bounds(self):
        """The bounds of the constraint as a tuple (lb, ub)"""
        return (self.lb, self.ub)

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        lb = self.lb
        return (lb is not None) and \
            (value(lb) != float('-inf'))

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        ub = self.ub
        return (ub is not None) and \
            (value(ub) != float('inf'))

class _MutableBoundsConstraintMixin(object):
    """
    Use as a base class for IConstraint implementations
    that allow adjusting the lb, ub, rhs, and equality
    properties.

    Assumes the derived class has _lb, _ub, and _equality
    attributes that can be modified.
    """
    __slots__ = ()

    #
    # Define some of the IConstraint abstract methods
    #

    @property
    def lb(self):
        """The lower bound of the constraint"""
        return self._lb
    @lb.setter
    def lb(self, lb):
        if self.equality:
            raise ValueError(
                "The lb property can not be set "
                "when the equality property is True.")
        if (lb is not None) and \
           (not is_numeric_data(lb)):
            raise TypeError(
                    "Constraint lower bounds must be "
                    "expressions restricted to numeric data.")
        self._lb = lb

    @property
    def ub(self):
        """The upper bound of the constraint"""
        return self._ub
    @ub.setter
    def ub(self, ub):
        if self.equality:
            raise ValueError(
                "The ub property can not be set "
                "when the equality property is True.")
        if (ub is not None) and \
           (not is_numeric_data(ub)):
            raise TypeError(
                    "Constraint upper bounds must be "
                    "expressions restricted to numeric data.")
        self._ub = ub

    @property
    def rhs(self):
        """The right-hand side of the constraint"""
        if not self.equality:
            raise ValueError(
                "The rhs property can not be read "
                "when the equality property is False.")
        return self._lb
    @rhs.setter
    def rhs(self, rhs):
        if rhs is None:
            # None has a different meaning depending on the
            # context (lb or ub), so there is no way to
            # interpret this
            raise ValueError(
                "Constraint right-hand side can not "
                "be assigned a value of None.")
        elif not is_numeric_data(rhs):
            raise TypeError(
                    "Constraint right-hand side must be numbers "
                    "or expressions restricted to data.")
        self._lb = rhs
        self._ub = rhs
        self._equality = True

    @property
    def bounds(self):
        """The bounds of the constraint as a tuple (lb, ub)"""
        return super(_MutableBoundsConstraintMixin, self).bounds
    @bounds.setter
    def bounds(self, bounds_tuple):
        self.lb, self.ub = bounds_tuple

    @property
    def equality(self):
        """Returns :const:`True` when this is an equality
        constraint.

        Disable equality by assigning
        :const:`False`. Equality can only be activated by
        assigning a value to the .rhs property."""
        return self._equality
    @equality.setter
    def equality(self, equality):
        if equality:
            raise ValueError(
                "The constraint equality flag can "
                "only be set to True by assigning "
                "a value to the rhs property "
                "(e.g., con.rhs = con.lb).")
        assert not equality
        self._equality = False

class constraint(_MutableBoundsConstraintMixin,
                 IConstraint):
    """A general algebraic constraint

    Algebraic constraints store relational expressions
    composed of linear or nonlinear functions involving
    decision variables.

    Args:
        expr: Sets the relational expression for the
            constraint. Can be updated later by assigning to
            the :attr:`expr` property on the
            constraint. When this keyword is used, values
            for the :attr:`body`, :attr:`lb`, :attr:`ub`,
            and :attr:`rhs` attributes are automatically
            determined based on the relational expression
            type. Default value is :const:`None`.
        body: Sets the body of the constraint. Can be
            updated later by assigning to the :attr:`body`
            property on the constraint. Default is
            :const:`None`. This keyword should not be used
            in combination with the :attr:`expr` keyword.
        lb: Sets the lower bound of the constraint. Can be
            updated later by assigning to the :attr:`lb`
            property on the constraint. Default is
            :const:`None`, which is equivalent to
            :const:`-inf`. This keyword should not be used
            in combination with the :attr:`expr` keyword.
        ub: Sets the upper bound of the constraint. Can be
            updated later by assigning to the :attr:`ub`
            property on the constraint. Default is
            :const:`None`, which is equivalent to
            :const:`+inf`. This keyword should not be used
            in combination with the :attr:`expr` keyword.
        rhs: Sets the right-hand side of the constraint. Can
            be updated later by assigning to the :attr:`rhs`
            property on the constraint. The default value of
            :const:`None` implies that this keyword is
            ignored. Otherwise, use of this keyword implies
            that the :attr:`equality` property is set to
            :const:`True`. This keyword should not be used
            in combination with the :attr:`expr` keyword.

    Examples:
        >>> import pyomo.kernel as pmo
        >>> # A decision variable used to define constraints
        >>> x = pmo.variable()
        >>> # An upper bound constraint
        >>> c = pmo.constraint(0.5*x <= 1)
        >>> # (equivalent form)
        >>> c = pmo.constraint(body=0.5*x, ub=1)
        >>> # A range constraint
        >>> c = pmo.constraint(lb=-1, body=0.5*x, ub=1)
        >>> # An nonlinear equality constraint
        >>> c = pmo.constraint(x**2 == 1)
        >>> # (equivalent form)
        >>> c = pmo.constraint(body=x**2, rhs=1)
    """
    _ctype = IConstraint
    _linear_canonical_form = False
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_body",
                 "_lb",
                 "_ub",
                 "_equality",
                 "__weakref__")

    def __init__(self,
                 expr=None,
                 body=None,
                 lb=None,
                 ub=None,
                 rhs=None):
        self._parent = None
        self._storage_key = None
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
        """The body of the constraint"""
        return self._body
    @body.setter
    def body(self, body):
        if body is not None:
            body = as_numeric(body)
        self._body = body

    #
    # Extend the IConstraint interface to allow the
    # expression on this constraint to be changed
    # after construction.
    #

    @property
    def expr(self):
        """Get or set the expression on this constraint."""
        return super(constraint,self).expr
    @expr.setter
    def expr(self, expr):

        self._equality = False
        if expr is None:
            self.body = None
            self.lb = None
            self.ub = None
            return

        _expr_type = expr.__class__
        if _expr_type is tuple:
            #
            # Form equality expression
            #
            if len(expr) == 2:
                arg0 = expr[0]
                arg1 = expr[1]
                # assigning to the rhs property
                # will set the equality flag to True
                if not is_potentially_variable(arg1):
                    self.rhs = arg1
                    self.body = arg0
                elif not is_potentially_variable(arg0):
                    self.rhs = arg0
                    self.body = arg1
                else:
                    self.rhs = ZeroConstant
                    self.body = arg0
                    self.body -= arg1

            #
            # Form inequality expression
            #
            elif len(expr) == 3:
                arg0 = expr[0]
                if arg0 is not None:
                    if not is_numeric_data(arg0):
                        raise ValueError(
                            "Constraint '%s' found a 3-tuple (lower,"
                            " expression, upper) but the lower "
                            "value was not numeric data or an "
                            "expression restricted to storage of "
                            "numeric data."
                            % (self.name))

                arg1 = expr[1]
                if arg1 is not None:
                    arg1 = as_numeric(arg1)

                arg2 = expr[2]
                if arg2 is not None:
                    if not is_numeric_data(arg2):
                        raise ValueError(
                            "Constraint '%s' found a 3-tuple (lower,"
                            " expression, upper) but the upper "
                            "value was not numeric data or an "
                            "expression restricted to storage of "
                            "numeric data."
                            % (self.name))

                self.lb = arg0
                self.body  = arg1
                self.ub = arg2
            else:
                raise ValueError(
                    "Constraint '%s' assigned a tuple "
                    "of length %d. Expecting a tuple of "
                    "length 2 or 3:\n"
                    "Equality:   (body, rhs)\n"
                    "Inequality: (lb, body, ub)"
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
                        "\n   sum_product(model.costs) == model.income"
                        "\n   (0, model.price[item], 50)"
                        % (self.name, str(expr)))
            except AttributeError:
                msg = ("Constraint '%s' does not have a proper "
                       "value. Found '%s'\nExpecting a tuple or "
                       "equation. Examples:"
                       "\n   sum_product(model.costs) == model.income"
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
        if logical_expr._using_chained_inequality and \
           (logical_expr._chainedInequality.prev is not None):
            raise TypeError(logical_expr._chainedInequality.error_message())
        #
        # Process relational expressions
        # (i.e. explicit '==', '<', and '<=')
        #
        if relational_expr:
            if _expr_type is logical_expr.EqualityExpression:
                # assigning to the rhs property
                # will set the equality flag to True
                if not is_potentially_variable(expr.arg(1)):
                    self.rhs = expr.arg(1)
                    self.body = expr.arg(0)
                elif not is_potentially_variable(expr.arg(0)):
                    self.rhs = expr.arg(0)
                    self.body = expr.arg(1)
                else:
                    self.rhs = ZeroConstant
                    self.body = expr.arg(0)
                    self.body -= expr.arg(1)

            elif _expr_type is logical_expr.InequalityExpression:
                if expr._strict:
                    raise ValueError(
                        "Constraint '%s' encountered a strict "
                        "inequality expression ('>' or '<'). All"
                        " constraints must be formulated using "
                        "using '<=', '>=', or '=='."
                        % (self.name))
                if not is_potentially_variable(expr.arg(1)):
                    self.lb = None
                    self.body = expr.arg(0)
                    self.ub = expr.arg(1)
                elif not is_potentially_variable(expr.arg(0)):
                    self.lb = expr.arg(0)
                    self.body = expr.arg(1)
                    self.ub = None
                else:
                    self.lb = None
                    self.body  = expr.arg(0)
                    self.body -= expr.arg(1)
                    self.ub = ZeroConstant

            else:   # RangedExpression
                if any(expr._strict):
                    raise ValueError(
                        "Constraint '%s' encountered a strict "
                        "inequality expression ('>' or '<'). All"
                        " constraints must be formulated using "
                        "using '<=', '>=', or '=='."
                        % (self.name))

                if not is_numeric_data(expr.arg(0)):
                    raise ValueError(
                        "Constraint '%s' found a double-sided "
                        "inequality expression (lower <= "
                        "expression <= upper) but the lower "
                        "bound was not numeric data or an "
                        "expression restricted to storage of "
                        "numeric data."
                        % (self.name))
                if not is_numeric_data(expr.arg(2)):
                    raise ValueError(
                        "Constraint '%s' found a double-sided "\
                        "inequality expression (lower <= "
                        "expression <= upper) but the upper "
                        "bound was not numeric data or an "
                        "expression restricted to storage of "
                        "numeric data."
                        % (self.name))

                self.lb = expr.arg(0)
                self.body  = expr.arg(1)
                self.ub = expr.arg(2)

        #
        # Error check, to ensure that we don't have an equality
        # constraint with 'infinite' RHS
        #
        assert not (self.equality and (self.lb is None))
        assert (not self.equality) or (self.lb is self.ub)

#
# Note: This class is experimental. The implementation may
#       change or it may go away.
#
class linear_constraint(_MutableBoundsConstraintMixin,
                        IConstraint):
    """A linear constraint

    A linear constraint stores a linear relational
    expression defined by a list of variables and
    coefficients. This class can be used to reduce build
    time and memory for an optimization model. It also
    increases the speed at which the model can be output to
    a solver.

    Args:
        variables (list): Sets the list of variables in the
            linear expression defining the body of the
            constraint. Can be updated later by assigning to
            the :attr:`variables` property on the
            constraint.
        coefficients (list): Sets the list of coefficients
            for the variables in the linear expression
            defining the body of the constraint. Can be
            updated later by assigning to the
            :attr:`coefficients` property on the constraint.
        terms (list): An alternative way of initializing the
            :attr:`variables` and :attr:`coefficients` lists
            using an iterable of (variable, coefficient)
            tuples. Can be updated later by assigning to the
            :attr:`terms` property on the constraint. This
            keyword should not be used in combination with
            the :attr:`variables` or :attr:`coefficients`
            keywords.
        lb: Sets the lower bound of the constraint. Can be
            updated later by assigning to the :attr:`lb`
            property on the constraint. Default is
            :const:`None`, which is equivalent to
            :const:`-inf`.
        ub: Sets the upper bound of the constraint. Can be
            updated later by assigning to the :attr:`ub`
            property on the constraint. Default is
            :const:`None`, which is equivalent to
            :const:`+inf`.
        rhs: Sets the right-hand side of the constraint. Can
            be updated later by assigning to the :attr:`rhs`
            property on the constraint. The default value of
            :const:`None` implies that this keyword is
            ignored. Otherwise, use of this keyword implies
            that the :attr:`equality` property is set to
            :const:`True`.

    Examples:
        >>> import pyomo.kernel as pmo
        >>> # Decision variables used to define constraints
        >>> x = pmo.variable()
        >>> y = pmo.variable()
        >>> # An upper bound constraint
        >>> c = pmo.linear_constraint(variables=[x,y], coefficients=[1,2], ub=1)
        >>> # (equivalent form)
        >>> c = pmo.linear_constraint(terms=[(x,1), (y,2)], ub=1)
        >>> # (equivalent form using a general constraint)
        >>> c = pmo.constraint(x + 2*y <= 1)
    """
    _ctype = IConstraint
    _linear_canonical_form = True
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_variables",
                 "_coefficients",
                 "_lb",
                 "_ub",
                 "_equality",
                 "__weakref__")

    def __init__(self,
                 variables=None,
                 coefficients=None,
                 terms=None,
                 lb=None,
                 ub=None,
                 rhs=None):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._variables = None
        self._coefficients = None
        self._lb = None
        self._ub = None
        self._equality = False

        if terms is not None:
            if (variables is not None) or \
               (coefficients is not None):
                raise ValueError("Both the 'variables' and 'coefficients' "
                                 "keywords must be None when the 'terms' "
                                 "keyword is not None")
            # use the setter method
            self.terms = terms
        elif (variables is not None) or \
             (coefficients is not None):
            if (variables is None) or \
               (coefficients is None):
                raise ValueError("Both the 'variables' and 'coefficients' "
                                 "keywords must be set when the 'terms' "
                                 "keyword is None")
            self._variables = tuple(variables)
            self._coefficients = tuple(coefficients)
        else:
            # it is okay to initialize this with nothing
            self._variables = ()
            self._coefficients = ()

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
        """An iterator over the terms in the body of this
        constraint as (variable, coefficient) tuples"""
        return zip(self._variables, self._coefficients)
    @terms.setter
    def terms(self, terms):
        """Set the terms in the body of this constraint
        using an iterable of (variable, coefficient) tuples"""
        transpose = tuple(zip(*terms))
        if len(transpose) == 2:
            self._variables, self._coefficients = transpose
        else:
            assert transpose == ()
            self._variables = ()
            self._coefficients = ()

    #
    # Override a the default __call__ method on IConstraint
    # to avoid building the body expression
    #

    def __call__(self, exception=True):
        try:
            return sum(value(c, exception=exception) * \
                       v(exception=exception) for v,c in self.terms)
        except (ValueError, TypeError):
            if exception:
                raise ValueError("one or more terms "
                                 "could not be evaluated")
            return None

    #
    # Define the IConstraint abstract methods
    #

    @property
    def body(self):
        """The body of the constraint"""
        return sum(c * v for v, c in self.terms)

    #
    # Define methods that writers expect when the
    # _linear_canonical_form flag is True
    #

    def canonical_form(self, compute_values=True):
        """Build a canonical representation of the body of
        this constraints"""
        from pyomo.repn.standard_repn import \
            StandardRepn
        variables = []
        coefficients = []
        constant = 0
        for v, c in self.terms:
            if v.is_expression_type():
                v = v.expr
            if not v.fixed:
                variables.append(v)
                if compute_values:
                    coefficients.append(value(c))
                else:
                    coefficients.append(c)
            else:
                if compute_values:
                    constant += value(c) * v()
                else:
                    constant += c * v
        repn = StandardRepn()
        repn.linear_vars = tuple(variables)
        repn.linear_coefs = tuple(coefficients)
        repn.constant = constant
        return repn

# inserts class definitions for simple _tuple, _list, and
# _dict containers into this module
define_simple_containers(globals(),
                         "constraint",
                         IConstraint)
