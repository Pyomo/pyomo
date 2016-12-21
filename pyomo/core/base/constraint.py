#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['Constraint', '_ConstraintData', 'ConstraintList',
           'simple_constraint_rule', 'simple_constraintlist_rule']

import inspect
import sys
import logging
from weakref import ref as weakref_ref

import pyutilib.math
from pyomo.core.base import expr as EXPR
from pyomo.core.base.numvalue import (ZeroConstant,
                                      value,
                                      as_numeric,
                                      is_constant,
                                      _sub)
from pyomo.core.base.component import (ActiveComponentData,
                                       register_component)
from pyomo.core.base.indexed_component import \
    (ActiveIndexedComponent,
     UnindexedComponent_set)
from pyomo.core.base.misc import (apply_indexed_rule,
                                  tabular_writer)
from pyomo.core.base.sets import Set

from six import StringIO, iteritems

logger = logging.getLogger('pyomo.core')

_simple_constraint_rule_types = set([ type(None), bool ])

_rule_returned_none_error = """Constraint '%s': rule returned None.

Constraint rules must return either a valid expression, a 2- or 3-member
tuple, or one of Constraint.Skip, Constraint.Feasible, or
Constraint.Infeasible.  The most common cause of this error is
forgetting to include the "return" statement at the end of your rule.
"""

def simple_constraint_rule( fn ):
    """
    This is a decorator that translates None/True/False return
    values into Constraint.Skip/Constraint.Feasible/Constraint.Infeasible.
    This supports a simpler syntax in constraint rules, though these
    can be more difficult to debug when errors occur.

    Example use:

    @simple_constraint_rule
    def C_rule(model, i, j):
        ...

    model.c = Constraint(rule=simple_constraint_rule(...))
    """
    def wrapper_function ( *args, **kwargs ):
        if fn.__class__ in _simple_constraint_rule_types:
            #
            # If the argument is a boolean or None, then this is a
            # trivial constraint expression.
            #
            value = fn
        else:
            #
            # Otherwise, the argument is a functor, so call it to
            # generate the constraint expression.
            #
            value = fn( *args, **kwargs )
        #
        # Map the value to a constant:
        #   None        Skip this constraint
        #   True        Feasible constraint
        #   False       Infeasible constraint
        #
        if value.__class__ in _simple_constraint_rule_types:
            if value is None:
                return Constraint.Skip
            elif value is True:
                return Constraint.Feasible
            elif value is False:
                return Constraint.Infeasible
        return value
    return wrapper_function


def simple_constraintlist_rule( fn ):
    """
    This is a decorator that translates None/True/False return values
    into ConstraintList.End/Constraint.Feasible/Constraint.Infeasible.
    This supports a simpler syntax in constraint rules, though these can be
    more difficult to debug when errors occur.

    Example use:

    @simple_constraintlist_rule
    def C_rule(model, i, j):
        ...

    model.c = ConstraintList(expr=simple_constraintlist_rule(...))
    """
    def wrapper_function ( *args, **kwargs ):
        if fn.__class__ in _simple_constraint_rule_types:
            #
            # If the argument is a boolean or None, then this is a
            # trivial constraint expression.
            #
            value = fn
        else:
            #
            # Otherwise, the argument is a functor, so call it to
            # generate the constraint expression.
            #
            value = fn( *args, **kwargs )
        #
        # Map the value to a constant:
        #   None        End the constraint list
        #   True        Feasible constraint
        #   False       Infeasible constraint
        #
        if value.__class__ in _simple_constraint_rule_types:
            if value is None:
                return ConstraintList.End
            elif value is True:
                return Constraint.Feasible
            elif value is False:
                return Constraint.Infeasible
        return value
    return wrapper_function


#
# This class is a pure interface
#

class _ConstraintData(ActiveComponentData):
    """
    This class defines the data for a single constraint.

    Constructor arguments:
        component       The Constraint object that owns this data.

    Public class attributes:
        active          A boolean that is true if this constraint is
                            active in the model.
        body            The Pyomo expression for this constraint
        lower           The Pyomo expression for the lower bound
        upper           The Pyomo expression for the upper bound
        equality        A boolean that indicates whether this is an
                            equality constraint
        strict_lower    A boolean that indicates whether this
                            constraint uses a strict lower bound
        strict_upper    A boolean that indicates whether this
                            constraint uses a strict upper bound

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ()

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True

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
        Returns the value of f(x)-L for constraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        if self.lower is None:
            return float('-inf')
        else:
            return value(self.body)-value(self.lower)

    def uslack(self):
        """
        Returns the value of U-f(x) for constraints of the form:
            (L <=) f(x) <= U
            U >= f(x) (>= L)
        """
        if self.upper is None:
            return float('inf')
        else:
            return value(self.upper)-value(self.body)

    def slack(self):
        """
        Returns the smaller of lslack and uslack values
        """
        if self.lower is None:
            return value(self.upper)-value(self.body)
        elif self.upper is None:
            return value(self.body)-value(self.lower)
        return min(value(self.upper)-value(self.body),
                   value(self.body)-value(self.lower))

    #
    # Abstract Interface
    #

    @property
    def body(self):
        """Access the body of a constraint expression."""
        raise NotImplementedError

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        raise NotImplementedError

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        raise NotImplementedError

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        raise NotImplementedError

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        raise NotImplementedError

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        raise NotImplementedError

    def set_value(self, expr):
        """Set the expression on this constraint."""
        raise NotImplementedError

    def get_value(self):
        """Get the expression on this constraint."""
        raise NotImplementedError

class _GeneralConstraintData(_ConstraintData):
    """
    This class defines the data for a single general constraint.

    Constructor arguments:
        component       The Constraint object that owns this data.
        expr            The Pyomo expression stored in this constraint.

    Public class attributes:
        active          A boolean that is true if this constraint is
                            active in the model.
        body            The Pyomo expression for this constraint
        lower           The Pyomo expression for the lower bound
        upper           The Pyomo expression for the upper bound
        equality        A boolean that indicates whether this is an
                            equality constraint
        strict_lower    A boolean that indicates whether this
                            constraint uses a strict lower bound
        strict_upper    A boolean that indicates whether this
                            constraint uses a strict upper bound

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ('_body', '_lower', '_upper', '_equality')

    def __init__(self,  expr, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True

        self._body = None
        self._lower = None
        self._upper = None
        self._equality = False
        if expr is not None:
            self.set_value(expr)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_GeneralConstraintData, self).__getstate__()
        for i in _GeneralConstraintData.__slots__:
            result[i] = getattr(self, i)
        return result

    # Since this class requires no special processing of the state
    # dictionary, it does not need to implement __setstate__()

    #
    # Abstract Interface
    #

    @property
    def body(self):
        """Access the body of a constraint expression."""
        return self._body

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        return self._lower

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        return self._upper

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        return self._equality

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        return False

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        return False

    @property
    def expr(self):
        """Return the expression associated with this constraint."""
        return self.get_value()

    def set_value(self, expr):
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
                if arg1 is None or (not arg1._potentially_variable()):
                    self._lower = self._upper = arg1
                    self._body = arg0
                elif arg0 is None or (not arg0._potentially_variable()):
                    self._lower = self._upper = arg0
                    self._body = arg1
                else:
                    self._lower = self._upper = ZeroConstant
                    self._body = EXPR.generate_expression_bypassCloneCheck(
                        _sub, arg0, arg1)
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

                self._lower = arg0
                self._body  = arg1
                self._upper = arg2
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
                _args = expr._args
                # Explicitly dereference the original arglist (otherwise
                # this runs afoul of the getrefcount logic)
                expr._args = []

                if not _args[1]._potentially_variable():
                    self._lower = self._upper = _args[1]
                    self._body = _args[0]
                elif not _args[0]._potentially_variable():
                    self._lower = self._upper = _args[0]
                    self._body = _args[1]
                else:
                    self._lower = self._upper = ZeroConstant
                    self._body = EXPR.generate_expression_bypassCloneCheck(
                        _sub, _args[0], _args[1] )
            else:
                # Inequality expression: 2 or 3 arguments
                if expr._strict:
                    try:
                        _strict = any(expr._strict)
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

                _args = expr._args
                # Explicitly dereference the original arglist (otherwise
                # this runs afoul of the getrefcount logic)
                expr._args = []

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

                    self._lower = _args[0]
                    self._body  = _args[1]
                    self._upper = _args[2]

                else:
                    if not _args[1]._potentially_variable():
                        self._lower = None
                        self._body  = _args[0]
                        self._upper = _args[1]
                    elif not _args[0]._potentially_variable():
                        self._lower = _args[0]
                        self._body  = _args[1]
                        self._upper = None
                    else:
                        self._lower = None
                        self._body  = EXPR.generate_expression_bypassCloneCheck(
                            _sub, _args[0], _args[1])
                        self._upper = ZeroConstant

        #
        # Replace numeric bound values with a NumericConstant object,
        # and reset the values to 'None' if they are 'infinite'
        #
        if (self._lower is not None) and is_constant(self._lower):
            val = self._lower()
            if not pyutilib.math.is_finite(val):
                if val > 0:
                    raise ValueError(
                        "Constraint '%s' created with a +Inf lower "
                        "bound." % (self.name))
                self._lower = None
            elif bool(val > 0) == bool(val <= 0):
                raise ValueError(
                    "Constraint '%s' created with a non-numeric "
                    "lower bound." % (self.name))

        if (self._upper is not None) and is_constant(self._upper):
            val = self._upper()
            if not pyutilib.math.is_finite(val):
                if val < 0:
                    raise ValueError(
                        "Constraint '%s' created with a -Inf upper "
                        "bound." % (self.name))
                self._upper = None
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
            if self._lower is None:
                raise ValueError(
                    "Equality constraint '%s' defined with "
                    "non-finite term." % (self.name))
            assert self._lower is self._upper

    def get_value(self):
        """Get the expression on this constraint."""
        if self._equality:
            return self._body == self._lower
        else:
            if self._lower is None:
                return self._body <= self._upper
            elif self._upper is None:
                return self._lower <= self._body
            return self._lower <= self._body <= self._upper

class Constraint(ActiveIndexedComponent):
    """
    This modeling component defines a constraint expression using a
    rule function.

    Constructor arguments:
        expr            A Pyomo expression for this constraint
        rule            A function that is used to construct constraint
                            expressions
        doc             A text string describing this component
        name            A name for this component

    Public class attributes:
        doc             A text string describing this component
        name            A name for this component
        active          A boolean that is true if this component will be
                            used to construct a model instance
        rule            The rule used to initialize the constraint(s)

    Private class attributes:
        _constructed        A boolean that is true if this component has been
                                constructed
        _data               A dictionary from the index set to component data
                                objects
        _index              The set of valid indices
        _implicit_subsets   A tuple of set objects that represents the index set
        _model              A weakref to the model that owns this component
        _parent             A weakref to the parent block that owns this component
        _type               The class type for the derived subclass
    """

    NoConstraint    = (1000,)
    Skip            = (1000,)
    Infeasible      = (1001,)
    Violated        = (1001,)
    Feasible        = (1002,)
    Satisfied       = (1002,)

    def __new__(cls, *args, **kwds):
        if cls != Constraint:
            return super(Constraint, cls).__new__(cls)
        if args == () or (type(args[0]) == set and args[0] == UnindexedComponent_set and len(args)==1):
            return SimpleConstraint.__new__(SimpleConstraint)
        else:
            return IndexedConstraint.__new__(IndexedConstraint)

    def __init__(self, *args, **kwargs):
        self.rule = kwargs.pop('rule', None)
        self._init_expr = kwargs.pop('expr', None)
        #if self.rule is None and self._init_expr is None:
        #    raise ValueError("A simple Constraint component requires a 'rule' or 'expr' option")
        kwargs.setdefault('ctype', Constraint)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)


    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing constraint %s"
                         % (self.name))
        if self._constructed:
            return
        self._constructed=True

        _init_expr = self._init_expr
        _init_rule = self.rule
        #
        # We no longer need these
        #
        self._init_expr = None
        # Utilities like DAE assume this stays around
        #self.rule = None

        if (_init_rule is None) and \
           (_init_expr is None):
            # No construction role or expression specified.
            return

        _self_parent = self._parent()
        if not self.is_indexed():
            #
            # Scalar component
            #
            if _init_rule is None:
                tmp = _init_expr
            else:
                try:
                    tmp = _init_rule(_self_parent)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "constraint %s:\n%s: %s"
                        % (self.name,
                           type(err).__name__,
                           err))
                    raise
                if tmp is None:
                    raise ValueError(
                        _rule_returned_none_error % (self.name,) )

            assert None not in self._data
            cdata = self._check_skip_add(None, tmp, condata=self)
            if cdata is not None:
                # this happens as a side-effect of set_value on
                # SimpleConstraint (normally _check_skip_add does not
                # add anything to the _data dict but it does call
                # set_value on the condata object we pass in)
                assert None in self._data
            else:
                assert None not in self._data

        else:

            if not _init_expr is None:
                raise IndexError(
                    "Constraint '%s': Cannot initialize multiple indices "
                    "of a constraint with a single expression" %
                    (self.name,) )

            for ndx in self._index:
                try:
                    tmp = apply_indexed_rule(self,
                                             _init_rule,
                                             _self_parent,
                                             ndx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "constraint %s with index %s:\n%s: %s"
                        % (self.name,
                           str(ndx),
                           type(err).__name__,
                           err))
                    raise
                if tmp is None:
                    raise ValueError(
                        _rule_returned_none_error %
                        ('%s[%s]' % (self.name, str(ndx)),) )

                cdata = self._check_skip_add(ndx, tmp)
                if cdata is not None:
                    self._data[ndx] = cdata

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Size", len(self)),
             ("Index", self._index \
              if self._index != UnindexedComponent_set else None),
             ("Active", self.active),
             ],
            iteritems(self),
            ( "Lower","Body","Upper","Active" ),
            lambda k, v: [ "-Inf" if v.lower is None else v.lower,
                           v.body,
                           "+Inf" if v.upper is None else v.upper,
                           v.active,
                           ]
            )

    def display(self, prefix="", ostream=None):
        """
        Print component state information

        This duplicates logic in Component.pprint()
        """
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab="    "
        ostream.write(prefix+self.local_name+" : ")
        ostream.write("Size="+str(len(self)))

        ostream.write("\n")
        tabular_writer( ostream, prefix+tab,
                        ((k,v) for k,v in iteritems(self._data) if v.active),
                        ( "Lower","Body","Upper" ),
                        lambda k, v: [ value(v.lower),
                                       v.body(),
                                       value(v.upper),
                                       ] )

    #
    # Checks flags like Constraint.Skip, etc. before
    # actually creating a constraint object. Optionally
    # pass in the _ConstraintData object to set the value
    # on. Only returns the _ConstraintData object when it
    # should be added to the _data dict; otherwise, None
    # is returned or an exception is raised.
    #
    def _check_skip_add(self, index, expr, condata=None):

        #
        # Adds a dummy constraint object to the _data
        # dict just before an error message is generated
        # so that we can generate a fully qualified name
        #
        def _prep_for_error():
            if condata is None:
                self._data[index] = _GeneralConstraintData(None,
                                                           component=self)
            else:
                self._data[index] = condata

        _expr_type = expr.__class__
        #
        # Convert deprecated expression values
        #
        if _expr_type in _simple_constraint_rule_types:

            _prep_for_error()

            if expr is None:
                raise ValueError(
                    "Invalid constraint expression. The constraint "
                    "expression resolved to None instead of a Pyomo "
                    "object. Please modify your rule to return "
                    "Constraint.Skip instead of None."
                    "\n\nError thrown for Constraint '%s'"
                    % (self._data[index].name))

            #
            # There are cases where a user thinks they are generating
            # a valid 2-sided inequality, but Python's internal
            # systems for handling chained inequalities is doing
            # something very different and resolving it to True /
            # False.  In this case, chainedInequality will be
            # non-None, but the expression will be a bool.  For
            # example, model.a < 1 > 0.
            #
            if EXPR.generate_relational_expression.\
                    chainedInequality is not None:

                buf = StringIO()
                EXPR.generate_relational_expression.\
                    chainedInequality.pprint(buf)
                #
                # We are about to raise an exception, so it's OK to
                # reset chainedInequality
                #
                EXPR.generate_relational_expression.\
                    chainedInequality = None
                raise ValueError(
                    "Invalid chained (2-sided) inequality detected. "
                    "The expression is resolving to %s instead of a "
                    "Pyomo Expression object. This can occur when "
                    "the middle term of a chained inequality is a "
                    "constant or immutable parameter, for example, "
                    "'model.a <= 1 >= 0'.  The proper form for "
                    "2-sided inequalities is '0 <= model.a <= 1'."
                    "\n\nError thrown for Constraint '%s'"
                    "\n\nUnresolved (dangling) inequality "
                    "expression: %s"
                    % (expr, self._data[index].name, buf))
            else:
                raise ValueError(
                    "Invalid constraint expression. The constraint "
                    "expression resolved to a trivial Boolean (%s) "
                    "instead of a Pyomo object. Please modify your "
                    "rule to return Constraint.%s instead of %s."
                    "\n\nError thrown for Constraint '%s'"
                    % (expr,
                       expr and "Feasible" or "Infeasible",
                       expr,
                       self._data[index].name))

        #
        # Ignore an 'empty' constraint
        #
        if _expr_type is tuple and len(expr) == 1:
            if (expr == Constraint.Skip) or \
               (expr == Constraint.Feasible):
                return None
            if expr == Constraint.Infeasible:
                _prep_for_error()
                raise ValueError(
                    "Constraint '%s' is always infeasible"
                    % self._data[index].name)

        if condata is None:
            self._data[index] = condata = \
                _GeneralConstraintData(None, component=self)
        condata.set_value(expr)
        assert condata.parent_component() is self

        return condata

class SimpleConstraint(_GeneralConstraintData, Constraint):
    """
    SimpleConstraint is the implementation representing a single,
    non-indexed constraint.
    """

    def __init__(self, *args, **kwds):
        _GeneralConstraintData.__init__(self,
                                        component=self,
                                        expr=None)
        Constraint.__init__(self, *args, **kwds)

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
    def body(self):
        """Access the body of a constraint expression."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the body of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.body.fget(self)
        raise ValueError(
            "Accessing the body of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the lower bound of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.lower.fget(self)
        raise ValueError(
            "Accessing the lower bound of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the upper bound of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.upper.fget(self)
        raise ValueError(
            "Accessing the upper bound of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the equality flag of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.equality.fget(self)
        raise ValueError(
            "Accessing the equality flag of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the strict_lower flag of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.strict_lower.fget(self)
        raise ValueError(
            "Accessing the strict_lower flag of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the strict_upper flag of SimpleConstraint "
                    "'%s' before the Constraint has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralConstraintData.strict_upper.fget(self)
        raise ValueError(
            "Accessing the strict_upper flag of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    #
    # Singleton constraints are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Constraint.Skip are managed. But after that they will behave
    # like _ConstraintData objects where set_value does not handle
    # Constraint.Skip but expects a valid expression or None.
    #

    def set_value(self, expr):
        """Set the expression on this constraint."""
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return _GeneralConstraintData.set_value(self, expr)
        raise ValueError(
            "Setting the value of constraint '%s' "
            "before the Constraint has been constructed (there "
            "is currently no object to set)."
            % (self.name))

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add a constraint with a given index."""
        if index is not None:
            raise ValueError(
                "SimpleConstraint object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.name, index))
        self.set_value(expr)
        return self

class IndexedConstraint(Constraint):

    #
    # Leaving this method for backward compatibility reasons
    # Note: It allows adding members outside of self._index.
    #       This has always been the case. Not sure there is
    #       any reason to maintain a reference to a separate
    #       index set if we allow this.
    #
    def add(self, index, expr):
        """Add a constraint with a given index."""
        cdata = self._check_skip_add(index, expr)
        if cdata is not None:
            self._data[index] = cdata
        return cdata

    # This should be supported by all indexed components
    def __delitem__(self, index):
        del self._data[index]

class ConstraintList(IndexedConstraint):
    """
    A constraint component that represents a list of constraints.
    Constraints can be indexed by their index, but when they are
    added an index value is not specified.
    """

    End             = (1003,)

    def __init__(self, **kwargs):
        """Constructor"""
        args = (Set(),)
        self._nconstraints = 0
        if 'expr' in kwargs:
            raise ValueError(
                "ConstraintList does not accept the 'expr' keyword")
        Constraint.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        generate_debug_messages = \
            __debug__ and logger.isEnabledFor(logging.DEBUG)
        if generate_debug_messages:
            logger.debug("Constructing constraint list %s"
                         % (self.name))

        if self._constructed:
            return
        self._constructed=True

        assert self._init_expr is None
        _init_rule = self.rule

        #
        # We no longer need these
        #
        self._init_expr = None
        # Utilities like DAE assume this stays around
        #self.rule = None

        if _init_rule is None:
            return

        _generator = None
        _self_parent = self._parent()
        if inspect.isgeneratorfunction(_init_rule):
            _generator = _init_rule(_self_parent)
        elif inspect.isgenerator(_init_rule):
            _generator = _init_rule
        if _generator is None:
            while True:
                val = self._nconstraints + 1
                if generate_debug_messages:
                    logger.debug(
                        "   Constructing constraint index "+str(val))
                expr = apply_indexed_rule(self,
                                          _init_rule,
                                          _self_parent,
                                          val)
                if expr is None:
                    raise ValueError(
                        "ConstraintList '%s': rule returned None "
                        "instead of ConstraintList.End" % (self.name,) )
                if (expr.__class__ is tuple) and \
                   (expr == ConstraintList.End):
                    return
                self.add(expr)

        else:

            for expr in _generator:
                if expr is None:
                    raise ValueError(
                        "ConstraintList '%s': generator returned None "
                        "instead of ConstraintList.End" % (self.name,) )
                if (expr.__class__ is tuple) and \
                   (expr == ConstraintList.End):
                    return
                self.add(expr)

    def add(self, expr):
        """Add a constraint with an implicit index."""
        cdata = self._check_skip_add(self._nconstraints + 1, expr)
        self._nconstraints += 1
        self._index.add(self._nconstraints)
        if cdata is not None:
            self._data[self._nconstraints] = cdata
        return cdata

register_component(Constraint, "General constraint expressions.")
register_component(ConstraintList, "A list of constraint expressions.")
