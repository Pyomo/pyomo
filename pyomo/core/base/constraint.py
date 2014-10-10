#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['Constraint', '_ConstraintData', 'ConstraintList', 
           'simple_constraint_rule', 'simple_constraintlist_rule']

import inspect
import sys
import logging
from weakref import ref as weakref_ref
from inspect import isgenerator
from six import StringIO, iteritems

import pyutilib.math
import pyutilib.misc

from pyomo.core.base.expr import _ExpressionBase, _EqualityExpression, \
    generate_relational_expression, generate_expression_bypassCloneCheck
from pyomo.core.base.numvalue import ZeroConstant, value, as_numeric, _sub
from pyomo.core.base.component import ActiveComponentData, register_component
from pyomo.core.base.sparse_indexed_component import ActiveSparseIndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule, tabular_writer
from pyomo.core.base.sets import Set

logger = logging.getLogger('pyomo.core')

_simple_constraint_rule_types = set([ type(None), bool ])


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
            # Otherwise, the argument is a functor, so call it to generate the 
            # constraint expression.
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
            # Otherwise, the argument is a functor, so call it to generate the 
            # constraint expression.
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

    Private class attributes:
        _component      The constraint component.
        _equality       A boolean that indicates whether this is an 
                            equality constraint
    """

    __pickle_slots__ = ( 'body', 'lower', 'upper', '_equality')
    __slots__ = __pickle_slots__ + ( '__weakref__', )

    def __init__(self, owner):
        # the following lines represent in-lining of the ActiveComponentData
        # and ComponentData constructors.
        self._component = weakref_ref(owner)
        self._active = True
        #
        self.body = None
        self.lower = None
        self.upper = None
        self._equality = False

    #
    # 'equality' is a property because we want to limit how it is set.
    # 
    @property
    def equality(self):
        """
        Return a flag indicating whether this is an equality constraint.
        """
        return self._equality

    @equality.setter
    def equality(self, val):
        """
        Set the equality attribute to the given value.
        """
        raise AttributeError("Assignment not allowed. Set using the _equality attribute.")

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_ConstraintData, self).__getstate__()
        for i in _ConstraintData.__pickle_slots__:
            result[i] = getattr(self, i)
        return result

    # Since this class requires no special processing of the state
    # dictionary, it does not need to implement __setstate__()

    def __call__(self, exception=True):
        """
        Compute the value of the body of this constraint.

	    This method does not simply return self.value because
	    that data value may be out of date w.r.t. the value of
	    decision variables.
        """

        if self.body is None:
            return None
        return self.body(exception=exception)

    def lslack(self):
        """
        Returns the value of L-f(x) for constraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        if self.lower is None:
            return float('-inf')
        else:
            return value(self.lower)-value(self.body)

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


class Constraint(ActiveSparseIndexedComponent):
    """
    This modeling component defines a constraint expression using a
    rule function.

    Constructor arguments:
        expr            A Pyomo expression for this constraint
        noruleinit      Indicate that its OK that no initialization is
                            specified
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
        trivial         This boolean is True if all constraint indices have 
                            trivial expressions

    Private class attributes:
        _constructed        A boolean that is true if this component has been
                                constructed
        _data               A dictionary from the index set to component data 
                                objects
        _index              The set of valid indices
        _implicit_subsets   A tuple of set objects that represents the index set
        _model              A weakref to the model that owns this component
        _no_rule_init       A boolean that indicates if an initialization rule 
                                is needed
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
        if args == ():
            return SimpleConstraint.__new__(SimpleConstraint)
        else:
            return IndexedConstraint.__new__(IndexedConstraint)

    def __init__(self, *args, **kwargs):
        self.rule = kwargs.pop('rule', None )
        self._expr = kwargs.pop('expr', None )
        self._no_rule_init = kwargs.pop('noruleinit', False )
        #
        # This value is set when creating a canonical representation.
        #
        self.trivial = False
        #
        kwargs.setdefault('ctype', Constraint)
        ActiveSparseIndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing constraint %s",self.cname(True))
        if self._constructed:
            return
        _self_rule = self.rule
        if self._no_rule_init and (_self_rule is not None):
            logger.warning("The noruleinit keyword is being used in conjunction " \
                  "with the rule keyword for constraint '%s'; defaulting to " \
                  "rule-based construction", self.cname(True))
        if _self_rule is None and self._expr is None:
            if not self._no_rule_init:
                logger.warning("No construction rule or expression specified for "
                            "constraint '%s'", self.cname(True))
            else:
                self._constructed=True
            return
        self._constructed=True
        #
        _self_parent = self._parent()
        if not self.is_indexed():
            #
            # Scalar component
            #
            if _self_rule is None:
                self.add(None, self._expr)
            else:
                try:
                    tmp = _self_rule(_self_parent)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "constraint %s:\n%s: %s"
                        % ( self.cname(True), type(err).__name__, err ) )
                    raise
                if tmp is None:
                    raise ValueError("Constraint rule returned None instead of Constraint.Skip")
                self.add(None, tmp)
        else:
            if not self._expr is None:
                raise IndexError("Cannot initialize multiple indices of a constraint with a single expression")
            for val in self._index:
                try:
                    tmp = apply_indexed_rule(self, _self_rule, _self_parent, val)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for "
                        "constraint %s with index %s:\n%s: %s"
                        % ( self.cname(True), str(val), type(err).__name__, err ) )
                    raise
                if tmp is None:
                    raise ValueError("Constraint rule returned None instead of Constraint.Skip for index %s" % str(val))
                self.add(val, tmp)

    def add(self, index, expr):
        """
        Add a constraint with a specified index.
        """
        _expr_type = expr.__class__
        #
        # Convert deprecated expression values
        #
        if _expr_type in _simple_constraint_rule_types:
            if expr is None:
                raise ValueError("""
Invalid constraint expression.  The constraint expression resolved to
None instead of a Pyomo object.  Please modify your rule to return
Constraint.Skip instead of None.

Error thrown for Constraint "%s"
""" % ( self.cname(True), ) )
            #
            # There are cases where a user thinks they are generating a
            # valid 2-sided inequality, but Python's internal systems
            # for handling chained inequalities is doing something very
            # different and resolving it to True/False.  In this case,
            # chainedInequality will be non-None, but the expression
            # will be a bool.  For example, model.a < 1 > 0.
            #
            if generate_relational_expression.chainedInequality is not None:
                buf = StringIO.StringIO()
                generate_relational_expression.chainedInequality.pprint(buf)
                #
                # We are about to raise an exception, so it's OK to 
                # reset chainedInequality
                #
                generate_relational_expression.chainedInequality = None
                raise ValueError("""
Invalid chained (2-sided) inequality detected.  The expression is
resolving to %s instead of a Pyomo Expression object.  This can occur
when the middle term of a chained inequality is a constant or immutable
parameter, for example, "model.a <= 1 >= 0".  The proper form for
2-sided inequalities is "0 <= model.a <= 1".

Error thrown for Constraint "%s"

Unresolved (dangling) inequality expression:
    %s
""" % ( expr, self.cname(True), buf ) )
            else:
                raise ValueError("""
Invalid constraint expression.  The constraint expression resolved to a
trivial Boolean (%s) instead of a Pyomo object.  Please modify your rule
to return Constraint.%s instead of %s.

Error thrown for Constraint "%s"
""" % ( expr, expr and "Feasible" or "Infeasible", expr, self.cname(True) ) )
        #
        # Ignore an 'empty' constraint
        #
        if _expr_type is tuple and len(expr) == 1:
            if expr == Constraint.Skip or expr == Constraint.Feasible:
                return
            if expr == Constraint.Infeasible:
                raise ValueError( "Constraint '%s' is always infeasible" % 
                                  self.cname(True) )
        #
        # Local variables to optimize runtime performance
        #
        if index is None:
            conData = self
        else:
            conData = _ConstraintData(self)
        #
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

                conData._equality = True
                if arg1 is None or arg1.is_fixed():
                    conData.lower = conData.upper = arg1
                    conData.body = arg0
                elif arg0 is None or arg0.is_fixed():
                    conData.lower = conData.upper = arg0
                    conData.body = arg1
                else:
                    conData.lower = conData.upper = ZeroConstant
                    conData.body = arg0 - arg1
            #
            # Form inequality expression
            #
            elif len(expr) == 3:
                arg0 = expr[0]
                if arg0 is not None:
                    arg0 = as_numeric(arg0)
                    if not arg0.is_fixed():
                        raise ValueError(
                            "Constraint '%s' found a 3-tuple (lower, " 
                            "expression, upper) but the lower value was "
                            "non-constant." % self.cname(True) )

                arg1 = expr[1]
                if arg1 is not None:
                    arg1 = as_numeric(arg1)

                arg2 = expr[2]
                if arg2 is not None:
                    arg2 = as_numeric(arg2)
                    if not arg2.is_fixed():
                        raise ValueError(
                            "Constraint '%s' found a 3-tuple (lower, " 
                            "expression, upper) but the upper value was "
                            "non-constant" % self.cname(True) )

                conData.lower = arg0
                conData.body  = arg1
                conData.upper = arg2
            else:
                raise ValueError(
                    "Constructor rule for constraint '%s' returned a tuple" 
                    ' of length %d.  Expecting a tuple of length 2 or 3:\n' 
                    'Equality:   (left, right)\n' 
                    'Inequality: (lower, expression, upper)'
                    % ( self.cname(True), len(expr) ))

            relational_expr = False
        else:
            try:
                relational_expr = expr.is_relational()
                if not relational_expr:
                    raise ValueError(
                        "Constraint '%s' does not have a proper value.  " 
                        "Found '%s'\nExpecting a tuple or equation.  "
                        "Examples:\n" 
                        "    summation( model.costs ) == model.income\n" 
                        "    (0, model.price[ item ], 50)"
                        % ( self.cname(True), str(expr) ))
            except AttributeError:
                msg = "Constraint '%s' does not have a proper value.  " \
                      "Found '%s'\nExpecting a tuple or equation.  " \
                      "Examples:\n" \
                      "    summation( model.costs ) == model.income\n" \
                      "    (0, model.price[ item ], 50)" \
                      % ( self.cname(True), str(expr) )
                if type(expr) is bool:
                    msg +="""
Note: constant Boolean expressions are not valid constraint expressions.
Some apparently non-constant compound inequalities (e.g. "expr >= 0 <= 1")
can return boolean values; the proper form for compound inequalities is
always "lb <= expr <= ub"."""
                raise ValueError(msg)
        #
        # Special check for chainedInequality errors like "if var < 1:"
        # within rules.  Catching them here allows us to provide the
        # user with better (and more immediate) debugging information.
        # We don't want to check earlier because we want to provide a
        # specific debugging message if the construction rule returned
        # True/False; for example, if the user did ( var < 1 > 0 )
        # (which also results in a non-None chainedInequality value)
        #
        if generate_relational_expression.chainedInequality is not None:
            from expr import chainedInequalityErrorMessage
            raise TypeError(chainedInequalityErrorMessage())
        #
        # Process relational expressions (i.e. explicit '==', '<', and '<=')
        #
        if relational_expr:
            if _expr_type is _EqualityExpression:
                # Equality expression: only 2 arguments!
                conData._equality = True
                if expr._args[1].is_fixed():
                    conData.lower = conData.upper = expr._args[1]
                    conData.body = expr._args[0]
                elif expr._args[0].is_fixed():
                    conData.lower = conData.upper = expr._args[0]
                    conData.body = expr._args[1]
                else:
                    conData.lower = conData.upper = ZeroConstant
                    conData.body = generate_expression_bypassCloneCheck(_sub, expr._args[0], expr._args[1])
            else:
                # Inequality expression: 2 or 3 arguments
                if True in expr._strict:
                    #
                    # We can relax this when:
                    #   (a) we have a need for this
                    #   (b) we have problem writer that explicitly
                    #       handles this
                    #   (c) we make sure that all problem writers
                    #       that don't handle this make it known to
                    #       the user through an error or warning
                    #
                    raise ValueError(
                        "Constraint '%s' encountered a strict inequality "
                        "expression ('>' or '<'). All constraints must be "
                        "formulated using using '<=', '>=', or '=='." 
                        % (self.cname(True),) )
                if len(expr._args) == 3:
                    if not expr._args[0].is_fixed():
                        msg = "Constraint '%s' found a double-sided "\
                              "inequality expression (lower <= expression "\
                              "<= upper) but the lower bound was non-constant"
                        raise ValueError(msg % (self.cname(True),))
                    if not expr._args[2].is_fixed():
                        msg = "Constraint '%s' found a double-sided "\
                              "inequality expression (lower <= expression "\
                              "<= upper) but the upper bound was non-constant"
                        raise ValueError(msg % (self.cname(True),))
                    conData.lower = expr._args[0]
                    conData.body  = expr._args[1]
                    conData.upper = expr._args[2]
                else:
                    if expr._args[1].is_fixed():
                        conData.lower = None
                        conData.body  = expr._args[0]
                        conData.upper = expr._args[1]
                    elif expr._args[0].is_fixed():
                        conData.lower = expr._args[0]
                        conData.body  = expr._args[1]
                        conData.upper = None
                    else:
                        conData.lower = None
                        conData.body  = generate_expression_bypassCloneCheck(_sub, expr._args[0], expr._args[1])
                        conData.upper = ZeroConstant
        #
        # Replace numeric bound values with a NumericConstant object,
        # and reset the values to 'None' if they are 'infinite'
        #
        if conData.lower is not None:
            val = conData.lower()
            if not pyutilib.math.is_finite(val):
                if val > 0:
                    msg = "Constraint '%s' created with a +Inf lower bound"
                    raise ValueError(msg % ( self.cname(True), ))
                conData.lower = None
            elif bool(val > 0) == bool(val <= 0):
                msg = "Constraint '%s' created with a non-numeric lower bound"
                raise ValueError(msg % ( self.cname(True), ))
        if conData.upper is not None:
            val = conData.upper()
            if not pyutilib.math.is_finite(val):
                if val < 0:
                    msg = "Constraint '%s' created with a -Inf upper bound"
                    raise ValueError(msg % ( self.cname(True), ))
                conData.upper = None
            elif bool(val > 0) == bool(val <= 0):
                msg = "Constraint '%s' created with a non-numeric upper bound"
                raise ValueError(msg % ( self.cname(True), ))
        #
        # Error check, to ensure that we don't have a constraint that
        # doesn't depend on any variables / parameters
        #
        # Error check, to ensure that we don't have an equality constraint with
        # 'infinite' RHS
        #
        if conData._equality:
            if conData.lower != conData.upper: #pragma:nocover
                msg = "Equality constraint '%s' has non-equal lower and "\
                      "upper bounds (this is indicitive of a SERIOUS "\
                      "internal error in Pyomo)."
                raise RuntimeError(msg % self.cname(True))
            if conData.lower is None:
                msg = "Equality constraint '%s' defined with non-finite term"
                raise ValueError(msg % self.cname(True))
        #
        # hook up the constraint data object to the parent constraint.
        #
        self._data[index] = conData

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
            iteritems(self._data),
            ( "Key","Lower","Body","Upper","Active" ),
            lambda k, v: [ k,
                           "-Inf" if v.lower is None else v.lower,
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
        ostream.write(prefix+self.cname()+" : ")
        ostream.write("Size="+str(len(self))) 

        ostream.write("\n")
        tabular_writer( ostream, prefix+tab,  
                        ((k,v) for k,v in iteritems(self._data) if v.active), 
                        ( "Key","Lower","Body","Upper" ), 
                        lambda k, v: [ k, 
                                       value(v.lower), 
                                       v.body(), 
                                       value(v.upper), 
                                       ] ) 


class SimpleConstraint(Constraint, _ConstraintData):
    """
    SimpleConstraint is the implementation representing a single,
    non-indexed constraint.
    """

    def __init__(self, *args, **kwd):
        _ConstraintData.__init__(self, self)
        Constraint.__init__(self, *args, **kwd)

    def __call__(self, exception=True):
        """Compute the value of the constraint body"""

        if self._constructed:
            return _ConstraintData.__call__(self, exception=exception)
        if exception:
            raise ValueError( """Evaluating the numeric value of constraint '%s' before the Constraint has been
            constructed (there is currently no value to return).""" % self.cname(True) )

    #
    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, we do not
    # need to define the __getstate__ or __setstate__ methods.
    # We just defer to the super() get/set state.  Since all of our 
    # get/set state methods rely on super() to traverse the MRO, this 
    # will automatically pick up both the Component and Data base classes.
    #


class IndexedConstraint(Constraint):

    def __call__(self, exception=True):
        """Compute the value of the constraint body"""
        if exception:
            msg = 'Cannot compute the value of an array of constraints'
            raise TypeError(msg)

class ConstraintList(IndexedConstraint):
    """
    A constraint component that represents a list of constraints.
    Constraints can be indexed by their index, but when they are added
    an index value is not specified.
    """

    End             = (1003,)

    def __init__(self, **kwargs):
        """Constructor"""
        args = (Set(),)
        self._nconstraints = 0
        Constraint.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        generate_debug_messages = __debug__ and logger.isEnabledFor(logging.DEBUG)
        if generate_debug_messages:
            logger.debug("Constructing constraint list %s", self.cname(True))
        if self._constructed:
            return
        _self_rule = self.rule
        if self._no_rule_init and (_self_rule is not None):
            logger.warning("The noruleinit keyword is being used in conjunction " \
                  "with the rule keyword for constraint '%s'; defaulting to " \
                  "rule-based construction" % self.cname(True))
        self._constructed=True
        if _self_rule is None:
            if not self._no_rule_init:
                logger.warn("No construction rule or expression specified for "
                            "constraint '%s'", self.cname(True))
            return
        #
        _generator = None
        _self_parent = self._parent()
        if inspect.isgeneratorfunction(_self_rule):
            _generator = _self_rule(_self_parent)
        elif inspect.isgenerator(_self_rule):
            _generator = _self_rule
        if _generator is None:
            while True:
                val = self._nconstraints + 1
                if generate_debug_messages:
                    logger.debug("   Constructing constraint index "+str(val))
                expr = apply_indexed_rule( self, _self_rule, _self_parent, val )
                if expr is None:
                    raise ValueError( "Constraint rule returned None "
                                      "instead of ConstraintList.End" )
                if (expr.__class__ is tuple and expr == ConstraintList.End):
                    return
                self.add(expr)
        else:
            for expr in _generator:
                if expr is None:
                    raise ValueError( "Constraint generator returned None "
                                      "instead of ConstraintList.End" )
                if (expr.__class__ is tuple and expr == ConstraintList.End):
                    return
                self.add(expr)

    def add(self, expr):
        """
        Add a constraint with an implicit index.
        """
        self._nconstraints += 1
        self._index.add(self._nconstraints)
        Constraint.add(self, self._nconstraints, expr)


register_component(Constraint, "General constraint expressions.")
register_component(ConstraintList, "A list of constraint expressions.")

