#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = [
    'Constraint',
    '_ConstraintData',
    'ConstraintList',
    'simple_constraint_rule',
    'simple_constraintlist_rule',
]

import io
import sys
import logging
import math
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload

from pyomo.common.deprecation import RenamedClass
from pyomo.common.errors import DeveloperError
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.timing import ConstructionTimer

from pyomo.core.expr.numvalue import (
    NumericValue,
    value,
    as_numeric,
    is_fixed,
    native_numeric_types,
    native_types,
)
from pyomo.core.expr import (
    ExpressionType,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent,
    UnindexedComponent_set,
    rule_wrapper,
)
from pyomo.core.base.set import Set
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
    Initializer,
    IndexedCallInitializer,
    CountedCallInitializer,
)


logger = logging.getLogger('pyomo.core')

_inf = float('inf')
_nonfinite_values = {_inf, -_inf}
_known_relational_expressions = {
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
}
_rule_returned_none_error = """Constraint '%s': rule returned None.

Constraint rules must return either a valid expression, a 2- or 3-member
tuple, or one of Constraint.Skip, Constraint.Feasible, or
Constraint.Infeasible.  The most common cause of this error is
forgetting to include the "return" statement at the end of your rule.
"""


def simple_constraint_rule(rule):
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
    return rule_wrapper(
        rule,
        {
            None: Constraint.Skip,
            True: Constraint.Feasible,
            False: Constraint.Infeasible,
        },
    )


def simple_constraintlist_rule(rule):
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
    return rule_wrapper(
        rule,
        {
            None: ConstraintList.End,
            True: Constraint.Feasible,
            False: Constraint.Infeasible,
        },
    )


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

    # Set to true when a constraint class stores its expression
    # in linear canonical form
    _linear_canonical_form = False

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET
        self._active = True

    #
    # Interface
    #

    def __call__(self, exception=True):
        """Compute the value of the body of this constraint."""
        return value(self.body, exception=exception)

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        return self.lb is not None

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        return self.ub is not None

    def lslack(self):
        """
        Returns the value of f(x)-L for constraints of the form:
            L <= f(x) (<= U)
            (U >=) f(x) >= L
        """
        lb = self.lb
        if lb is None:
            return _inf
        else:
            return value(self.body) - lb

    def uslack(self):
        """
        Returns the value of U-f(x) for constraints of the form:
            (L <=) f(x) <= U
            U >= f(x) (>= L)
        """
        ub = self.ub
        if ub is None:
            return _inf
        else:
            return ub - value(self.body)

    def slack(self):
        """
        Returns the smaller of lslack and uslack values
        """
        lb = self.lb
        ub = self.ub
        body = value(self.body)
        if lb is None:
            return ub - body
        elif ub is None:
            return body - lb
        return min(ub - body, body - lb)

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
    def lb(self):
        """Access the value of the lower bound of a constraint expression."""
        raise NotImplementedError

    @property
    def ub(self):
        """Access the value of the upper bound of a constraint expression."""
        raise NotImplementedError

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        raise NotImplementedError

    @property
    def strict_lower(self):
        """True if this constraint has a strict lower bound."""
        raise NotImplementedError

    @property
    def strict_upper(self):
        """True if this constraint has a strict upper bound."""
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

    __slots__ = ('_body', '_lower', '_upper', '_expr')

    def __init__(self, expr=None, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) else None
        self._active = True

        self._body = None
        self._lower = None
        self._upper = None
        self._expr = None
        if expr is not None:
            self.set_value(expr)

    #
    # Abstract Interface
    #

    @property
    def body(self):
        """Access the body of a constraint expression."""
        if self._body is not None:
            return self._body
        # The incoming RangedInequality had a potentially variable
        # bound.  The "body" is fine, but the bounds may not be
        # (although the responsibility for those checks lies with the
        # lower/upper properties)
        body = self._expr.arg(1)
        if body.__class__ in native_types and body is not None:
            return as_numeric(body)
        return body

    def _get_range_bound(self, range_arg):
        # Equalities and simple inequalities can always be (directly)
        # reformulated at construction time to force constant bounds.
        # The only time we need to defer the determination of bounds is
        # for ranged inequalities that contain non-constant bounds (so
        # we *know* that the expr will have 3 args)
        #
        # It is possible that there is no expression at all (so catch that)
        if self._expr is None:
            return None
        bound = self._expr.arg(range_arg)
        if not is_fixed(bound):
            raise ValueError(
                "Constraint '%s' is a Ranged Inequality with a "
                "variable %s bound.  Cannot normalize the "
                "constraint or send it to a solver."
                % (self.name, {0: 'lower', 2: 'upper'}[range_arg])
            )
        return bound

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        bound = self._lower if self._body is not None else self._get_range_bound(0)
        # Historically, constraint.lower was guaranteed to return a type
        # derived from Pyomo NumericValue (or None).  Replicate that
        # functionality, although clients should in almost all cases
        # move to using ConstraintData.lb instead of accessing
        # lower/body/upper to avoid the unnecessary creation (and
        # inevitable destruction) of the NumericConstant wrappers.
        if bound is None:
            return None
        return as_numeric(bound)

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        bound = self._upper if self._body is not None else self._get_range_bound(2)
        # Historically, constraint.upper was guaranteed to return a type
        # derived from Pyomo NumericValue (or None).  Replicate that
        # functionality, although clients should in almost all cases
        # move to using ConstraintData.ub instead of accessing
        # lower/body/upper to avoid the unnecessary creation (and
        # inevitable destruction) of the NumericConstant wrappers.
        if bound is None:
            return None
        return as_numeric(bound)

    @property
    def lb(self):
        """Access the value of the lower bound of a constraint expression."""
        bound = self._lower if self._body is not None else self._get_range_bound(0)
        if bound.__class__ not in native_numeric_types:
            if bound is None:
                return None
            bound = float(value(bound))
        if bound in _nonfinite_values or bound != bound:
            # Note that "bound != bound" catches float('nan')
            if bound == -_inf:
                return None
            else:
                raise ValueError(
                    "Constraint '%s' created with an invalid non-finite "
                    "lower bound (%s)." % (self.name, bound)
                )
        return bound

    @property
    def ub(self):
        """Access the value of the upper bound of a constraint expression."""
        bound = self._upper if self._body is not None else self._get_range_bound(2)
        if bound.__class__ not in native_numeric_types:
            if bound is None:
                return None
            bound = float(value(bound))
        if bound in _nonfinite_values or bound != bound:
            # Note that "bound != bound" catches float('nan')
            if bound == _inf:
                return None
            else:
                raise ValueError(
                    "Constraint '%s' created with an invalid non-finite "
                    "upper bound (%s)." % (self.name, bound)
                )
        return bound

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        if self._expr.__class__ is EqualityExpression:
            return True
        elif self._expr.__class__ is RangedExpression:
            # TODO: this is a very restrictive form of structural equality.
            lb = self._expr.arg(0)
            if lb is not None and lb is self._expr.arg(2):
                return True
        return False

    @property
    def strict_lower(self):
        """True if this constraint has a strict lower bound."""
        return False

    @property
    def strict_upper(self):
        """True if this constraint has a strict upper bound."""
        return False

    @property
    def expr(self):
        """Return the expression associated with this constraint."""
        return self._expr

    def get_value(self):
        """Get the expression on this constraint."""
        return self._expr

    def set_value(self, expr):
        """Set the expression on this constraint."""
        # Clear any previously-cached normalized constraint
        self._lower = self._upper = self._body = self._expr = None

        if expr.__class__ in _known_relational_expressions:
            self._expr = expr
        elif expr.__class__ is tuple:  # or expr_type is list:
            for arg in expr:
                if (
                    arg is None
                    or arg.__class__ in native_numeric_types
                    or isinstance(arg, NumericValue)
                ):
                    continue
                raise ValueError(
                    "Constraint '%s' does not have a proper value. "
                    "Constraint expressions expressed as tuples must "
                    "contain native numeric types or Pyomo NumericValue "
                    "objects. Tuple %s contained invalid type, %s"
                    % (self.name, expr, arg.__class__.__name__)
                )
            if len(expr) == 2:
                #
                # Form equality expression
                #
                if expr[0] is None or expr[1] is None:
                    raise ValueError(
                        "Constraint '%s' does not have a proper value. "
                        "Equality Constraints expressed as 2-tuples "
                        "cannot contain None [received %s]" % (self.name, expr)
                    )
                self._expr = EqualityExpression(expr)
            elif len(expr) == 3:
                #
                # Form (ranged) inequality expression
                #
                if expr[0] is None:
                    self._expr = InequalityExpression(expr[1:], False)
                elif expr[2] is None:
                    self._expr = InequalityExpression(expr[:2], False)
                else:
                    self._expr = RangedExpression(expr, False)
            else:
                raise ValueError(
                    "Constraint '%s' does not have a proper value. "
                    "Found a tuple of length %d. Expecting a tuple of "
                    "length 2 or 3:\n"
                    "    Equality:   (left, right)\n"
                    "    Inequality: (lower, expression, upper)"
                    % (self.name, len(expr))
                )
        #
        # Ignore an 'empty' constraint
        #
        elif expr.__class__ is type:
            del self.parent_component()[self.index()]
            if expr is Constraint.Skip:
                return
            elif expr is Constraint.Infeasible:
                # TODO: create a trivial infeasible constraint.  This
                # could be useful in the case of GDP where certain
                # disjuncts are trivially infeasible, but we would still
                # like to express the disjunction.
                # del self.parent_component()[self.index()]
                raise ValueError("Constraint '%s' is always infeasible" % (self.name,))
            else:
                raise ValueError(
                    "Constraint '%s' does not have a proper "
                    "value. Found '%s'\nExpecting a tuple or "
                    "relational expression. Examples:"
                    "\n   sum(model.costs) == model.income"
                    "\n   (0, model.price[item], 50)" % (self.name, str(expr))
                )

        elif expr is None:
            raise ValueError(_rule_returned_none_error % (self.name,))

        elif expr.__class__ is bool:
            raise ValueError(
                "Invalid constraint expression. The constraint "
                "expression resolved to a trivial Boolean (%s) "
                "instead of a Pyomo object. Please modify your "
                "rule to return Constraint.%s instead of %s."
                "\n\nError thrown for Constraint '%s'"
                % (expr, "Feasible" if expr else "Infeasible", expr, self.name)
            )

        else:
            try:
                if expr.is_expression_type(ExpressionType.RELATIONAL):
                    self._expr = expr
            except AttributeError:
                pass
            if self._expr is None:
                msg = (
                    "Constraint '%s' does not have a proper "
                    "value. Found '%s'\nExpecting a tuple or "
                    "relational expression. Examples:"
                    "\n   sum(model.costs) == model.income"
                    "\n   (0, model.price[item], 50)" % (self.name, str(expr))
                )
                raise ValueError(msg)
        #
        # Normalize the incoming expressions, if we can
        #
        args = self._expr.args
        if self._expr.__class__ is InequalityExpression:
            if self._expr.strict:
                raise ValueError(
                    "Constraint '%s' encountered a strict "
                    "inequality expression ('>' or '< '). All"
                    " constraints must be formulated using "
                    "using '<=', '>=', or '=='." % (self.name,)
                )
            if (
                args[1] is None
                or args[1].__class__ in native_numeric_types
                or not args[1].is_potentially_variable()
            ):
                self._body = args[0]
                self._upper = args[1]
            elif (
                args[0] is None
                or args[0].__class__ in native_numeric_types
                or not args[0].is_potentially_variable()
            ):
                self._lower = args[0]
                self._body = args[1]
            else:
                self._body = args[0] - args[1]
                self._upper = 0
        elif self._expr.__class__ is EqualityExpression:
            if args[0] is None or args[1] is None:
                # Error check: ensure equality does not have infinite RHS
                raise ValueError(
                    "Equality constraint '%s' defined with "
                    "non-finite term (%sHS == None)."
                    % (self.name, 'L' if args[0] is None else 'R')
                )
            if (
                args[0].__class__ in native_numeric_types
                or not args[0].is_potentially_variable()
            ):
                self._lower = self._upper = args[0]
                self._body = args[1]
            elif (
                args[1].__class__ in native_numeric_types
                or not args[1].is_potentially_variable()
            ):
                self._lower = self._upper = args[1]
                self._body = args[0]
            else:
                self._lower = self._upper = 0
                self._body = args[0] - args[1]
            # The following logic is caught below when checking for
            # invalid non-finite bounds:
            #
            # if self._lower.__class__ in native_numeric_types and \
            #    not math.isfinite(self._lower):
            #     raise ValueError(
            #         "Equality constraint '%s' defined with "
            #         "non-finite term." % (self.name))
        elif self._expr.__class__ is RangedExpression:
            if any(self._expr.strict):
                raise ValueError(
                    "Constraint '%s' encountered a strict "
                    "inequality expression ('>' or '< '). All"
                    " constraints must be formulated using "
                    "using '<=', '>=', or '=='." % (self.name,)
                )
            if all(
                (
                    arg is None
                    or arg.__class__ in native_numeric_types
                    or not arg.is_potentially_variable()
                )
                for arg in (args[0], args[2])
            ):
                self._lower, self._body, self._upper = args
        else:
            # Defensive programming: we currently only support three
            # relational expression types.  This will only be hit if
            # someone defines a fourth...
            raise DeveloperError(
                "Unrecognized relational expression type: %s"
                % (self._expr.__class__.__name__,)
            )

        # We have historically forced the body to be a numeric expression.
        # TODO: remove this requirement
        if self._body.__class__ in native_types and self._body is not None:
            self._body = as_numeric(self._body)

        # We have historically mapped incoming inf to None
        if self._lower.__class__ in native_numeric_types:
            bound = self._lower
            if bound in _nonfinite_values or bound != bound:
                # Note that "bound != bound" catches float('nan')
                if bound == -_inf:
                    self._lower = None
                else:
                    raise ValueError(
                        "Constraint '%s' created with an invalid non-finite "
                        "lower bound (%s)." % (self.name, self._lower)
                    )
        if self._upper.__class__ in native_numeric_types:
            bound = self._upper
            if bound in _nonfinite_values or bound != bound:
                # Note that "bound != bound" catches float('nan')
                if bound == _inf:
                    self._upper = None
                else:
                    raise ValueError(
                        "Constraint '%s' created with an invalid non-finite "
                        "upper bound (%s)." % (self.name, self._upper)
                    )


@ModelComponentFactory.register("General constraint expressions.")
class Constraint(ActiveIndexedComponent):
    """
    This modeling component defines a constraint expression using a
    rule function.

    Constructor arguments:
        expr
            A Pyomo expression for this constraint
        rule
            A function that is used to construct constraint expressions
        name
            A name for this component
        doc
            A text string describing this component

    Public class attributes:
        doc
            A text string describing this component
        name
            A name for this component
        active
            A boolean that is true if this component will be used to
            construct a model instance
        rule
           The rule used to initialize the constraint(s)

    Private class attributes:
        _constructed
            A boolean that is true if this component has been constructed
        _data
            A dictionary from the index set to component data objects
        _index
            The set of valid indices
        _implicit_subsets
            A tuple of set objects that represents the index set
        _model
            A weakref to the model that owns this component
        _parent
            A weakref to the parent block that owns this component
        _type
            The class type for the derived subclass
    """

    _ComponentDataClass = _GeneralConstraintData

    class Infeasible(object):
        pass

    Feasible = ActiveIndexedComponent.Skip
    NoConstraint = ActiveIndexedComponent.Skip
    Violated = Infeasible
    Satisfied = Feasible

    def __new__(cls, *args, **kwds):
        if cls != Constraint:
            return super(Constraint, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return super(Constraint, cls).__new__(AbstractScalarConstraint)
        else:
            return super(Constraint, cls).__new__(IndexedConstraint)

    @overload
    def __init__(self, *indexes, expr=None, rule=None, name=None, doc=None):
        ...

    def __init__(self, *args, **kwargs):
        _init = self._pop_from_kwargs('Constraint', kwargs, ('rule', 'expr'), None)
        # Special case: we accept 2- and 3-tuples as constraints
        if type(_init) is tuple:
            self.rule = Initializer(_init, treat_sequences_as_mappings=False)
        else:
            self.rule = Initializer(_init)

        kwargs.setdefault('ctype', Constraint)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug("Constructing constraint %s" % (self.name))

        rule = self.rule
        try:
            # We do not (currently) accept data for constructing Constraints
            index = None
            assert data is None

            if rule is None:
                # If there is no rule, then we are immediately done.
                return

            if rule.constant() and self.is_indexed():
                raise IndexError(
                    "Constraint '%s': Cannot initialize multiple indices "
                    "of a constraint with a single expression" % (self.name,)
                )

            block = self.parent_block()
            if rule.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in rule.indices():
                    self[index] = rule(block, index)
            elif not self.index_set().isfinite():
                # If the index is not finite, then we cannot iterate
                # over it.  Since the rule doesn't provide explicit
                # indices, then there is nothing we can do (the
                # assumption is that the user will trigger specific
                # indices to be created at a later time).
                pass
            else:
                # Bypass the index validation and create the member directly
                for index in self.index_set():
                    self._setitem_when_not_present(index, rule(block, index))
        except Exception:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed when generating expression for "
                "Constraint %s with index %s:\n%s: %s"
                % (self.name, str(index), type(err).__name__, err)
            )
            raise
        finally:
            timer.report()

    def _getitem_when_not_present(self, idx):
        if self.rule is None:
            raise KeyError(idx)
        con = self._setitem_when_not_present(idx, self.rule(self.parent_block(), idx))
        if con is None:
            raise KeyError(idx)
        return con

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [
                ("Size", len(self)),
                ("Index", self._index_set if self.is_indexed() else None),
                ("Active", self.active),
            ],
            self.items(),
            ("Lower", "Body", "Upper", "Active"),
            lambda k, v: [
                "-Inf" if v.lower is None else v.lower,
                v.body,
                "+Inf" if v.upper is None else v.upper,
                v.active,
            ],
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
        tab = "    "
        ostream.write(prefix + self.local_name + " : ")
        ostream.write("Size=" + str(len(self)))

        ostream.write("\n")
        tabular_writer(
            ostream,
            prefix + tab,
            ((k, v) for k, v in self._data.items() if v.active),
            ("Lower", "Body", "Upper"),
            lambda k, v: [
                value(v.lower, exception=False),
                value(v.body, exception=False),
                value(v.upper, exception=False),
            ],
        )


class ScalarConstraint(_GeneralConstraintData, Constraint):
    """
    ScalarConstraint is the implementation representing a single,
    non-indexed constraint.
    """

    def __init__(self, *args, **kwds):
        _GeneralConstraintData.__init__(self, component=self, expr=None)
        Constraint.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

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
    @property
    def body(self):
        """Access the body of a constraint expression."""
        if not self._data:
            raise ValueError(
                "Accessing the body of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return _GeneralConstraintData.body.fget(self)

    @property
    def lower(self):
        """Access the lower bound of a constraint expression."""
        if not self._data:
            raise ValueError(
                "Accessing the lower bound of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return _GeneralConstraintData.lower.fget(self)

    @property
    def upper(self):
        """Access the upper bound of a constraint expression."""
        if not self._data:
            raise ValueError(
                "Accessing the upper bound of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return _GeneralConstraintData.upper.fget(self)

    @property
    def equality(self):
        """A boolean indicating whether this is an equality constraint."""
        if not self._data:
            raise ValueError(
                "Accessing the equality flag of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return _GeneralConstraintData.equality.fget(self)

    @property
    def strict_lower(self):
        """A boolean indicating whether this constraint has a strict lower bound."""
        if not self._data:
            raise ValueError(
                "Accessing the strict_lower flag of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return _GeneralConstraintData.strict_lower.fget(self)

    @property
    def strict_upper(self):
        """A boolean indicating whether this constraint has a strict upper bound."""
        if not self._data:
            raise ValueError(
                "Accessing the strict_upper flag of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return _GeneralConstraintData.strict_upper.fget(self)

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression on this constraint."""
        if not self._data:
            self._data[None] = self
        return super(ScalarConstraint, self).set_value(expr)

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add a constraint with a given index."""
        if index is not None:
            raise ValueError(
                "ScalarConstraint object '%s' does not accept "
                "index values other than None. Invalid value: %s" % (self.name, index)
            )
        self.set_value(expr)
        return self


class SimpleConstraint(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarConstraint
    __renamed__version__ = '6.0'


@disable_methods(
    {
        'add',
        'set_value',
        'body',
        'lower',
        'upper',
        'equality',
        'strict_lower',
        'strict_upper',
    }
)
class AbstractScalarConstraint(ScalarConstraint):
    pass


class AbstractSimpleConstraint(metaclass=RenamedClass):
    __renamed__new_class__ = AbstractScalarConstraint
    __renamed__version__ = '6.0'


class IndexedConstraint(Constraint):
    #
    # Leaving this method for backward compatibility reasons
    #
    # Note: Beginning after Pyomo 5.2 this method will now validate that
    # the index is in the underlying index set (through 5.2 the index
    # was not checked).
    #
    def add(self, index, expr):
        """Add a constraint with a given index."""
        return self.__setitem__(index, expr)


@ModelComponentFactory.register("A list of constraint expressions.")
class ConstraintList(IndexedConstraint):
    """
    A constraint component that represents a list of constraints.
    Constraints can be indexed by their index, but when they are
    added an index value is not specified.
    """

    class End(object):
        pass

    def __init__(self, **kwargs):
        """Constructor"""
        if 'expr' in kwargs:
            raise ValueError("ConstraintList does not accept the 'expr' keyword")
        _rule = kwargs.pop('rule', None)
        self._starting_index = kwargs.pop('starting_index', 1)

        args = (Set(dimen=1),)
        super(ConstraintList, self).__init__(*args, **kwargs)

        self.rule = Initializer(
            _rule, treat_sequences_as_mappings=False, allow_generators=True
        )
        # HACK to make the "counted call" syntax work.  We wait until
        # after the base class is set up so that is_indexed() is
        # reliable.
        if self.rule is not None and type(self.rule) is IndexedCallInitializer:
            self.rule = CountedCallInitializer(self, self.rule, self._starting_index)

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        if self._constructed:
            return
        self._constructed = True

        if is_debug_set(logger):
            logger.debug("Constructing constraint list %s" % (self.name))

        self.index_set().construct()

        if self.rule is not None:
            _rule = self.rule(self.parent_block(), ())
            for cc in iter(_rule):
                if cc is ConstraintList.End:
                    break
                if cc is Constraint.Skip:
                    continue
                self.add(cc)

    def add(self, expr):
        """Add a constraint with an implicit index."""
        next_idx = len(self._index_set) + self._starting_index
        self._index_set.add(next_idx)
        return self.__setitem__(next_idx, expr)
