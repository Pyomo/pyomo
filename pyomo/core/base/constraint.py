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

from __future__ import annotations
import sys
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from typing import Union, Type

from pyomo.common.deprecation import RenamedClass, deprecated
from pyomo.common.errors import DeveloperError, TemplateExpressionError
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
    native_logical_types,
    native_types,
)
from pyomo.core.expr import (
    ExpressionType,
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
)
from pyomo.core.expr.expr_common import _type_check_exception_arg
from pyomo.core.expr.relational_expr import TrivialRelationalExpression
from pyomo.core.expr.template_expr import templatize_constraint
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent,
    UnindexedComponent_set,
    rule_wrapper,
    IndexedComponent,
)
from pyomo.core.base.set import Set
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
    Initializer,
    IndexedCallInitializer,
    CountedCallInitializer,
)


logger = logging.getLogger('pyomo.core')

TEMPLATIZE_CONSTRAINTS = False

_inf = float('inf')
_ninf = -_inf
_nonfinite_values = {_inf, _ninf}
_known_relational_expression_types = {
    EqualityExpression,
    InequalityExpression,
    RangedExpression,
    TrivialRelationalExpression,
}
_strict_relational_exprs = {True, (False, True), (True, False), (True, True)}
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

    .. code::

        @simple_constraint_rule
        def C_rule(model, i, j):
            # ...

        model.c = Constraint(rule=simple_constraint_rule(...))

    """
    map_types = set([type(None)]) | native_logical_types
    result_map = {None: Constraint.Skip}
    for l_type in native_logical_types:
        result_map[l_type(True)] = Constraint.Feasible
        result_map[l_type(False)] = Constraint.Infeasible
    # Note: some logical types hash the same as bool (e.g., np.bool_), so
    # we will pass the set of all logical types in addition to the
    # result_map
    return rule_wrapper(rule, result_map, map_types=map_types)


def simple_constraintlist_rule(rule):
    """
    This is a decorator that translates None/True/False return values
    into ConstraintList.End/Constraint.Feasible/Constraint.Infeasible.
    This supports a simpler syntax in constraint rules, though these can be
    more difficult to debug when errors occur.

    Example use:

    .. code::

        @simple_constraintlist_rule
        def C_rule(model, i, j):
             # ...

        model.c = ConstraintList(expr=simple_constraintlist_rule(...))

    """
    map_types = set([type(None)]) | native_logical_types
    result_map = {None: ConstraintList.End}
    for l_type in native_logical_types:
        result_map[l_type(True)] = Constraint.Feasible
        result_map[l_type(False)] = Constraint.Infeasible
    # Note: some logical types hash the same as bool (e.g., np.bool_), so
    # we will pass the set of all logical types in addition to the
    # result_map
    return rule_wrapper(rule, result_map, map_types=map_types)


class ConstraintData(ActiveComponentData):
    """This class defines the data for a single algebraic constraint.

    Parameters
    ----------
    expr : ExpressionBase
        The Pyomo expression stored in this constraint.

    component : Constraint
        The Constraint object that owns this data.

    """

    __slots__ = ('_expr',)

    # Set to true when a constraint class stores its expression
    # in linear canonical form
    _linear_canonical_form = False

    def __init__(self, expr=None, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ConstraintData
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) else None
        self._active = True

        self._expr = None
        if expr is not None:
            self.set_value(expr)

    def __call__(self, exception=NOTSET):
        """Compute the value of the body of this constraint."""
        exception = _type_check_exception_arg(self, exception)
        body = self.to_bounded_expression()[1]
        if body.__class__ not in native_numeric_types:
            body = value(self.body, exception=exception)
        return body

    def to_bounded_expression(self, evaluate_bounds=False):
        """Convert this constraint to a tuple of 3 expressions (lb, body, ub)

        This method "standardizes" the expression into a 3-tuple of
        expressions: (`lower_bound`, `body`, `upper_bound`).  Upon
        conversion, `lower_bound` and `upper_bound` are guaranteed to be
        `None`, numeric constants, or fixed (not necessarily constant)
        expressions.

        Note
        ----
        As this method operates on the *current state* of the
        expression, any required expression manipulations (and by
        extension, the result) can change after fixing / unfixing
        :py:class:`Var` objects.

        Parameters
        ----------
        evaluate_bounds: bool

            If True, then the lower and upper bounds will be evaluated
            to a finite numeric constant or None.

        Raises
        ------

        ValueError: Raised if the expression cannot be mapped to this
            form (i.e., :py:class:`RangedExpression` constraints with
            variable lower or upper bounds.

        """
        expr = self._expr
        if expr.__class__ is RangedExpression:
            lb, body, ub = ans = expr.args
            if (
                lb.__class__ not in native_types
                and lb.is_potentially_variable()
                and not lb.is_fixed()
            ):
                raise ValueError(
                    f"Constraint '{self.name}' is a Ranged Inequality with a "
                    "variable lower bound.  Cannot normalize the "
                    "constraint or send it to a solver."
                )
            if (
                ub.__class__ not in native_types
                and ub.is_potentially_variable()
                and not ub.is_fixed()
            ):
                raise ValueError(
                    f"Constraint '{self.name}' is a Ranged Inequality with a "
                    "variable upper bound.  Cannot normalize the "
                    "constraint or send it to a solver."
                )
        elif expr is None:
            ans = None, None, None
        else:
            lhs, rhs = expr.args
            if rhs.__class__ in native_types or not rhs.is_potentially_variable():
                ans = rhs if expr.__class__ is EqualityExpression else None, lhs, rhs
            elif lhs.__class__ in native_types or not lhs.is_potentially_variable():
                ans = lhs, rhs, lhs if expr.__class__ is EqualityExpression else None
            else:
                ans = 0 if expr.__class__ is EqualityExpression else None, lhs - rhs, 0

        if evaluate_bounds:
            lb, body, ub = ans
            return self._evaluate_bound(lb, _ninf), body, self._evaluate_bound(ub, _inf)
        return ans

    def _evaluate_bound(self, bound, unbounded):
        if bound is None:
            return None
        if bound.__class__ not in native_numeric_types:
            bound = value(bound)
            if bound.__class__ not in native_numeric_types:
                # Starting in numpy 1.25, casting 1-element ndarray to
                # float is deprecated.  We still want to support
                # that... but without enforcing a hard numpy dependence
                for cls in bound.__class__.__mro__:
                    if cls.__name__ == 'ndarray' and cls.__module__ == 'numpy':
                        if len(bound) == 1:
                            bound = bound[0]
                        break
                bound = float(bound)
        # Note that "bound != bound" catches float('nan')
        if bound in _nonfinite_values or bound != bound:
            if bound == unbounded:
                return None
            raise ValueError(
                f"Constraint '{self.name}' created with an invalid non-finite "
                f"{'upper' if unbounded==_inf else 'lower'} bound ({bound})."
            )
        return bound

    @property
    def body(self):
        """The body (variable portion) of a constraint expression."""
        try:
            ans = self.to_bounded_expression()[1]
        except ValueError:
            # It is possible that the expression is not currently valid
            # (i.e., a ranged expression with a non-fixed bound).  We
            # will catch that exception here and - if this actually *is*
            # a RangedExpression - return the body.
            if self._expr.__class__ is RangedExpression:
                _, ans, _ = self._expr.args
            else:
                raise
        if ans.__class__ in native_types and ans is not None:
            # Historically, constraint.lower was guaranteed to return a type
            # derived from Pyomo NumericValue (or None).  Replicate that.
            #
            # [JDS 6/2024: it would be nice to remove this behavior,
            # although possibly unnecessary, as people should use
            # to_bounded_expression() instead]
            return as_numeric(ans)
        return ans

    @property
    def lower(self):
        """The lower bound of a constraint expression.

        This is the fixed lower bound of a Constraint as a Pyomo
        expression.  This may contain potentially variable terms
        that are currently fixed.  If there is no lower bound, this will
        return `None`.

        """
        ans = self.to_bounded_expression()[0]
        if ans.__class__ in native_types and ans is not None:
            # Historically, constraint.lower was guaranteed to return a type
            # derived from Pyomo NumericValue (or None).  Replicate that
            # functionality, although clients should in almost all cases
            # move to using ConstraintData.lb instead of accessing
            # lower/body/upper to avoid the unnecessary creation (and
            # inevitable destruction) of the NumericConstant wrappers.
            return as_numeric(ans)
        return ans

    @property
    def upper(self):
        """Access the upper bound of a constraint expression.

        This is the fixed upper bound of a Constraint as a Pyomo
        expression.  This may contain potentially variable terms
        that are currently fixed.  If there is no upper bound, this will
        return `None`.

        """
        ans = self.to_bounded_expression()[2]
        if ans.__class__ in native_types and ans is not None:
            # Historically, constraint.upper was guaranteed to return a type
            # derived from Pyomo NumericValue (or None).  Replicate that
            # functionality, although clients should in almost all cases
            # move to using ConstraintData.lb instead of accessing
            # lower/body/upper to avoid the unnecessary creation (and
            # inevitable destruction) of the NumericConstant wrappers.
            return as_numeric(ans)
        return ans

    @property
    def lb(self):
        """float : the value of the lower bound of a constraint expression."""
        return self._evaluate_bound(self.to_bounded_expression()[0], _ninf)

    @property
    def ub(self):
        """float : the value of the upper bound of a constraint expression."""
        return self._evaluate_bound(self.to_bounded_expression()[2], _inf)

    @property
    def equality(self):
        """bool : True if this is an equality constraint."""
        expr = self.expr
        if expr.__class__ is EqualityExpression:
            return True
        elif expr.__class__ is RangedExpression:
            # TODO: this is a very restrictive form of structural equality.
            lb = expr.arg(0)
            if lb is not None and lb is expr.arg(2):
                return True
        return False

    @property
    def strict_lower(self):
        """bool : True if this constraint has a strict lower bound."""
        return False

    @property
    def strict_upper(self):
        """bool : True if this constraint has a strict upper bound."""
        return False

    def has_lb(self):
        """Returns :const:`False` when the lower bound is
        :const:`None` or negative infinity"""
        return self.lb is not None

    def has_ub(self):
        """Returns :const:`False` when the upper bound is
        :const:`None` or positive infinity"""
        return self.ub is not None

    @property
    def expr(self):
        """Return the expression associated with this constraint."""
        return self._expr

    def get_value(self):
        """Get the expression on this constraint."""
        return self.expr

    def set_value(self, expr):
        """Set the expression on this constraint."""
        if expr.__class__ in _known_relational_expression_types:
            if getattr(expr, 'strict', False) in _strict_relational_exprs:
                raise ValueError(
                    "Constraint '%s' encountered a strict "
                    "inequality expression ('>' or '<').  All "
                    "constraints must be formulated using "
                    "using '<=', '>=', or '=='." % (self.name,)
                )
            self._expr = expr
            return

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
                    % (self.name, expr, type(arg).__name__)
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
                return
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
                return
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
        if expr is Constraint.Skip:
            del self.parent_component()[self.index()]
            return

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
                    return
            except AttributeError:
                pass

        raise ValueError(
            "Constraint '%s' does not have a proper "
            "value. Found %s '%s'\nExpecting a tuple or "
            "relational expression. Examples:"
            "\n   sum(model.costs) == model.income"
            "\n   (0, model.price[item], 50)"
            % (self.name, type(expr).__name__, str(expr))
        )

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


class _ConstraintData(metaclass=RenamedClass):
    __renamed__new_class__ = ConstraintData
    __renamed__version__ = '6.7.2'


class _GeneralConstraintData(metaclass=RenamedClass):
    __renamed__new_class__ = ConstraintData
    __renamed__version__ = '6.7.2'


class TemplateDataMixin(object):
    __slots__ = ()

    @property
    def expr(self):
        # Note that it is faster to just generate the expression from
        # scratch than it is to clone it and replace the IndexTemplate objects
        self.set_value(self.parent_component()._rule(self.parent_block(), self.index()))
        return self.expr

    def template_expr(self):
        return self._expr

    def set_value(self, expr):
        # Setting a value will convert this instance from a templatized
        # type to the original Data type (and call the original set_value()).
        #
        # Note: We assume that the templatized type is created by
        # inheriting (TemplateDataMixin, <original data class>), and
        # that this instance doesn't have additional multiple
        # inheritance that could re-order the MRO.
        self.__class__ = self.__class__.__mro__[
            self.__class__.__mro__.index(TemplateDataMixin) + 1
        ]
        return self.set_value(expr)

    def to_bounded_expression(self, evaluate_bounds=False):
        tmp, self._expr = self._expr, self._expr[0]
        try:
            return super().to_bounded_expression(evaluate_bounds)
        finally:
            self._expr = tmp


class TemplateConstraintData(TemplateDataMixin, ConstraintData):
    __slots__ = ()

    def __init__(self, template_info, component, index):
        # These lines represent in-lining of the
        # following constructors:
        #   - ConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = component
        self._active = True
        self._index = index
        self._expr = template_info


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
        _model
            A weakref to the model that owns this component
        _parent
            A weakref to the parent block that owns this component
        _type
            The class type for the derived subclass
    """

    _ComponentDataClass = ConstraintData

    Infeasible = TrivialRelationalExpression('Infeasible', (1, 0))
    Feasible = TrivialRelationalExpression('Feasible', (0, 0))

    NoConstraint = ActiveIndexedComponent.Skip
    Violated = Infeasible
    Satisfied = Feasible

    @overload
    def __new__(
        cls: Type[Constraint], *args, **kwds
    ) -> Union[ScalarConstraint, IndexedConstraint]: ...

    @overload
    def __new__(cls: Type[ScalarConstraint], *args, **kwds) -> ScalarConstraint: ...

    @overload
    def __new__(cls: Type[IndexedConstraint], *args, **kwds) -> IndexedConstraint: ...

    def __new__(cls, *args, **kwds):
        if cls != Constraint:
            return super().__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return super().__new__(AbstractScalarConstraint)
        else:
            return super().__new__(IndexedConstraint)

    @overload
    def __init__(self, *indexes, expr=None, rule=None, name=None, doc=None): ...

    def __init__(self, *args, **kwargs):
        _init = self._pop_from_kwargs('Constraint', kwargs, ('rule', 'expr'), None)
        # Special case: we accept 2- and 3-tuples as constraints
        if type(_init) is tuple:
            self._rule = Initializer(_init, treat_sequences_as_mappings=False)
        else:
            self._rule = Initializer(_init)

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

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        rule = self._rule
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
                if TEMPLATIZE_CONSTRAINTS:
                    try:
                        template_info = templatize_constraint(self)
                        if self.is_indexed():
                            comp = weakref_ref(self)
                            self._data = {
                                idx: TemplateConstraintData(template_info, comp, idx)
                                for idx in self.index_set()
                            }
                        else:
                            assert self.__class__ is ScalarConstraint
                            self.__class__ = TemplateScalarConstraint
                            self._expr = template_info
                            self._data = {None: self}
                        return
                    except TemplateExpressionError:
                        pass

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
        if self._rule is None:
            raise KeyError(idx)
        con = self._setitem_when_not_present(idx, self._rule(self.parent_block(), idx))
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

    @property
    def rule(self):
        return self._rule

    @rule.setter
    @deprecated(
        f"The 'Constraint.rule' attribute will be made "
        "read-only in a future Pyomo release.",
        version='6.9.3',
        remove_in='6.11',
    )
    def rule(self, rule):
        self._rule = rule

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


class ScalarConstraint(ConstraintData, Constraint):
    """
    ScalarConstraint is the implementation representing a single,
    non-indexed constraint.
    """

    def __init__(self, *args, **kwds):
        ConstraintData.__init__(self, component=self, expr=None)
        Constraint.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    #
    # Singleton constraints are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Constraint.Skip are managed. But after that they will behave
    # like ConstraintData objects where set_value does not handle
    # Constraint.Skip but expects a valid expression or None.
    #
    @property
    def body(self):
        """The body (variable portion) of a constraint expression."""
        if not self._data:
            raise ValueError(
                "Accessing the body of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return ConstraintData.body.fget(self)

    @property
    def lower(self):
        """The lower bound of a constraint expression.

        This is the fixed lower bound of a Constraint as a Pyomo
        expression.  This may contain potentially variable terms
        that are currently fixed.  If there is no lower bound, this will
        return `None`.

        """
        if not self._data:
            raise ValueError(
                "Accessing the lower bound of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return ConstraintData.lower.fget(self)

    @property
    def upper(self):
        """Access the upper bound of a constraint expression.

        This is the fixed upper bound of a Constraint as a Pyomo
        expression.  This may contain potentially variable terms
        that are currently fixed.  If there is no upper bound, this will
        return `None`.

        """
        if not self._data:
            raise ValueError(
                "Accessing the upper bound of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return ConstraintData.upper.fget(self)

    @property
    def equality(self):
        """bool : True if this is an equality constraint."""
        if not self._data:
            raise ValueError(
                "Accessing the equality flag of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return ConstraintData.equality.fget(self)

    @property
    def strict_lower(self):
        """bool : True if this constraint has a strict lower bound."""
        if not self._data:
            raise ValueError(
                "Accessing the strict_lower flag of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return ConstraintData.strict_lower.fget(self)

    @property
    def strict_upper(self):
        """bool : True if this constraint has a strict upper bound."""
        if not self._data:
            raise ValueError(
                "Accessing the strict_upper flag of ScalarConstraint "
                "'%s' before the Constraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % (self.name)
            )
        return ConstraintData.strict_upper.fget(self)

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression on this constraint."""
        if not self._data:
            self._data[None] = self
        return super().set_value(expr)

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
        '__call__',
        'add',
        'set_value',
        'to_bounded_expression',
        'expr',
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


class TemplateScalarConstraint(TemplateDataMixin, ScalarConstraint):
    pass


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

    @overload
    def __getitem__(self, index) -> ConstraintData: ...

    __getitem__ = IndexedComponent.__getitem__  # type: ignore


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

        super().__init__(Set(dimen=1), **kwargs)

        self._rule = Initializer(
            _rule, treat_sequences_as_mappings=False, allow_generators=True
        )
        # HACK to make the "counted call" syntax work.  We wait until
        # after the base class is set up so that is_indexed() is
        # reliable.
        if self._rule is not None and type(self._rule) is IndexedCallInitializer:
            self._rule = CountedCallInitializer(self, self._rule, self._starting_index)

    def construct(self, data=None):
        """
        Construct the expression(s) for this constraint.
        """
        if self._constructed:
            return
        self._constructed = True

        if is_debug_set(logger):
            logger.debug("Constructing constraint list %s" % (self.name))

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        if self._rule is not None:
            _rule = self._rule(self.parent_block(), ())
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
