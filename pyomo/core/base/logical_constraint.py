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

from pyomo.common.deprecation import RenamedClass, deprecated
from pyomo.common.formatting import tabular_writer
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer

from pyomo.core.expr.expr_common import _type_check_exception_arg
from pyomo.core.expr.boolean_value import as_boolean, BooleanConstant
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.base.component import ActiveComponentData, ModelComponentFactory
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
    ActiveIndexedComponent,
    UnindexedComponent_set,
)
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.set import Set

logger = logging.getLogger(__name__)
_known_logical_expression_types = set()
_rule_returned_none_error = """LogicalConstraint '%s': rule returned None.

logical constraint rules must return a valid logical proposition.
The most common cause of this error is
forgetting to include the "return" statement at the end of your rule.
"""


class LogicalConstraintData(ActiveComponentData):
    """
    This class defines the data for a single general logical constraint.

    Constructor arguments:
        component
            The LogicalConstraint object that owns this data.
        expr
            The Pyomo expression stored in this logical constraint.

    Public class attributes:
        active
            A boolean that is true if this logical constraint is
            active in the model.
        expr
            The Pyomo expression for this logical constraint

    Private class attributes:
        _component
            The logical constraint component.
        _active
            A boolean that indicates whether this data is active

    """

    __slots__ = ('_expr',)

    def __init__(self, expr=None, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - LogicalConstraintData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) else None
        self._index = NOTSET
        self._active = True

        self._expr = None
        if expr is not None:
            self.set_value(expr)

    def __call__(self, exception=NOTSET):
        """Compute the value of the body of this logical constraint."""
        exception = _type_check_exception_arg(self, exception)
        return self.expr(exception=exception)

    #
    # Abstract Interface
    #

    @property
    def body(self):
        """Access the body of a logical constraint expression."""
        return self.expr

    @property
    def expr(self):
        """Return the expression associated with this logical constraint."""
        return self._expr

    def set_value(self, expr):
        """Set the expression on this logical constraint."""
        if expr.__class__ in _known_logical_expression_types:
            self._expr = expr
            return
        #
        # Ignore an 'empty' constraint
        #
        if expr is LogicalConstraint.Skip:
            del self.parent_component()[self.index()]
            return

        elif expr.__class__ in native_logical_types:
            self._expr = as_boolean(expr)
            return

        elif expr is None:
            raise ValueError(_rule_returned_none_error % (self.name,))

        elif expr.__class__ not in native_types:
            try:
                if expr.is_logical_type():
                    self._expr = expr
                    _known_logical_expression_types.add(expr.__class__)
                    return

                # FIXME: we should extend the templating system to
                # handle things like IntervalVar types so that
                # indirection and things like CallExpression can
                # properly propagate the expression type.  In the
                # meantime, we will just assume that all template
                # expressions are acceptable (which, while not correct,
                # is consistent with prior behavior)
                if hasattr(expr, '_resolve_template'):
                    self._expr = expr
                    return
            except (AttributeError, TypeError):
                pass

        raise ValueError(
            "Assigning improper value to LogicalConstraint '%s'. "
            "Found %s '%s'.\n"
            "Expecting a logical expression or Boolean value. Examples:"
            "\n   (m.Y1 & m.Y2).implies(m.Y3)"
            "\n   atleast(1, m.Y1, m.Y2)" % (self.name, type(expr).__name__, str(expr))
        )

    def get_value(self):
        """Get the expression on this logical constraint."""
        return self.expr


class _LogicalConstraintData(metaclass=RenamedClass):
    __renamed__new_class__ = LogicalConstraintData
    __renamed__version__ = '6.7.2'


class _GeneralLogicalConstraintData(metaclass=RenamedClass):
    __renamed__new_class__ = LogicalConstraintData
    __renamed__version__ = '6.7.2'


@ModelComponentFactory.register("General logical constraints.")
class LogicalConstraint(ActiveIndexedComponent):
    """
    This modeling component defines a logical constraint using a
    rule function.

    Constructor arguments:
        expr
            A Pyomo expression for this logical constraint
        rule
            A function that is used to construct logical constraints
        doc
            A text string describing this component
        name
            A name for this component

    Public class attributes:
        doc
            A text string describing this component
        name
            A name for this component
        active
            A boolean that is true if this component will be used to
            construct a model instance
        rule
           The rule used to initialize the logical constraint(s)

    Private class attributes:
        _constructed
            A boolean that is true if this component has been constructed
        _data
            A dictionary from the index set to component data objects
        _index_set
            The set of valid indices
        _model
            A weakref to the model that owns this component
        _parent
            A weakref to the parent block that owns this component
        _type
            The class type for the derived subclass
    """

    _ComponentDataClass = LogicalConstraintData

    Infeasible = BooleanConstant(False, 'Infeasible')
    Feasible = BooleanConstant(True, 'Feasible')

    NoConstraint = ActiveIndexedComponent.Skip
    Violated = Infeasible
    Satisfied = Feasible

    def __new__(cls, *args, **kwds):
        if cls != LogicalConstraint:
            return super().__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return super().__new__(AbstractScalarLogicalConstraint)
        else:
            return super().__new__(IndexedLogicalConstraint)

    def __init__(self, *args, **kwargs):
        _init = self._pop_from_kwargs('Constraint', kwargs, ('rule', 'expr'), None)
        self._rule = Initializer(_init)

        kwargs.setdefault('ctype', LogicalConstraint)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this logical constraint.
        """
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug("Constructing logical constraint %s" % self.name)

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        rule = self._rule
        try:
            # We do not (currently) accept data for constructing LogicalConstraints
            index = None
            assert data is None

            if rule is None:
                # If there is no rule, then we are immediately done.
                return

            if rule.constant() and self.is_indexed():
                raise IndexError(
                    "LogicalConstraint '%s': Cannot initialize multiple indices "
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
                "LogicalConstraint %s with index %s:\n%s: %s"
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
            ("Body", "Active"),
            lambda k, v: [v.body, v.active],
        )

    @property
    def rule(self):
        return self._rule

    @rule.setter
    @deprecated(
        f"The 'LogicalConstraint.rule' attribute will be made read-only",
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
            ("Body",),
            lambda k, v: [v.body()],
        )


class ScalarLogicalConstraint(LogicalConstraintData, LogicalConstraint):
    """
    ScalarLogicalConstraint is the implementation representing a single,
    non-indexed logical constraint.
    """

    def __init__(self, *args, **kwds):
        LogicalConstraintData.__init__(self, component=self, expr=None)
        LogicalConstraint.__init__(self, *args, **kwds)
        self._index = UnindexedComponent_index

    #
    # Override abstract interface methods to first check for
    # construction
    #

    @property
    def expr(self):
        """Access the body of a logical constraint."""
        if not self._data:
            raise ValueError(
                "Accessing the expr of ScalarLogicalConstraint "
                "'%s' before the LogicalConstraint has been assigned "
                "an expression. There is currently "
                "nothing to access." % self.name
            )
        return LogicalConstraintData.expr.fget(self)

    #
    # Singleton logical constraints are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # True are managed. But after that they will behave
    # like LogicalConstraintData objects where set_value expects
    # a valid expression or None.
    #

    def set_value(self, expr):
        """Set the expression on this logical constraint."""
        if not self._data:
            self._data[None] = self
        return super().set_value(expr)

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add a logical constraint with a given index."""
        if index is not None:
            raise ValueError(
                "ScalarLogicalConstraint object '%s' does not accept "
                "index values other than None. Invalid value: %s" % (self.name, index)
            )
        self.set_value(expr)
        return self


class SimpleLogicalConstraint(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarLogicalConstraint
    __renamed__version__ = '6.0'


@disable_methods({'add', 'set_value', 'expr', 'body'})
class AbstractScalarLogicalConstraint(ScalarLogicalConstraint):
    pass


class IndexedLogicalConstraint(LogicalConstraint):
    #
    # Leaving this method for backward compatibility reasons
    #
    # Note: Beginning after Pyomo 5.2 this method will now validate that
    # the index is in the underlying index set (through 5.2 the index
    # was not checked).
    #
    def add(self, index, expr):
        """Add a logical constraint with a given index."""
        return self.__setitem__(index, expr)

    @overload
    def __getitem__(self, index) -> LogicalConstraintData: ...

    __getitem__ = ActiveIndexedComponent.__getitem__  # type: ignore


@ModelComponentFactory.register("A list of logical constraints.")
class LogicalConstraintList(IndexedLogicalConstraint):
    """
    A logical constraint component that represents a list of constraints.
    Constraints can be indexed by their index, but when they are
    added an index value is not specified.
    """

    class End(object):
        pass

    def __init__(self, **kwargs):
        """Constructor"""
        if 'expr' in kwargs:
            raise ValueError("LogicalConstraintList does not accept the 'expr' keyword")
        _rule = kwargs.pop('rule', None)
        self._starting_index = kwargs.pop('starting_index', 1)

        super().__init__(Set(dimen=1), **kwargs)

        self._rule = Initializer(
            _rule, treat_sequences_as_mappings=False, allow_generators=True
        )

    def construct(self, data=None):
        """
        Construct the expression(s) for this logical constraint.
        """
        if self._constructed:
            return
        self._constructed = True

        if is_debug_set(logger):
            logger.debug("Constructing logical constraint list %s" % (self.name))

        if self._anonymous_sets is not None:
            for _set in self._anonymous_sets:
                _set.construct()

        if self._rule is not None:
            _rule = self._rule(self.parent_block(), ())
            for cc in iter(_rule):
                if cc is LogicalConstraintList.End:
                    break
                if cc is LogicalConstraint.Skip:
                    continue
                self.add(cc)

    def add(self, expr):
        """Add a logical constraint with an implicit index."""
        next_idx = len(self._index_set) + self._starting_index
        self._index_set.add(next_idx)
        return self.__setitem__(next_idx, expr)
