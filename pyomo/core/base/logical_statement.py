#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['LogicalStatement', '_LogicalStatementData', 'LogicalStatementList']

import inspect
import sys
import logging
from weakref import ref as weakref_ref

import pyutilib.math
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr import logical_expr
from pyomo.core.expr.logicalvalue import as_logical, LogicalConstant
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.base.plugin import ModelComponentFactory
from pyomo.core.base.component import ActiveComponentData
from pyomo.core.base.indexed_component import \
    (ActiveIndexedComponent,
     UnindexedComponent_set,
     _get_indexed_component_data_name, )
from pyomo.core.base.misc import (apply_indexed_rule,
                                  tabular_writer)
from pyomo.core.base.sets import Set

from six import StringIO, iteritems

logger = logging.getLogger('pyomo.core')

_rule_returned_none_error = """LogicalStatement '%s': rule returned None.

Logical statement rules must return a valid logical proposition.
The most common cause of this error is
forgetting to include the "return" statement at the end of your rule.
"""


class _LogicalStatementData(ActiveComponentData):
    """
    This class defines the data for a single logical statement.

    It functions as a pure interface.

    Constructor arguments:
        component       The LogicalStatement object that owns this data.

    Public class attributes:
        active          A boolean that is true if this statement is
                            active in the model.
        body            The Pyomo logical expression for this statement

    Private class attributes:
        _component      The statement component.
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ()

    def __init__(self, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
            else None
        self._active = True

    #
    # Interface
    #
    def __call__(self, exception=True):
        """Compute the value of the body of this logical statement."""
        if self.body is None:
            return None
        return self.body(exception=exception)

    #
    # Abstract Interface
    #
    @property
    def body(self):
        """Access the body of a logical statement."""
        raise NotImplementedError

    def set_value(self, expr):
        """Set the expression on this logical statement."""
        raise NotImplementedError

    def get_value(self):
        """Get the expression on this constraint."""
        raise NotImplementedError


class _GeneralLogicalStatementData(_LogicalStatementData):
    """
    This class defines the data for a single general logical statement.

    Constructor arguments:
        component       The LogicalStatment object that owns this data.
        expr            The Pyomo expression stored in this logical statement.

    Public class attributes:
        active          A boolean that is true if this logical statement is
                            active in the model.
        body            The Pyomo expression for this logical statement

    Private class attributes:
        _component      The logical statement component.
        _active         A boolean that indicates whether this data is active
    """

    __slots__ = ('_body',)

    def __init__(self, expr=None, component=None):
        #
        # These lines represent in-lining of the
        # following constructors:
        #   - _LogicalStatmentData,
        #   - ActiveComponentData
        #   - ComponentData
        self._component = weakref_ref(component) if (component is not None) \
            else None
        self._active = True

        self._body = None
        if expr is not None:
            self.set_value(expr)

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        result = super(_GeneralLogicalStatementData, self).__getstate__()
        for i in _GeneralLogicalStatementData.__slots__:
            result[i] = getattr(self, i)
        return result

    # Since this class requires no special processing of the state
    # dictionary, it does not need to implement __setstate__()

    #
    # Abstract Interface
    #

    @property
    def body(self):
        """Access the body of a logical statement expression."""
        return self._body

    @property
    def expr(self):
        """Return the expression associated with this logical statement."""
        return self.get_value()

    def set_value(self, expr):
        """Set the expression on this logical statement."""

        if expr is None:
            self._body = LogicalConstant(True)
            return

        expr_type = type(expr)
        if expr_type in native_types and expr_type not in native_logical_types:
            msg = (
                "LogicalStatment '%s' does not have a proper value. "
                "Found '%s'.\n"
                "Expecting a logical expression or Boolean value. Examples:"
                "\n   (m.Y1 & m.Y2).implies(m.Y3)"
                "\n   AtLeast(1, m.Y1, m.Y2)"
            )
            raise ValueError(msg)

        self._body = as_logical(expr)

    def get_value(self):
        """Get the expression on this logical statement."""
        return self._body


@ModelComponentFactory.register("General logical statements.")
class LogicalStatement(ActiveIndexedComponent):
    """
    This modeling component defines a logical statement using a
    rule function.

    Constructor arguments:
        expr
            A Pyomo expression for this logical statement
        rule
            A function that is used to construct logical statements
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
           The rule used to initialize the logical statement(s)

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

    _ComponentDataClass = _GeneralLogicalStatementData
    NoConstraint = (1000,)
    Skip = (1000,)
    Infeasible = (1001,)
    Violated = (1001,)
    Feasible = (1002,)
    Satisfied = (1002,)

    def __new__(cls, *args, **kwds):
        if cls != LogicalStatement:
            return super(LogicalStatement, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args) == 1):
            return SimpleLogicalStatement.__new__(SimpleLogicalStatement)
        else:
            return IndexedLogicalStatement.__new__(IndexedLogicalStatement)

    def __init__(self, *args, **kwargs):
        self.rule = kwargs.pop('rule', None)
        self._init_expr = kwargs.pop('expr', None)
        kwargs.setdefault('ctype', LogicalStatement)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

    #
    # TODO: Ideally we would not override these methods and instead add
    # the contents of _check_skip_add to the set_value() method.
    # Unfortunately, until IndexedComponentData objects know their own
    # index, determining the index is a *very* expensive operation.  If
    # we refactor things so that the Data objects have their own index,
    # then we can remove these overloads.
    #

    def _setitem_impl(self, index, obj, value):
        if self._check_skip_add(index, value) is None:
            del self[index]
            return None
        else:
            obj.set_value(value)
            return obj

    def _setitem_when_not_present(self, index, value):
        if self._check_skip_add(index, value) is None:
            return None
        else:
            return super(LogicalStatement, self)._setitem_when_not_present(
                index=index, value=value)

    def construct(self, data=None):
        """
        Construct the expression(s) for this logical statement.
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing logical statement %s" % self.name)
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        self._constructed = True

        _init_expr = self._init_expr
        _init_rule = self.rule
        #
        # We no longer need these
        #
        self._init_expr = None
        # Utilities like DAE assume this stays around
        # self.rule = None

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
                        "logical statement %s:\n%s: %s"
                        % (self.name,
                           type(err).__name__,
                           err))
                    raise
            self._setitem_when_not_present(None, tmp)

        else:
            if _init_expr is not None:
                raise IndexError(
                    "LogicalStatement '%s': Cannot initialize multiple indices "
                    "of a logical statement with a single expression" %
                    (self.name,))

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
                        "logical statement %s with index %s:\n%s: %s"
                        % (self.name,
                           str(ndx),
                           type(err).__name__,
                           err))
                    raise
                self._setitem_when_not_present(ndx, tmp)
        timer.report()

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Size", len(self)),
             ("Index", self._index if self.is_indexed() else None),
             ("Active", self.active),
             ],
            iteritems(self),
            ("Body", "Active"),
            lambda k, v: [v.body, v.active, ]
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
        tabular_writer(ostream, prefix + tab,
                       ((k, v) for k, v in iteritems(self._data) if v.active),
                       ("Body",),
                       lambda k, v: [v.body(), ])

    #
    # Checks flags like Constraint.Skip, etc. before actually creating a
    # constraint object. Returns the _ConstraintData object when it should be
    #  added to the _data dict; otherwise, None is returned or an exception
    # is raised.
    #
    def _check_skip_add(self, index, expr):
        _expr_type = expr.__class__
        if expr is None:
            raise ValueError(
                _rule_returned_none_error %
                (_get_indexed_component_data_name(self, index),))

        if expr is True:
            return None
        if expr is False:
            raise ValueError(
                "LogicalStatement '%s' is always False"
                % (_get_indexed_component_data_name(self, index),))

        return expr


class SimpleLogicalStatement(_GeneralLogicalStatementData, LogicalStatement):
    """
    SimpleLogicalStatement is the implementation representing a single,
    non-indexed logical statement.
    """

    def __init__(self, *args, **kwds):
        _GeneralLogicalStatementData.__init__(
            self, component=self, expr=None)
        LogicalStatement.__init__(self, *args, **kwds)

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
        """Access the body of a logical statement."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the body of SimpleLogicalStatement "
                    "'%s' before the LogicalStatement has been assigned "
                    "an expression. There is currently "
                    "nothing to access." % self.name)
            return _GeneralLogicalStatementData.body.fget(self)
        raise ValueError(
            "Accessing the body of logical statement '%s' "
            "before the LogicalStatement has been constructed (there "
            "is currently no value to return)."
            % self.name)

    #
    # Singleton logical statements are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # True are managed. But after that they will behave
    # like _LogicalStatementData objects where set_value expects
    # a valid expression or None.
    #

    def set_value(self, expr):
        """Set the expression on this logical statement."""
        if not self._constructed:
            raise ValueError(
                "Setting the value of logical statement '%s' "
                "before the LogicalStatement has been constructed (there "
                "is currently no object to set)."
                % self.name)

        if len(self._data) == 0:
            self._data[None] = self
        if self._check_skip_add(None, expr) is None:
            del self[None]
            return None
        return super(SimpleLogicalStatement, self).set_value(expr)

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add a logical statement with a given index."""
        if index is not None:
            raise ValueError(
                "SimpleLogicalStatement object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.name, index))
        self.set_value(expr)
        return self


class IndexedLogicalStatement(LogicalStatement):

    #
    # Leaving this method for backward compatibility reasons
    #
    # Note: Beginning after Pyomo 5.2 this method will now validate that
    # the index is in the underlying index set (through 5.2 the index
    # was not checked).
    #
    def add(self, index, expr):
        """Add a logical statement with a given index."""
        return self.__setitem__(index, expr)


@ModelComponentFactory.register("A list of logical statements.")
class LogicalStatementList(IndexedLogicalStatement):
    """
    A logical statement component that represents a list of constraints.
    Constraints can be indexed by their index, but when they are
    added an index value is not specified.
    """

    End = (1003,)

    def __init__(self, **kwargs):
        """Constructor"""
        args = (Set(),)
        if 'expr' in kwargs:
            raise ValueError(
                "LogicalStatementList does not accept the 'expr' keyword")
        LogicalStatement.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this logical statement.
        """
        generate_debug_messages = \
            __debug__ and logger.isEnabledFor(logging.DEBUG)
        if generate_debug_messages:
            logger.debug("Constructing logical statement list %s"
                         % self.name)

        if self._constructed:
            return
        self._constructed = True

        assert self._init_expr is None
        _init_rule = self.rule

        #
        # We no longer need these
        #
        self._init_expr = None
        # Utilities like DAE assume this stays around
        # self.rule = None

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
                val = len(self._index) + 1
                if generate_debug_messages:
                    logger.debug(
                        "   Constructing logical statement index " + str(val))
                expr = apply_indexed_rule(self,
                                          _init_rule,
                                          _self_parent,
                                          val)
                if expr is None:
                    raise ValueError(
                        "LogicalStatementList '%s': rule returned None "
                        "instead of LogicalStatementList.End" % (self.name,))
                if (expr.__class__ is tuple) and \
                        (expr == LogicalStatementList.End):
                    return
                self.add(expr)

        else:

            for expr in _generator:
                if expr is None:
                    raise ValueError(
                        "LogicalStatementList '%s': generator returned None "
                        "instead of LogicalStatementList.End" % (self.name,))
                if (expr.__class__ is tuple) and \
                        (expr == LogicalStatementList.End):
                    return
                self.add(expr)

    def add(self, expr):
        """Add a logical statement with an implicit index."""
        next_idx = len(self._index) + 1
        self._index.add(next_idx)
        return self.__setitem__(next_idx, expr)

