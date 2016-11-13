#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ('Objective',
           'simple_objective_rule',
           '_ObjectiveData',
           'minimize',
           'maximize',
           'simple_objectivelist_rule',
           'ObjectiveList')

import sys
import logging
from weakref import ref as weakref_ref
import inspect

from pyomo.core.base.numvalue import as_numeric, value
from pyomo.core.base.component import (ActiveComponentData,
                                       register_component)
from pyomo.core.base.indexed_component import (ActiveIndexedComponent,
                                               UnindexedComponent_set)
from pyomo.core.base.expression import (_ExpressionData,
                                        _GeneralExpressionDataImpl)
from pyomo.core.base.misc import apply_indexed_rule, tabular_writer
from pyomo.core.base.sets import Set

from six import iteritems

logger = logging.getLogger('pyomo.core')

# Constants used to define the optimization sense
minimize=1
maximize=-1

def simple_objective_rule(fn):
    """
    This is a decorator that translates None into Objective.Skip.
    This supports a simpler syntax in objective rules, though these
    can be more difficult to debug when errors occur.

    Example use:

    @simple_objective_rule
    def O_rule(model, i, j):
        ...

    model.o = Objective(rule=simple_objective_rule(...))
    """

    def wrapper_function (*args, **kwargs):
        #
        # If the function is None, then skip this objective.
        #
        if fn is None:
            return Objective.Skip
        #
        # Otherwise, the argument is a functor, so call it to generate
        # the objective expression.
        #
        value = fn(*args, **kwargs)
        if value is None:
            return Objective.Skip
        return value
    return wrapper_function

def simple_objectivelist_rule(fn):
    """

    This is a decorator that translates None into ObjectiveList.End.
    This supports a simpler syntax in objective rules, though these
    can be more difficult to debug when errors occur.

    Example use:

    @simple_objectivelist_rule
    def O_rule(model, i, j):
        ...

    model.o = ObjectiveList(expr=simple_objectivelist_rule(...))
    """
    def wrapper_function (*args, **kwargs):
        #
        # If the function is None, then the list is finished.
        #
        if fn is None:
            return ObjectiveList.End
        #
        # Otherwise, the argument is a functor, so call it to generate
        # the objective expression.
        #
        value = fn(*args, **kwargs)
        if value is None:
            return ObjectiveList.End
        return value
    return wrapper_function

#
# This class is a pure interface
#

class _ObjectiveData(_ExpressionData):
    """
    This class defines the data for a single objective.

    Public class attributes:
        expr            The Pyomo expression for this objective
        sense           The direction for this objective.
    """

    __slots__ = ()

    #
    # Interface
    #

    def is_minimizing(self):
        """Return True if this is a minimization objective."""
        return self.sense == minimize

    #
    # Abstract Interface
    #

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        raise NotImplementedError

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        raise NotImplementedError

class _GeneralObjectiveData(_GeneralExpressionDataImpl,
                            _ObjectiveData,
                            ActiveComponentData):
    """
    This class defines the data for a single objective.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        expr            The Pyomo expression stored in this objective.
        sense           The direction for this objective.
        component       The Objective object that owns this data.

    Public class attributes:
        expr            The Pyomo expression for this objective
        active          A boolean that is true if this objective is active
                            in the model.
        sense           The direction for this objective.

    Private class attributes:
        _component      The objective component.
        _active         A boolean that indicates whether this data is active
    """

    __pickle_slots__ = ("_sense",)
    __slots__ = __pickle_slots__ + \
                _GeneralExpressionDataImpl.__expression_slots__

    def __init__(self, expr, sense=minimize, component=None):
        _GeneralExpressionDataImpl.__init__(self, expr)
        # Inlining ActiveComponentData.__init__
        self._component = weakref_ref(component) if (component is not None) \
                          else None
        self._active = True
        self._sense = sense

        if (self._sense != minimize) and \
           (self._sense != maximize):
            raise ValueError("Objective sense must be set to one of "
                             "'minimize' (%s) or 'maximize' (%s). Invalid "
                             "value: %s'" % (minimize, maximize, sense))

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = _GeneralExpressionDataImpl.__getstate__(self)
        for i in _GeneralObjectiveData.__pickle_slots__:
            state[i] = getattr(self,i)
        return state

    # Note: because NONE of the slots on this class need to be edited,
    #       we don't need to implement a specialized __setstate__
    #       method.

    #
    # Abstract Interface
    #

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        return self._sense
    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        if (sense == minimize) or \
           (sense == maximize):
            self._sense = sense
        else:
            raise ValueError("Objective sense must be set to one of "
                             "'minimize' (%s) or 'maximize' (%s). Invalid "
                             "value: %s'" % (minimize, maximize, sense))

class Objective(ActiveIndexedComponent):
    """
    This modeling component defines an objective expression.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        expr            A Pyomo expression for this objective
        rule            A function that is used to construct objective
                            expressions
        sense           Indicate whether minimizing (the default) or maximizing
        doc             A text string describing this component
        name            A name for this component

    Public class attributes:
        doc             A text string describing this component
        name            A name for this component
        active          A boolean that is true if this component will be
                            used to construct a model instance
        rule            The rule used to initialize the objective(s)
        sense           The objective sense

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

    NoObjective = (1000,)
    Skip        = (1000,)

    def __new__(cls, *args, **kwds):
        if cls != Objective:
            return super(Objective, cls).__new__(cls)
        if args == () or (type(args[0]) == set and args[0] == UnindexedComponent_set and len(args)==1):
            return SimpleObjective.__new__(SimpleObjective)
        else:
            return IndexedObjective.__new__(IndexedObjective)

    def __init__(self, *args, **kwargs):
        self._init_sense = kwargs.pop('sense', minimize)
        self.rule  = kwargs.pop('rule', None)
        self._init_expr  = kwargs.pop('expr', None)
        kwargs.setdefault('ctype', Objective)
        ActiveIndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Constructing objective %s" % (self.name))
        if self._constructed:
            return
        self._constructed = True

        _init_expr = self._init_expr
        _init_sense = self._init_sense
        _init_rule = self.rule
        #
        # We no longer need these
        #
        self._init_expr = None
        self._init_sense = None
        # Utilities like DAE assume this stays around
        #self.rule = None

        if (_init_rule is None) and \
           (_init_expr is None):
            # No construction rule or expression specified.
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
                        "objective %s:\n%s: %s"
                        % (self.name,
                           type(err).__name__,
                           err))
                    raise
                if tmp is None:
                    raise ValueError(
                        "Objective rule returned None instead of "
                        "Objective.Skip")

            assert None not in self._data
            cdata = self._check_skip_add(None, tmp, objdata=self)
            if cdata is not None:
                # this happens as a side-effect of set_value on
                # SimpleObjective (normally _check_skip_add does not
                # add anything to the _data dict but it does call
                # set_value on the objdata object we pass in)
                assert None in self._data
                cdata.set_sense(_init_sense)
            else:
                assert None not in self._data

        else:

            if not _init_expr is None:
                raise IndexError(
                    "Cannot initialize multiple indices of an "
                    "objective with a single expression")
            for ndx in self._index:
                try:
                    tmp = apply_indexed_rule(self,
                                             _init_rule,
                                             _self_parent,
                                             ndx)
                except Exception:
                    err = sys.exc_info()[1]
                    logger.error(
                        "Rule failed when generating expression for"
                        " objective %s with index %s:\n%s: %s"
                        % (self.name,
                           str(ndx),
                           type(err).__name__,
                           err))
                    raise
                if tmp is None:
                    raise ValueError(
                        "Objective rule returned None instead of "
                        "Objective.Skip for index %s" % (str(ndx)))

                cdata = self._check_skip_add(ndx, tmp)
                if cdata is not None:
                    cdata.set_sense(_init_sense)
                    self._data[ndx] = cdata

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return (
            [("Size", len(self)),
             ("Index", self._index \
                       if self._index != UnindexedComponent_set else None),
             ("Active", self.active)
             ],
            iteritems(self._data),
            ( "Active","Sense","Expression"),
            lambda k, v: [ v.active,
                           ("minimize" if (v.sense == minimize) else "maximize"),
                           v.expr
                           ]
            )

    def display(self, prefix="", ostream=None):
        """Provide a verbose display of this object"""
        if not self.active:
            return
        tab = "    "
        if ostream is None:
            ostream = sys.stdout
        ostream.write(prefix+self.local_name+" : ")
        ostream.write(", ".join("%s=%s" % (k,v) for k,v in [
                    ("Size", len(self)),
                    ("Index", self._index \
                     if self._index != UnindexedComponent_set else None),
                    ("Active", self.active),
                    ] ))

        ostream.write("\n")
        tabular_writer( ostream, prefix+tab,
                        ((k,v) for k,v in iteritems(self._data) if v.active),
                        ( "Active","Value" ),
                        lambda k, v: [ v.active, value(v), ] )

    #
    # Checks flags like Objective.Skip, etc. before
    # actually creating an objective object. Optionally
    # pass in the _ObjectiveData object to set the value
    # on. Only returns the _ObjectiveData object when it
    # should be added to the _data dict; otherwise, None
    # is returned or an exception is raised.
    #
    def _check_skip_add(self, index, expr, objdata=None):

        #
        # Adds a dummy objective object to the _data
        # dict just before an error message is generated
        # so that we can generate a fully qualified name
        #
        def _prep_for_error():
            if objdata is None:
                self._data[index] = _GeneralObjectiveData(None,
                                                          component=self)
            else:
                self._data[index] = objdata

        _expr_type = expr.__class__
        #
        # Convert deprecated expression values
        #
        if expr is None:
            _prep_for_error()
            raise ValueError(
                "Invalid objective expression. The objective "
                "expression resolved to None instead of a Pyomo "
                "object or numeric value. Please modify your rule "
                "to return Objective.Skip instead of None."
                "\n\nError thrown for Objective '%s'"
                % (self._data[index].name))

        #
        # Ignore an 'empty' objective
        #
        if _expr_type is tuple:
            if expr == Objective.Skip:
                return None

        if objdata is None:
            objdata = _GeneralObjectiveData(expr, component=self)
        else:
            objdata.set_value(expr)
            assert objdata.parent_component() is self

        return objdata

class SimpleObjective(_GeneralObjectiveData, Objective):
    """
    SimpleObjective is the implementation representing a single,
    non-indexed objective.
    """

    def __init__(self, *args, **kwd):
        _GeneralObjectiveData.__init__(self, None, component=self)
        Objective.__init__(self, *args, **kwd)

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
    def expr(self):
        """Access the expression of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the expression of SimpleObjective "
                    "'%s' before the Objective has been assigned "
                    "a sense or expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralObjectiveData.expr.fget(self)
        raise ValueError(
            "Accessing the expression of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no value to return)."
            % (self.name))
    @expr.setter
    def expr(self, expr):
        """Set the expression of this objective."""
        self.set_value(expr)

    # for backwards compatibility reasons
    @property
    def value(self):
        logger.warning("DEPRECATED: The .value property getter on "
                       "SimpleObjective is deprecated. Use "
                       "the .expr property getter instead")
        return self.expr
    @value.setter
    def value(self, expr):
        logger.warning("DEPRECATED: The .value property setter on "
                       "SimpleObjective is deprecated. Use the "
                       "set_value(expr) method instead")
        self.set_value(expr)

    @property
    def sense(self):
        """Access sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                raise ValueError(
                    "Accessing the sense of SimpleObjective "
                    "'%s' before the Objective has been assigned "
                    "a sense or expression. There is currently "
                    "nothing to access." % (self.name))
            return _GeneralObjectiveData.sense.fget(self)
        raise ValueError(
            "Accessing the sense of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no value to return)."
            % (self.name))
    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        self.set_sense(sense)

    #
    # Singleton objectives are strange in that we want them to be
    # both be constructed but have len() == 0 when not initialized with
    # anything (at least according to the unit tests that are
    # currently in place). So during initialization only, we will
    # treat them as "indexed" objects where things like
    # Objective.Skip are managed. But after that they will behave
    # like _ObjectiveData objects where set_value does not handle
    # Objective.Skip but expects a valid expression or None
    #

    def set_value(self, expr):
        """Set the expression of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return _GeneralObjectiveData.set_value(self, expr)
        raise ValueError(
            "Setting the value of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no object to set)."
            % (self.name))

    def set_sense(self, sense):
        """Set the sense (direction) of this objective."""
        if self._constructed:
            if len(self._data) == 0:
                self._data[None] = self
            return _GeneralObjectiveData.set_sense(self, sense)
        raise ValueError(
            "Setting the sense of objective '%s' "
            "before the Objective has been constructed (there "
            "is currently no object to set)."
            % (self.name))

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise ValueError(
                "SimpleObjective object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.name, index))
        self.set_value(expr)
        return self

class IndexedObjective(Objective):

    #
    # Leaving this method for backward compatibility reasons
    # Note: It allows adding members outside of self._index.
    #       This has always been the case. Not sure there is
    #       any reason to maintain a reference to a separate
    #       index set if we allow this.
    #
    def add(self, index, expr):
        """Add an objective with a given index."""
        cdata = self._check_skip_add(index, expr)
        if cdata is not None:
            self._data[index] = cdata
        return cdata

class ObjectiveList(IndexedObjective):
    """
    An objective component that represents a list of objectives.
    Objectives can be indexed by their index, but when they are added
    an index value is not specified.
    """

    End             = (1003,)

    def __init__(self, **kwargs):
        """Constructor"""
        args = (Set(),)
        self._nobjectives = 0
        if 'expr' in kwargs:
            raise ValueError(
                "ObjectiveList does not accept the 'expr' keyword")
        Objective.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        generate_debug_messages = \
            __debug__ and logger.isEnabledFor(logging.DEBUG)
        if generate_debug_messages:
            logger.debug(
                "Constructing objective %s" % (self.name))

        if self._constructed:
            return
        self._constructed=True

        assert self._init_expr is None
        _init_rule = self.rule
        _init_sense = self._init_sense

        #
        # We no longer need these
        #
        self._init_expr = None
        self._init_sense = None
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
                val = self._nobjectives + 1
                if generate_debug_messages:
                    logger.debug(
                        "   Constructing objective index "+str(val))
                expr = apply_indexed_rule(self,
                                          _init_rule,
                                          _self_parent,
                                          val)
                if expr is None:
                    raise ValueError(
                        "Objective rule returned None "
                        "instead of ObjectiveList.End")
                if (expr.__class__ is tuple) and \
                   (expr == ObjectiveList.End):
                    return
                self.add(expr, sense=_init_sense)

        else:

            for expr in _generator:
                if expr is None:
                    raise ValueError(
                        "Objective generator returned None "
                        "instead of ObjectiveList.End")
                if (expr.__class__ is tuple) and \
                   (expr == ObjectiveList.End):
                    return
                self.add(expr, sense=_init_sense)

    def add(self, expr, sense=minimize):
        """Add an objective to the list."""
        cdata = self._check_skip_add(self._nobjectives + 1, expr)
        self._nobjectives += 1
        self._index.add(self._nobjectives)
        if cdata is not None:
            cdata.set_sense(sense)
            self._data[self._nobjectives] = cdata
        return cdata

register_component(Objective,
                   "Expressions that are minimized or maximized.")
register_component(ObjectiveList,
                   "A list of objective expressions.")
