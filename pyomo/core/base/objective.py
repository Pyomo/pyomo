#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['Objective', 'simple_objective_rule', '_ObjectiveData', 'minimize', 'maximize', 'simple_objectivelist_rule', 'ObjectiveList']

import sys
import logging
from weakref import ref as weakref_ref
import inspect

from six import iteritems

import pyutilib.math
import pyutilib.misc

from pyomo.core.base.numvalue import NumericValue, as_numeric, value, native_numeric_types
from pyomo.core.base.expr import _ExpressionBase
from pyomo.core.base.component import ActiveComponentData, register_component
from pyomo.core.base.sparse_indexed_component import ActiveSparseIndexedComponent, UnindexedComponent_set
from pyomo.core.base.misc import apply_indexed_rule, tabular_writer
from pyomo.core.base.util import is_functor
from pyomo.core.base.sets import Set
from pyomo.core.base.var import _VarData
from pyomo.core.base.set_types import Reals

logger = logging.getLogger('pyomo.core')

_simple_objective_rule_types = set([ type(None), bool ])


# Constants used to define the optimization sense
minimize=1
maximize=-1


def simple_objective_rule( fn ):
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

    def wrapper_function ( *args, **kwargs ):
        #
        # If the function is None, then skip this objective.
        #
        if fn is None:
            return Objective.Skip
        #
        # Otherwise, the argument is a functor, so call it to generate the 
        # objective expression.
        #
        value = fn( *args, **kwargs )
        if value is None:
            return Objective.Skip
        return value
    return wrapper_function

def simple_objectivelist_rule( fn ):
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
    def wrapper_function ( *args, **kwargs ):
        #
        # If the function is None, then the list is finished.
        #
        if fn is None:
            return ObjectiveList.End
        #
        # Otherwise, the argument is a functor, so call it to generate the 
        # objective expression.
        #
        value = fn( *args, **kwargs )
        if value is None:
            return ObjectiveList.End
        return value
    return wrapper_function


class _ObjectiveData(ActiveComponentData, NumericValue):
    """
    This class defines the data for a single objective.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        component       The Objective object that owns this data.
        expr            The expression for this objective.

    Public class attributes:
        active          A boolean that is true if this objective is active 
                            in the model.
        expr            The Pyomo expression for this objective
        value           The numeric value of the expression

    Private class attributes:
        _component      The objective component.
    """

    __pickle_slots__ = ( 'value', 'expr')
    __slots__ = __pickle_slots__ + ( '__weakref__', )

    def __init__(self, component, expr):
        # the following lines represent in-lining of the ActiveComponentData
        # and ComponentData constructors.
        self._component = weakref_ref(component)
        self._active = True
        #
        self.value = None
        self.expr = expr

    def __getstate__(self):
        """
        This method must be defined because this class uses slots.
        """
        state = super(_ObjectiveData, self).__getstate__()
        for i in _ObjectiveData.__pickle_slots__:
            state[i] = getattr(self,i)
        return state

    # Note: because NONE of the slots on this class need to be edited,
    # we don't need to implement a specialized __setstate__ method.

    def __call__(self, exception=True):
        """
        Compute the value of this objective.

	    This method does not simply return self.value because that
	    data value may be out of date w.r.t. the value of decision
	    variables.
        """
        if self.expr is None:
            return None
        return self.expr(exception=exception)

    def polynomial_degree(self):
        """
        Return the polynomial degree of the objective expression.
        """
        if self.expr is None:
            return None
        return self.expr.polynomial_degree()

    def is_minimizing ( self ):
        """
        Return True if this is a minimization objective
        """
        return self.parent_component().sense == minimize


class Objective(ActiveSparseIndexedComponent):
    """
    This modeling component defines an objective expression.

    Note that this is a subclass of NumericValue to allow
    objectives to be used as part of expressions.

    Constructor arguments:
        expr            A Pyomo expression for this objective
        noruleinit      Indicate that its OK that no initialization is
                            specified
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
        trivial         This boolean is True if all objective indices have 
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

    NoObjective = (1000,)
    Skip        = (1000,)

    def __new__(cls, *args, **kwds):
        if cls != Objective:
            return super(Objective, cls).__new__(cls)
        if args == ():
            return SimpleObjective.__new__(SimpleObjective)
        else:
            return IndexedObjective.__new__(IndexedObjective)

    def __init__(self, *args, **kwargs):
        self.sense = kwargs.pop('sense', minimize )
        self.rule  = kwargs.pop('rule', None )
        self._expr  = kwargs.pop('expr', None )
        self._no_rule_init = kwargs.pop('noruleinit', False )
        #
        # This value is set when creating a canonical representation.
        #
        self.trivial = False
        #
        kwargs.setdefault('ctype', Objective)
        ActiveSparseIndexedComponent.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing objective %s", self.cname(True))
        if self._constructed:
            return
        _self_rule = self.rule
        if self._no_rule_init and (_self_rule is not None):
            logger.warning("The noruleinit keyword is being used in conjunction " \
                  "with the rule keyword for objective '%s'; defaulting to " \
                  "rule-based construction" % self.cname(True))
        if _self_rule is None and self._expr is None:
            if not self._no_rule_init:
                logger.warn("No construction rule or expression specified for "
                            "objective '%s'", self.cname(True))
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
                        "objective %s:\n%s: %s"
                        % ( self.cname(True), type(err).__name__, err ) )
                    raise
                if tmp is None:
                    raise ValueError("Objective rule returned None instead of Objective.Skip")
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
                    raise ValueError("Objective rule returned None instead of Objective.Skip for index %s" % str(val))
                self.add(val, tmp)

    def add(self, index, expr):
        """
        Add an objective with a specified index.
        """
        _expr_type = expr.__class__
        if _expr_type is tuple:
            #
            # Ignore an 'empty' objective
            #
            if expr == Objective.Skip:
                return

        # Always store the objective expression
        # as an instance of NumericValue (it might
        # be a float when passed in)
        expr = as_numeric(expr)

        #
        # Add the objective data
        #
        if index is None:
            self._data[index] = self
            self.expr = expr
        else:
            self._data[index] = _ObjectiveData(self, expr)

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return ( 
            [("Size", len(self)),
             ("Index", self._index \
                       if self._index != UnindexedComponent_set else None),
             ("Active", self.active),
             ("Sense", "minimize" if self.sense == minimize else "maximize"),
             ],
            iteritems(self._data),
            ( "Key","Active","Expression" ),
            lambda k, v: [ k,
                           v.active,
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
        ostream.write(prefix+self.cname()+" : ") 
        ostream.write(", ".join("%s=%s" % (k,v) for k,v in [
                    ("Size", len(self)),
                    ("Index", self._index \
                         if self._index != UnindexedComponent_set else None),
                    ("Active", self.active),
                    ] ))
        
        ostream.write("\n") 
        tabular_writer( ostream, prefix+tab,  
                        ((k,v) for k,v in iteritems(self._data) if v.active), 
                        ( "Key","Active","Value" ), 
                        lambda k, v: [ k, v.active, value(v), ] ) 


class SimpleObjective(Objective, _ObjectiveData):
    """
    SimpleObjective is the implementation representing a single,
    non-indexed objective.
    """

    def __init__(self, *args, **kwd):
        _ObjectiveData.__init__(self, self, None)
        Objective.__init__(self, *args, **kwd)

    def __call__(self, exception=True):
        """
        Compute the value of the objective expression
        """
        if self._constructed:
            return _ObjectiveData.__call__(self, exception=exception)
        if exception:
            raise ValueError( """Evaluating the numeric value of objective '%s' before the Objective has been
            constructed (there is currently no value to return).""" % self.cname(True) )


    #
    # Since this class derives from Component and Component.__getstate__
    # just packs up the entire __dict__ into the state dict, we do not
    # need to define the __getstate__ or __setstate__ methods.
    # We just defer to the super() get/set state.  Since all of our 
    # get/set state methods rely on super() to traverse the MRO, this 
    # will automatically pick up both the Component and Data base classes.
    #


class IndexedObjective(Objective):

    def __call__(self, exception=True):
        """Compute the value of the objective body"""
        if exception:
            msg = 'Cannot compute the value of an array of objectives'
            raise TypeError(msg)

    def is_minimizing ( self ):
        """Return true if this is a minimization objective"""
        return self.sense == minimize


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
        Objective.__init__(self, *args, **kwargs)

    def construct(self, data=None):
        """
        Construct the expression(s) for this objective.
        """
        generate_debug_messages = __debug__ and logger.isEnabledFor(logging.DEBUG)
        if generate_debug_messages:
            logger.debug("Constructing objective %s", self.cname(True))
        if self._constructed:
            return
        _self_rule = self.rule
        if self._no_rule_init and (_self_rule is not None):
            logger.warning("The noruleinit keyword is being used in conjunction " \
                  "with the rule keyword for objective '%s'; defaulting to " \
                  "rule-based construction" % self.cname(True))
        self._constructed=True
        if _self_rule is None:
            if not self._no_rule_init:
                logger.warn("No construction rule or expression specified for "
                            "objective '%s'", self.cname(True))
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
                val = self._nobjectives + 1
                if generate_debug_messages:
                    logger.debug("   Constructing objective index "+str(val))
                expr = apply_indexed_rule( self, _self_rule, _self_parent, val )
                if expr is None:
                    raise ValueError( "Objective rule returned None "
                                      "instead of ObjectiveList.End" )
                if (expr.__class__ is tuple and expr == ObjectiveList.End):
                    return
                self.add(expr)
        else:
            for expr in _generator:
                if expr is None:
                    raise ValueError( "Objective generator returned None "
                                      "instead of ObjectiveList.End" )
                if (expr.__class__ is tuple and expr == ObjectiveList.End):
                    return
                self.add(expr)

    def add(self, expr):
        """
        Add an objective to the list.
        """
        self._nobjectives += 1
        Objective.add(self, self._nobjectives, expr)


register_component(Objective, 'Expressions that are minimized or maximized.')
register_component(ObjectiveList, "A list of objectives.")
