#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['Expression', '_ExpressionData']

import sys
import logging
from weakref import ref as weakref_ref
from typing import overload

from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecated, RenamedClass
from pyomo.common.modeling import NOTSET
from pyomo.common.formatting import tabular_writer
from pyomo.common.timing import ConstructionTimer

from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.indexed_component import (
    IndexedComponent,
    UnindexedComponent_set, )
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.numvalue import (NumericValue,
                                      as_numeric)
from pyomo.core.base.initializer import Initializer

logger = logging.getLogger('pyomo.core')


class _ExpressionData(NumericValue):
    """
    An object that defines a named expression.

    Public Class Attributes
        expr       The expression owned by this data.
    """

    __slots__ = ()

    #
    # Interface
    #

    def __call__(self, exception=True):
        """Compute the value of this expression."""
        if self.expr is None:
            return None
        return self.expr(exception=exception)

    def is_named_expression_type(self):
        """A boolean indicating whether this in a named expression."""
        return True

    def is_expression_type(self):
        """A boolean indicating whether this in an expression."""
        return True

    def arg(self, index):
        if index < 0 or index >= 1:
            raise KeyError("Invalid index for expression argument: %d" % index)
        return self.expr

    @property
    def _args_(self):
        return (self.expr,)

    @property
    def args(self):
        return (self.expr,)

    def nargs(self):
        return 1

    def _precedence(self):
        return 0

    def _associativity(self):
        return 0

    def _to_string(self, values, verbose, smap, compute_values):
        if verbose:
            return "%s{%s}" % (str(self), values[0])
        if self.expr is None:
            return "%s{None}" % str(self)
        return values[0]

    def clone(self):
        """Return a clone of this expression (no-op)."""
        return self

    def _apply_operation(self, result):
        # This "expression" is a no-op wrapper, so just return the inner
        # result
        return result[0]

    def polynomial_degree(self):
        """A tuple of subexpressions involved in this expressions operation."""
        return self.expr.polynomial_degree()

    def _compute_polynomial_degree(self, result):
        return result[0]

    def _is_fixed(self, values):
        return values[0]

    #
    # Abstract Interface
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        raise NotImplementedError

    def set_value(self, expr):
        """Set the expression on this expression."""
        raise NotImplementedError

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        raise NotImplementedError

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        raise NotImplementedError

    # _ExpressionData should never return False because
    # they can store subexpressions that contain variables
    def is_potentially_variable(self):
        return True


class _GeneralExpressionDataImpl(_ExpressionData):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        expr        The Pyomo expression stored in this expression.
        component   The Expression object that owns this data.

    Public Class Attributes
        expr       The expression owned by this data.
    """

    __pickle_slots__ = ('_expr',)

    __slots__ = ()

    def __init__(self, expr=None):
        self._expr = as_numeric(expr) if (expr is not None) else None

    def create_node_with_local_data(self, values):
        """
        Construct a simple expression after constructing the
        contained expression.

        This class provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.
        """
        obj = ScalarExpression()
        obj.construct()
        obj.expr = values[0]
        return obj

    def __getstate__(self):
        state = super(_GeneralExpressionDataImpl, self).__getstate__()
        for i in _GeneralExpressionDataImpl.__pickle_slots__:
            state[i] = getattr(self, i)
        return state

    # Note: because NONE of the slots on this class need to be edited,
    #       we don't need to implement a specialized __setstate__
    #       method.

    #
    # Abstract Interface
    #

    @property
    def expr(self):
        """Return expression on this expression."""
        return self._expr
    
    @expr.setter
    def expr(self, expr):
        self.set_value(expr)

    def set_value(self, expr):
        """Set the expression on this expression."""
        self._expr = as_numeric(expr) if (expr is not None) else None

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        # The underlying expression can always be changed
        # so this should never evaluate as constant
        return False

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        return self._expr.is_fixed()

class _GeneralExpressionData(_GeneralExpressionDataImpl,
                             ComponentData):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        expr        The Pyomo expression stored in this expression.
        component   The Expression object that owns this data.

    Public Class Attributes
        expr        The expression owned by this data.

    Private class attributes:
        _component  The expression component.
    """

    __slots__ = _GeneralExpressionDataImpl.__pickle_slots__

    def __init__(self, expr=None, component=None):
        _GeneralExpressionDataImpl.__init__(self, expr)
        # Inlining ComponentData.__init__
        self._component = weakref_ref(component) if (component is not None) \
                          else None


@ModelComponentFactory.register(
    "Named expressions that can be used in other expressions.")
class Expression(IndexedComponent):
    """
    A shared expression container, which may be defined over a index.

    Constructor Arguments:
        initialize  A Pyomo expression or dictionary of expressions
                        used to initialize this object.
        expr        A synonym for initialize.
        rule        A rule function used to initialize this object.
        name        Name for this component.
        doc         Text describing this component.
    """

    _ComponentDataClass = _GeneralExpressionData
    # This seems like a copy-paste error, and should be renamed/removed
    NoConstraint = IndexedComponent.Skip

    def __new__(cls, *args, **kwds):
        if cls != Expression:
            return super(Expression, cls).__new__(cls)
        if not args or (args[0] is UnindexedComponent_set and len(args)==1):
            return ScalarExpression.__new__(ScalarExpression)
        else:
            return IndexedExpression.__new__(IndexedExpression)

    @overload
    def __init__(self, *indexes, rule=None, expr=None, initialize=None,
                 name=None, doc=None): ...

    def __init__(self, *args, **kwds):
        _init = self._pop_from_kwargs(
            'Expression', kwds, ('rule', 'expr', 'initialize'), None)
        # Historically, Expression objects were dense (but None):
        # setting arg_not_specified causes Initializer to recognize
        # _init==None as a constant initializer returning None
        #
        # To initialize a completely empty Expression, pass either
        # initialize={} (to require explicit setitem before a getitem),
        # or initialize=NOTSET (to allow getitem before setitem)
        self._rule = Initializer(_init, arg_not_specified=NOTSET)

        kwds.setdefault('ctype', Expression)
        IndexedComponent.__init__(self, *args, **kwds)

    def _pprint(self):
        return (
            [('Size', len(self)),
             ('Index', None if (not self.is_indexed())
                  else self._index)
             ],
            self.items(),
            ("Expression",),
            lambda k,v: \
               ["Undefined" if v.expr is None else v.expr]
            )

    def display(self, prefix="", ostream=None):
        """TODO"""
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab="    "
        ostream.write(prefix+self.local_name+" : ")
        ostream.write("Size="+str(len(self)))

        ostream.write("\n")
        tabular_writer(
            ostream,
            prefix+tab,
            ((k,v) for k,v in self._data.items()),
            ( "Value", ),
            lambda k, v: \
               ["Undefined" if v.expr is None else v()])

    #
    # A utility to extract all index-value pairs defining this
    # expression, returned as a dictionary. useful in many contexts,
    # in which key iteration and repeated __getitem__ calls are too
    # expensive to extract the contents of an expression.
    #
    def extract_values(self):
        return {key:expression_data.expr
                for key, expression_data in self.items()}

    #
    # takes as input a (index, value) dictionary for updating this
    # Expression.  if check=True, then both the index and value are
    # checked through the __getitem__ method of this class.
    #
    def store_values(self, new_values):

        if (self.is_indexed() is False) and \
           (not None in new_values):
            raise KeyError(
                "Cannot store value for scalar Expression"
                "="+self.name+"; no value with index "
                "None in input new values map.")

        for index, new_value in new_values.items():
            self._data[index].set_value(new_value)

    def _getitem_when_not_present(self, idx):
        if self._rule is None:
            _init = None
            # TBD: Is this desired behavior?  I can see implicitly setting
            # an Expression if it was not originally defined, but I am less
            # convinced that implicitly creating an Expression (like what
            # works with a Var) makes sense.  [JDS 25 Nov 17]
            #raise KeyError(idx)
        else:
            _init = self._rule(self.parent_block(), idx)
            if _init is Expression.Skip:
                raise KeyError(idx)
        return self._setitem_when_not_present(idx, _init)

    def construct(self, data=None):
        """ Apply the rule to construct values in this set """
        if self._constructed:
            return
        self._constructed = True

        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug(
                "Constructing Expression, name=%s, from data=%s"
                % (self.name, str(data)))

        try:
            # We do not (currently) accept data for constructing Constraints
            assert data is None
            self._construct_from_rule_using_setitem()
        finally:
            timer.report()


class ScalarExpression(_GeneralExpressionData, Expression):

    def __init__(self, *args, **kwds):
        _GeneralExpressionData.__init__(self, expr=None, component=self)
        Expression.__init__(self, *args, **kwds)

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
        """Return expression on this expression."""
        if self._constructed:
            return _GeneralExpressionData.expr.fget(self)
        raise ValueError(
            "Accessing the expression of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)."
            % (self.name))
    @expr.setter
    def expr(self, expr):
        """Set the expression on this expression."""
        self.set_value(expr)

    def clear(self):
        self._data = {}

    def set_value(self, expr):
        """Set the expression on this expression."""
        if self._constructed:
            return _GeneralExpressionData.set_value(self, expr)
        raise ValueError(
            "Setting the expression of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no object to set)."
            % (self.name))

    def is_constant(self):
        """A boolean indicating whether this expression is constant."""
        if self._constructed:
            return _GeneralExpressionData.is_constant(self)
        raise ValueError(
            "Accessing the is_constant flag of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        if self._constructed:
            return _GeneralExpressionData.is_fixed(self)
        raise ValueError(
            "Accessing the is_fixed flag of expression '%s' "
            "before the Expression has been constructed (there "
            "is currently no value to return)."
            % (self.name))

    #
    # Leaving this method for backward compatibility reasons.
    # (probably should be removed)
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if index is not None:
            raise KeyError(
                "ScalarExpression object '%s' does not accept "
                "index values other than None. Invalid value: %s"
                % (self.name, index))
        if (type(expr) is tuple) and \
           (expr == Expression.Skip):
            raise ValueError(
                "Expression.Skip can not be assigned "
                "to an Expression that is not indexed: %s"
                % (self.name))
        self.set_value(expr)
        return self


class SimpleExpression(metaclass=RenamedClass):
    __renamed__new_class__ = ScalarExpression
    __renamed__version__ = '6.0'


class IndexedExpression(Expression):

    #
    # Leaving this method for backward compatibility reasons
    # Note: It allows adding members outside of self._index.
    #       This has always been the case. Not sure there is
    #       any reason to maintain a reference to a separate
    #       index set if we allow this.
    #
    def add(self, index, expr):
        """Add an expression with a given index."""
        if (type(expr) is tuple) and \
           (expr == Expression.Skip):
            return None
        cdata = _GeneralExpressionData(expr, component=self)
        self._data[index] = cdata
        return cdata

