#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['Expression', '_ExpressionData']

import sys
import logging
from six import iteritems

from pyomo.core.base.component import ComponentData, register_component
from pyomo.core.base.indexed_component import IndexedComponent, normalize_index
from pyomo.core.base.misc import apply_indexed_rule, tabular_writer
from pyomo.core.base.numvalue import NumericValue, as_numeric
import pyomo.core.base.expr
from pyomo.core.base.util import is_functor

logger = logging.getLogger('pyomo.core')


class _ExpressionData(ComponentData, NumericValue):
    """
    An object that defines an expression that is never cloned

    Constructor Arguments
        owner       The Expression that owns this data.

    Public Class Attributes
        value       The expression owned by this data.
    """

    __slots__ = ('_value',)

    def __init__(self, owner, value):
        """Constructor"""
        ComponentData.__init__(self, owner)
        self.value = value

    def __getstate__(self):
        state = super(_ExpressionData, self).__getstate__()
        for i in _ExpressionData.__slots__:
            state[i] = getattr(self, i)
        return state

    # Note: None of  the slots on this class need to be edited, so
    # we don't need to implement a specialized __setstate__ method, and
    # can quietly rely on the super() class's implementation.

    # We make value a property so we can ensure that the value
    # assigned to this component is numeric and that all uses of
    # .value on the NumericValue base class will work
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value is not None:
            self._value = as_numeric(value)
        else:
            self._value = None

    def set_value(self, value):
        self.value = value

    @property
    def _parent_expr(self):
        return None

    @value.setter
    def _parent_expr(self, value):
        pass

    def is_constant(self):
        # The underlying expression can always be changed
        # so this should never evaluate as constant
        return False

    def is_fixed(self):
        return self._value.is_fixed()

    def __call__(self, exception=True):
        """Return the value of this object."""
        if self._value is None:
            return None
        return self._value(exception=exception)

    #################
    # Methods which mock _ExpressionBase behavior defined below
    #################

    # Note: using False here COULD potentially improve performance
    #       inside expression generation and means we wouldn't need to
    #       define a (no-op) clone method. However, there are TONS of
    #       places throughout the code base where is_expression is
    #       used to check whether one needs to "dive deeper" into the
    #       _args
    def is_expression(self):
        return True

    @property
    def _args(self):
        return (self._value,)

    def _arguments(self):
        yield self._value

    def clone(self):
        return self

    def polynomial_degree(self):
        return self._value.polynomial_degree()

    def _polynomial_degree(self, result):
        return result.pop()

    def to_string(self, ostream=None, verbose=None, precedence=0):
        if ostream is None:
            ostream = sys.stdout
        _verbose = pyomo.core.base.expr_common.TO_STRING_VERBOSE if \
            verbose is None else verbose
        if _verbose:
            ostream.write(str(self))
            ostream.write("{")
        if self._value is None:
            ostream.write("Undefined")
        else:# self._value.is_expression():
            self._value.to_string( ostream=ostream, verbose=verbose,
                                   precedence=precedence )
        #else:
        #    ostream.write(str(self._value))
        if _verbose:
            ostream.write("}")


class Expression(IndexedComponent):
    """
    A shared expression container, which may be defined over a index.

    Constructor Arguments:
        initialize  A Pyomo expression or dictionary of expressions
                        used to initialize this object.
        expr        A synonym for initialize.
        rule        A rule function used to initialize this object.
    """

    def __new__(cls, *args, **kwds):
        if cls != Expression:
            return super(Expression, cls).__new__(cls)
        if args == ():
            return SimpleExpression.__new__(SimpleExpression)
        else:
            return IndexedExpression.__new__(IndexedExpression)

    def __init__(self, *args, **kwd):
        self._init_rule = kwd.pop('rule', None)
        self._init_value = kwd.pop('initialize', None)
        self._init_value = kwd.pop('expr', self._init_value)
        if is_functor(self._init_value) and \
           (not isinstance(self._init_value, NumericValue)):
            raise TypeError("A callable type that is not a Pyomo "
                            "expression can not be used to initialize "
                            "an Expression object. Use 'rule' to initalize "
                            "with function types.")
        if (self._init_rule is not None) and \
           (self._init_value is not None):
            raise ValueError("Both a rule and an expression can not be "
                             "used to initialized an Expression object")

        kwd.setdefault('ctype', Expression)
        IndexedComponent.__init__(self, *args, **kwd)

    def _pprint(self):
        return (
            [('Size', len(self)),
             ('Index', None if (not self.is_indexed())
                  else self._index)
             ],
            self.iteritems(),
            ("Key","Expression"),
            lambda k,v: [ k,
                          "Undefined" if v.value is None else v.value
                          ]
            )

    def display(self, prefix="", ostream=None):
        """TODO"""
        if not self.active:
            return
        if ostream is None:
            ostream = sys.stdout
        tab="    "
        ostream.write(prefix+self.cname()+" : ")
        ostream.write("Size="+str(len(self)))

        ostream.write("\n")
        tabular_writer( ostream, prefix+tab,
                        ((k,v) for k,v in iteritems(self._data)),
                        ( "Key","Value" ),
                        lambda k, v: [ k,
                                       "Undefined" if v.value is None else v(),
                                       ] )

    # TODO: Not sure what "reset" really means in this context...
    def reset(self):
        pass

    #
    # A utility to extract all index-value pairs defining this
    # expression, returned as a dictionary. useful in many contexts,
    # in which key iteration and repeated __getitem__ calls are too
    # expensive to extract the contents of an expression.
    #
    def extract_values(self):
        return dict((key, expression_data.value) \
                    for key, expression_data in iteritems(self))
    #
    # takes as input a (index, value) dictionary for updating this
    # Expression.  if check=True, then both the index and value are
    # checked through the __getitem__ method of this class.
    #
    def store_values(self, new_values):

        if (self.is_indexed() is False) and (not None in new_values):
            raise RuntimeError("Cannot store value for scalar Expression"
                               "="+self.cname(True)+"; no value with index "
                               "None in input new values map.")

        for index, new_value in iteritems(new_values):
            self._data[index].value = new_value

    def __setitem__(self, ndx, val):
        #
        # Get the expression data object
        #
        if ndx in self._data:
            exprdata = self._data[ndx]
        else:
            _ndx = normalize_index(ndx)
            if _ndx in self._data:
                exprdata = self._data[_ndx]
            else:
                msg = "Cannot set the value of Expression '%s' with invalid " \
                    "index '%s'"
                raise KeyError(msg % ( self.cname(True), str(ndx) ))
        #
        # Set the value
        #
        exprdata.value = val

    #
    # This method must be defined on subclasses of
    # IndexedComponent
    #
    def _default(self, idx):
        """
        Returns the default component data value
        """
        exprdata = self._data[idx] = _ExpressionData(self, None)
        return exprdata

    def _add_members(self, init_set):
        """
        Create expression data for all indices in a set
        """
        self._data.update((ndx,_ExpressionData(self,None)) for ndx in init_set)

    def _initialize_members(self, init_set):
        """
        Initialize variable data for all indices in a set
        """
        if self._init_value is not None:
            #
            # Initialize values with a value
            #
            if self._init_value.__class__ is dict:
                for key in init_set:
                    # Skip indices that are not in the dictionary
                    if not key in self._init_value:
                        continue
                    self._data[key].value = self._init_value[key]
            else:
                #
                # Optimization: only call as_numeric once
                #
                val = as_numeric(self._init_value)
                for key in init_set:
                    self._data[key]._value = val
        elif self._init_rule is not None:
            #
            # Initialize values with a rule
            #
            if self.is_indexed():
                for key in init_set:
                    self._data[key].value = \
                        apply_indexed_rule(self, self._init_rule, self._parent(), key)
            else:
                self.value = self._init_rule(self._parent())

    def construct(self, data=None):
        """ Apply the rule to construct values in this set """

        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing Expression, name=%s, from data=%s"
                         % ( self.cname(True), str(data) ))

        if self._constructed:
            return
        self._constructed = True
        #
        # Construct _ExpressionData objects for all index values
        #
        if self.is_indexed():
            self._add_members(self._index)
        else:
            self._data[None] = self
        self._initialize_members(self._index)


# Since this class derives from Component and Component.__getstate__
# just packs up the entire __dict__ into the state dict, there s
# nothing special that we need to do here.  We will just defer to the
# super() get/set state.  Since all of our get/set state methods
# rely on super() to traverse the MRO, this will automatically pick
# up both the Component and Data base classes.

class SimpleExpression(_ExpressionData, Expression):

    def __init__(self, *args, **kwds):
        Expression.__init__(self, *args, **kwds)
        _ExpressionData.__init__(self, self, None)

    def __call__(self, exception=True):
        """
        Compute the value of the expression
        """
        if self._constructed:
            return _ExpressionData.__call__(self, exception=exception)
        if exception:
            raise ValueError("Evaluating the numeric value of expression '%s' "
                             "before the Expression has been constructed (there "
                             "is currently no value to return)."
                             % self.cname(True))


class IndexedExpression(Expression):

    def __call__(self, exception=True):
        """
        Compute the value of the expression
        """
        if exception:
            msg = 'Cannot compute the value of an array of expressions'
            raise TypeError(msg)


register_component(Expression, "Named expressions that can be used in other expressions.")

