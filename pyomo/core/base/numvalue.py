#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = [ 'value', 'NumericValue', 'as_numeric', 'NumericConstant',
            'is_constant', 'is_fixed']

import sys
import logging
from six import iteritems, PY3, string_types, text_type, binary_type

from pyomo.core.base.expr_common import \
    _add, _sub, _mul, _div, _pow, _neg, _abs, _inplace, \
    _radd, _rsub, _rmul, _rdiv, _rpow, _iadd, _isub, _imul, _idiv, _ipow, \
    _lt, _le, _eq

logger = logging.getLogger('pyomo.core')

def generate_expression(etype, _self,_other):
    raise RuntimeError("incomplete import of Pyomo expression system")
def generate_relational_expression(etype, lhs, rhs):
    raise RuntimeError("incomplete import of Pyomo expression system")

##------------------------------------------------------------------------
##
## Standard types of expressions
##
##------------------------------------------------------------------------

def _old_value(obj):
    """
    A utility function that returns the value of a Pyomo object or expression.

    If the argument is None, a numeric value or a string, then this
    function simply returns the argument.  Otherwise, if the argument is
    a NumericValue then the __call__ method is executed.
    """
    if obj is None:
        return None
    if type(obj) in (bool,int,long,float,str):
        return obj
    if not isinstance(obj, NumericValue):
        raise ValueError("Object %s is not a NumericValue object" % (obj,))
    tmp = obj()
    if tmp is None:
        raise ValueError("No value for uninitialized NumericValue object %s"
                         % (obj.name,))
    return tmp

# It is *significantly* faster to build the list of types we want to
# test against as a "static" set, and not to regenerate it locally for
# every call.  Plus, this allows us to dynamically augment the set
# with new "native" types (e.g., from NumPy)
#
# Note: These type sets are used in set_types.py for domain validation
#       For backward compatibility reasons, we include str in the set
#       of valid types for bool. We also avoid updating the numeric
#       and integer type sets when a new boolean type is registered
#       because not all boolean types exhibit numeric properties
#       (e.g., numpy.bool_)
#
native_numeric_types = set([ int, float, bool ])
native_integer_types = set([ int, bool ])
native_boolean_types = set([ int, bool, str ])
try:
    native_numeric_types.add(long)
    native_integer_types.add(long)
    native_boolean_types.add(long)
except:
    pass

native_types = set([ bool, str, type(None) ])
if PY3:
    native_types.add(bytes)
    native_boolean_types.add(bytes)
else:
    native_types.add(unicode)
    native_boolean_types.add(unicode)
native_types.update( native_numeric_types )
native_types.update( native_integer_types )
native_types.update( native_boolean_types )

def RegisterNumericType(new_type):
    """
    A utility function for updating the set of types that are
    recognized to handle numeric values.

    The argument should be a class (e.g, numpy.float64).
    """
    global native_numeric_types
    global native_types
    native_numeric_types.add(new_type)
    native_types.add(new_type)

def RegisterIntegerType(new_type):
    """
    A utility function for updating the set of types that are
    recognized to handle integer values. This also registers the type
    as numeric but does not register it as boolean.

    The argument should be a class (e.g., numpy.int64).
    """
    global native_numeric_types
    global native_integer_types
    global native_types
    native_numeric_types.add(new_type)
    native_integer_types.add(new_type)
    native_types.add(new_type)

def RegisterBooleanType(new_type):
    """
    A utility function for updating the set of types that are
    recognized as handling boolean values. This function does not
    register the type of integer or numeric.

    The argument should be a class (e.g., numpy.bool_).
    """
    global native_boolean_types
    global native_types
    native_boolean_types.add(new_type)
    native_types.add(new_type)

def value(obj, exception=True):
    """
    A utility function that returns the value of a Pyomo object or
    expression.

    If the argument is None, a numeric value or a string, then this
    function simply returns the argument.  Otherwise, if the argument
    is a NumericValue then the __call__ method is executed.
    """
    if obj.__class__ in native_types:
        return obj
    try:
        numeric = obj.as_numeric()
    except AttributeError:
        try:
            numeric = as_numeric(obj)
        except ValueError:
            if isinstance( obj, string_types + (text_type, binary_type) ):
                native_types.add(type(obj))
                return obj
            raise
    try:
        tmp = numeric(exception=exception)
    except:
        logger.error(
            "evaluating object as numeric value: %s\n    (object: %s)\n%s"
            % (obj, type(obj), sys.exc_info()[1]))
        raise

    if exception and (tmp is None):
        raise ValueError("No value for uninitialized NumericValue object %s"
                         % (obj.name,))
    return tmp


def is_constant(obj):
    """
    A utility function that returns a boolean that indicates
    whether the object is a constant.
    """
    # This method is rarely, if ever, called.  Plus, since the
    # expression generation (and constraint generation) system converts
    # everything to NumericValues, it is better (i.e., faster) to assume
    # that the obj is a NumericValue
    #
    # JDS: NB: I am not sure why we allow str to be a constant, but
    # since we have historically done so, we check for type membership
    # in native_types and not in native_numeric_types.
    #
    if obj.__class__ in native_types:
        return True
    try:
        return obj.is_constant()
    except AttributeError:
        pass
    return as_numeric(obj).is_constant()

def is_fixed(obj):
    """
    A utility function that returns a boolean that indicates
    whether the input object's value is fixed.
    """
    # JDS: NB: I am not sure why we allow str to be a constant, but
    # since we have historically done so, we check for type membership
    # in native_types and not in native_numeric_types.
    #
    if obj.__class__ in native_types:
        return True
    try:
        return obj.is_fixed()
    except AttributeError:
        pass
    return as_numeric(obj).is_fixed()

def potentially_variable(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object can reference variables.
    """
    if obj.__class__ in native_types:
        return False
    try:
        return obj._potentially_variable()
    except AttributeError:
        pass
    return as_numeric(obj)._potentially_variable()

# It is very common to have only a few constants in a model, but those
# constants get repeated many times.  KnownConstants lets us re-use /
# share constants we have seen before.
KnownConstants = {}

def as_numeric(obj):
    """
    Verify that this obj is a NumericValue or intrinsic value.
    """
    # int and float are *so* common that it pays to treat them specially
    if obj.__class__ in native_numeric_types:
        if obj in KnownConstants:
            return KnownConstants[obj]
        else:
            # Because INT, FLOAT, and sometimes LONG hash the same, we
            # want to convert them to a common type (at the very least,
            # so that the order in which tests run does not change the
            # results!)
            try:
                tmp = float(obj)
                if tmp == obj:
                    tmp = NumericConstant(tmp)
                    KnownConstants[obj] = tmp
                    return tmp
            except:
                pass

            tmp = NumericConstant(obj)
            KnownConstants[obj] = tmp
            return tmp
    try:
        return obj.as_numeric()
    except AttributeError:
        pass
    try:
        if obj.__class__ is (obj + 0).__class__:
            # obj may (or may not) be hashable, so we need this try
            # block so that things proceed normally for non-hashable
            # "numeric" types
            try:
                if obj in KnownConstants:
                    return KnownConstants[obj]
                else:
                    tmp = NumericConstant(obj)
                    KnownConstants[obj] = tmp

                    # If we get here, this is a reasonably well-behaving
                    # numeric type: add it to the native numeric types
                    # so that future lookups will be faster.
                    native_numeric_types.add(obj.__class__)
                    # native numeric types are also native types
                    native_types.add(obj.__class__)

                    return tmp
            except:
                return NumericConstant(obj)
    except:
        pass
    raise TypeError(
        "Cannot convert object of type '%s' (value = %s) to a numeric value."
        % (type(obj).__name__, obj, ))


class NumericValue(object):
    """This is the base class for numeric values used in Pyomo.

    For efficiency purposes, some derived classes do not call the base
    class __init__() (e.g. see the "_ExpressionBase" class defined in
    "expr.py").
    """

    __slots__ = ()

    # This is required because we define __eq__
    __hash__ = None

    def __getstate__(self):
        """Prepare a picklable state of this instance for pickling.

        Nominally, __getstate__() should return:

            state = super(Class, self).__getstate__()
            for i in Class.__slots__:
                state[i] = getattr(self,i)
            return state

        However, in this case, the (nominal) parent class is 'object',
        and object does not implement __getstate__.  So, we will check
        to make sure that there is a base __getstate__() to call...
        You might think that there is nothing to check, but multiple
        inheritance could mean that another class got stuck between
        this class and "object" in the MRO.

        Further, since there are actually no slots defined here, the
        real question is to either return an empty dict or the
        parent's dict."""
        _base = super(NumericValue, self)
        if hasattr(_base, '__getstate__'):
            return _base.__getstate__()
        else:
            return {}

    def __setstate__(self, state):
        """Restore a pickled state into this instance

        Note: our model for setstate is for derived classes to modify
        the state dictionary as control passes up the inheritance
        hierarchy (using super() calls).  All assignment of state ->
        object attributes is handled at the last class before 'object'
        (which may -- or may not (thanks to MRO) -- be here."""
        _base = super(NumericValue, self)
        if hasattr(_base, '__setstate__'):
            return _base.__setstate__(state)
        else:
            for key, val in iteritems(state):
                # Note: per the Python data model docs, we explicitly
                # set the attribute using object.__setattr__() instead
                # of setting self.__dict__[key] = val.
                object.__setattr__(self, key, val)

    def getname(self, fully_qualified=False, name_buffer=None):
        """If this is a component, return the component's name on the owning
        block; otherwise return the value converted to a string"""
        _base = super(NumericValue, self)
        if hasattr(_base,'getname'):
            return _base.getname(fully_qualified, name_buffer)
        else:
            return str(type(self))

    @property
    def name(self):
        return self.getname(fully_qualified=True)

    @property
    def local_name(self):
        return self.getname(fully_qualified=False)

    def cname(self, *args, **kwds):
        logger.warning(
            "DEPRECATED: The cname() method has been renamed to getname()." )
        return self.getname(*args, **kwds)

    def is_constant(self):
        """Return True if this numeric value is a constant value"""
        return False

    def is_fixed(self):
        """Return True if this is a non-constant value that has been fixed"""
        return False

    def _potentially_variable(self):
        """Return True if variables can appear in this expression"""
        return True

    def is_expression(self):
        """Return True if this numeric value is an expression"""
        return False

    def is_relational(self):
        """
        Return True if this numeric value represents a relational expression.
        """
        return False

    def is_indexed(self):
        """Return True if this numeric value is an indexed object"""
        return False

    def as_numeric(self):
        return self

    def polynomial_degree(self):
        """Return the polynomial degree of this expression"""
        return self._polynomial_degree(None)

    def _polynomial_degree(self, result):
        """Private method that computes the polynomial degree of this
        expression"""
        return None

    def __float__(self):
        """Coerce the value to a floating point"""
        raise TypeError(
"""Implicit conversion of Pyomo NumericValue type `%s' to a float is
disabled. This error is often the result of using Pyomo components as
arguments to one of the Python built-in math module functions when
defining expressions. Avoid this error by using Pyomo-provided math
functions.""" % (self.name,))

    def __int__(self):
        """Coerce the value to an integer"""
        raise TypeError(
"""Implicit conversion of Pyomo NumericValue type `%s' to an integer is
disabled. This error is often the result of using Pyomo components as
arguments to one of the Python built-in math module functions when
defining expressions. Avoid this error by using Pyomo-provided math
functions.""" % (self.name,))

    def __lt__(self,other):
        """Less than operator

        (Called in response to 'self < other' or 'other > self'.)
        """
        return generate_relational_expression(_lt, self, other)

    def __gt__(self,other):
        """Greater than operator

        (Called in response to 'self > other' or 'other < self'.)
        """
        return generate_relational_expression(_lt, other, self)

    def __le__(self,other):
        """Less than or equal operator

        (Called in response to 'self <= other' or 'other >= self'.)
        """
        return generate_relational_expression(_le, self, other)

    def __ge__(self,other):
        """Greater than or equal operator

        (Called in response to 'self >= other' or 'other <= self'.)
        """
        return generate_relational_expression(_le, other, self)

    def __eq__(self,other):
        """Equal to operator

        (Called in response to 'self = other'.)
        """
        return generate_relational_expression(_eq, self, other)

    def __add__(self,other):
        """Binary addition

        (Called in response to 'self + other'.)
        """
        return generate_expression(_add,self,other)

    def __sub__(self,other):
        """ Binary subtraction

        (Called in response to 'self - other'.)
        """
        return generate_expression(_sub,self,other)

    def __mul__(self,other):
        """ Binary multiplication

        (Called in response to 'self * other'.)
        """
        return generate_expression(_mul,self,other)

    def __div__(self,other):
        """ Binary division

        (Called in response to 'self / other'.)
        """
        return generate_expression(_div,self,other)

    def __truediv__(self,other):
        """ Binary division

        (Called in response to 'self / other' with __future__.division.)
        """
        return generate_expression(_div,self,other)

    def __pow__(self,other):
        """ Binary power

        (Called in response to 'self ** other'.)
        """
        return generate_expression(_pow,self,other)

    def __radd__(self,other):
        """Binary addition

        (Called in response to 'other + self'.)
        """
        return generate_expression(_radd,self,other)

    def __rsub__(self,other):
        """ Binary subtraction

        (Called in response to 'other - self'.)
        """
        return generate_expression(_rsub,self,other)

    def __rmul__(self,other):
        """ Binary multiplication

        (Called in response to 'other * self' when other is not a NumericValue.)
        """
        return generate_expression(_rmul,self,other)

    def __rdiv__(self,other):
        """ Binary division

        (Called in response to 'other / self'.)
        """
        return generate_expression(_rdiv,self,other)

    def __rtruediv__(self,other):
        """ Binary division

        (Called in response to 'other / self' with __future__.division.)
        """
        return generate_expression(_rdiv,self,other)

    def __rpow__(self,other):
        """ Binary power

        (Called in response to 'other ** self'.)
        """
        return generate_expression(_rpow,self,other)

    def __iadd__(self,other):
        """Binary addition

        (Called in response to 'self += other'.)
        """
        return generate_expression(_iadd,self,other)

    def __isub__(self,other):
        """ Binary subtraction

        (Called in response to 'self -= other'.)
        """
        return generate_expression(_isub,self,other)

    def __imul__(self,other):
        """ Binary multiplication

        (Called in response to 'self *= other'.)
        """
        return generate_expression(_imul,self,other)

    def __idiv__(self,other):
        """ Binary division

        (Called in response to 'self /= other'.)
        """
        return generate_expression(_idiv,self,other)

    def __itruediv__(self,other):
        """ Binary division

        (Called in response to 'self /= other' with __future__.division.)
        """
        return generate_expression(_idiv,self,other)

    def __ipow__(self,other):
        """ Binary power

        (Called in response to 'self **= other'.)
        """
        return generate_expression(_ipow,self,other)

    def __neg__(self):
        """ Negation

        (Called in response to '- self'.)
        """
        return generate_expression(_neg,self, None)

    def __pos__(self):
        """ Positive expression

        (Called in response to '+ self'.)
        """
        return self

    def __abs__(self):
        """ Absolute value

        (Called in response to 'abs(self)'.)
        """
        return generate_expression(_abs,self, None)

class NumericConstant(NumericValue):
    """An object that contains a constant numeric value.

    Constructor Arguments:
        value           The initial value.
    """

    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __getstate__(self):
        state = super(NumericConstant, self).__getstate__()
        for i in NumericConstant.__slots__:
            state[i] = getattr(self,i)
        return state

    def is_constant(self):
        return True

    def is_fixed(self):
        return True

    def _potentially_variable(self):
        return False

    def _polynomial_degree(self, result):
        return 0

    def __str__(self):
        return str(self.value)

    def to_string(self, ostream=None, verbose=None, precedence=0):
        if ostream is None:
            ostream = sys.stdout
        ostream.write(self.__str__())

    def __nonzero__(self):
        """Return True if the value is defined and non-zero"""
        if self.value:
            return True
        if self.value is None:
            raise ValueError("Numeric Constant: value is undefined")
        return False

    __bool__ = __nonzero__

    def __call__(self, exception=True):
        """Return the constant value"""
        return self.value

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:         #pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))

# We use as_numeric() so that the constant is also in the cache
ZeroConstant = as_numeric(0)
