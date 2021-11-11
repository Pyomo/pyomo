#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('value', 'is_constant', 'is_fixed', 'is_variable_type',
           'is_potentially_variable', 'NumericValue', 'ZeroConstant',
           'native_numeric_types', 'native_types', 'nonpyomo_leaf_types',
           'polynomial_degree')

import sys
import logging

from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.errors import PyomoException
from pyomo.core.expr.expr_common import (
    _add, _sub, _mul, _div, _pow,
    _neg, _abs, _radd,
    _rsub, _rmul, _rdiv, _rpow,
    _iadd, _isub, _imul, _idiv,
    _ipow, _lt, _le, _eq
)
# TODO: update imports of these objects to pull from numeric_types
from pyomo.common.numeric_types import (
    nonpyomo_leaf_types, native_types, native_numeric_types,
    native_integer_types, native_boolean_types, native_logical_types,
    RegisterNumericType, RegisterIntegerType, RegisterBooleanType,
    pyomo_constant_types,
)
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_errors import TemplateExpressionError

logger = logging.getLogger('pyomo.core')


def _generate_sum_expression(etype, _self, _other):
    raise RuntimeError("incomplete import of Pyomo expression system")  #pragma: no cover
def _generate_mul_expression(etype, _self, _other):
    raise RuntimeError("incomplete import of Pyomo expression system")  #pragma: no cover
def _generate_other_expression(etype, _self, _other):
    raise RuntimeError("incomplete import of Pyomo expression system")  #pragma: no cover
def _generate_relational_expression(etype, lhs, rhs):
    raise RuntimeError("incomplete import of Pyomo expression system")  #pragma: no cover

##------------------------------------------------------------------------
##
## Standard types of expressions
##
##------------------------------------------------------------------------

class NonNumericValue(object):
    """An object that contains a non-numeric value

    Constructor Arguments:
        value           The initial value.
    """
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __getstate__(self):
        state = {}
        state['value'] = getattr(self,'value')
        return state

    def __setstate__(self, state):
        setattr(self, 'value', state['value'])

nonpyomo_leaf_types.add(NonNumericValue)


def value(obj, exception=True):
    """
    A utility function that returns the value of a Pyomo object or
    expression.

    Args:
        obj: The argument to evaluate. If it is None, a
            string, or any other primative numeric type,
            then this function simply returns the argument.
            Otherwise, if the argument is a NumericValue
            then the __call__ method is executed.
        exception (bool): If :const:`True`, then an exception should
            be raised when instances of NumericValue fail to
            evaluate due to one or more objects not being
            initialized to a numeric value (e.g, one or more
            variables in an algebraic expression having the
            value None). If :const:`False`, then the function
            returns :const:`None` when an exception occurs.
            Default is True.

    Returns: A numeric value or None.
    """
    if obj.__class__ in native_types:
        return obj
    if obj.__class__ in pyomo_constant_types:
        #
        # I'm commenting this out for now, but I think we should never expect
        # to see a numeric constant with value None.
        #
        #if exception and obj.value is None:
        #    raise ValueError(
        #        "No value for uninitialized NumericConstant object %s"
        #        % (obj.name,))
        return obj.value
    #
    # Test if we have a duck types for Pyomo expressions
    #
    try:
        obj.is_expression_type()
    except AttributeError:
        #
        # If not, then try to coerce this into a numeric constant.  If that
        # works, then return the object
        #
        try:
            check_if_numeric_type_and_cache(obj)
            return obj
        except:
            raise TypeError(
                "Cannot evaluate object with unknown type: %s" %
                (type(obj).__name__,))
    #
    # Evaluate the expression object
    #
    if exception:
        #
        # Here, we try to catch the exception
        #

        try:
            tmp = obj(exception=True)
            if tmp is None:
                raise ValueError(
                    "No value for uninitialized NumericValue object %s"
                    % (obj.name,))
            return tmp
        except TemplateExpressionError:
            # Template expressions work by catching this error type. So
            # we should defer this error handling and not log an error
            # message.
            raise
        except:
            logger.error(
                "evaluating object as numeric value: %s\n    (object: %s)\n%s"
                % (obj, type(obj), sys.exc_info()[1]))
            raise
    else:
        #
        # Here, we do not try to catch the exception
        #
        return obj(exception=False)


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
    try:
        # Now we need to confirm that we have an unknown numeric type
        check_if_numeric_type_and_cache(obj)
        # As this branch is only hit for previously unknown (to Pyomo)
        # types that behave reasonably like numbers, we know they *must*
        # be constant.
        return True
    except:
        raise TypeError(
            "Cannot assess properties of object with unknown type: %s"
            % (type(obj).__name__,))

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
    try:
        # Now we need to confirm that we have an unknown numeric type
        check_if_numeric_type_and_cache(obj)
        # As this branch is only hit for previously unknown (to Pyomo)
        # types that behave reasonably like numbers, we know they *must*
        # be fixed.
        return True
    except:
        raise TypeError(
            "Cannot assess properties of object with unknown type: %s"
            % (type(obj).__name__,))

def is_variable_type(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object is a variable.
    """
    if obj.__class__ in native_types:
        return False
    try:
        return obj.is_variable_type()
    except AttributeError:
        return False

def is_potentially_variable(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object can reference variables.
    """
    if obj.__class__ in native_types:
        return False
    try:
        return obj.is_potentially_variable()
    except AttributeError:
        return False

def is_numeric_data(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object is numeric and not potentially
    variable.
    """
    if obj.__class__ in native_numeric_types:
        return True
    elif obj.__class__ in native_types:
        # this likely means it is a string
        return False
    try:
        # Test if this is an expression object that 
        # is not potentially variable
        return not obj.is_potentially_variable()
    except AttributeError:
        pass
    try:
        # Now we need to confirm that we have an unknown numeric type
        check_if_numeric_type_and_cache(obj)
        # As this branch is only hit for previously unknown (to Pyomo)
        # types that behave reasonably like numbers, we know they *must*
        # be numeric data (unless an exception is raised).
        return True
    except:
        pass
    return False

def polynomial_degree(obj):
    """
    A utility function that returns an integer
    that indicates the polynomial degree for an
    object. boolean indicating
    """
    if obj.__class__ in native_numeric_types:
        return 0
    elif obj.__class__ in native_types:
        raise TypeError(
            "Cannot evaluate the polynomial degree of a non-numeric type: %s"
            % (type(obj).__name__,))
    try:
        return obj.polynomial_degree()
    except AttributeError:
        pass
    try:
        # Now we need to confirm that we have an unknown numeric type
        check_if_numeric_type_and_cache(obj)
        # As this branch is only hit for previously unknown (to Pyomo)
        # types that behave reasonably like numbers, we know they *must*
        # be a numeric constant.
        return 0
    except:
        raise TypeError(
            "Cannot assess properties of object with unknown type: %s"
            % (type(obj).__name__,))

#
# It is very common to have only a few constants in a model, but those
# constants get repeated many times.  KnownConstants lets us re-use /
# share constants we have seen before.
#
# Note:
#   For now, all constants are coerced to floats.  This avoids integer
#   division in Python 2.x.  (At least some of the time.)
#
#   When we eliminate support for Python 2.x, we will not need this
#   coercion.  The main difference in the following code is that we will
#   need to index KnownConstants by both the class type and value, since
#   INT, FLOAT and LONG values sometimes hash the same.
#
_KnownConstants = {}

def as_numeric(obj):
    """
    A function that creates a NumericConstant object that
    wraps Python numeric values.

    This function also manages a cache of constants.

    NOTE:  This function is only intended for use when
        data is added to a component.

    Args:
        obj: The numeric value that may be wrapped.

    Raises: TypeError if the object is in native_types and not in 
        native_numeric_types

    Returns: A NumericConstant object or the original object.
    """
    if obj.__class__ in native_numeric_types:
        val = _KnownConstants.get(obj, None)
        if val is not None:
            return val 
        #
        # Coerce the value to a float, if possible
        #
        try:
            obj = float(obj)
        except:
            pass
        #
        # Create the numeric constant.  This really
        # should be the only place in the code
        # where these objects are constructed.
        #
        retval = NumericConstant(obj)
        #
        # Cache the numeric constants.  We used a bounded cache size
        # to avoid unexpectedly large lists of constants.  There are
        # typically a small number of constants that need to be cached.
        #
        # NOTE:  A LFU policy might be more sensible here, but that
        # requires a more complex cache.  It's not clear that that
        # is worth the extra cost.
        #
        if len(_KnownConstants) < 1024:
            _KnownConstants[obj] = retval
            return retval
        #
        return retval
    #
    # Ignore objects that are duck types to work with Pyomo expressions
    #
    try:
        obj.is_expression_type()
        return obj
    except AttributeError:
        pass
    #
    # Test if the object looks like a number.  If so, register that type with a
    # warning.
    #
    try:
        return check_if_numeric_type_and_cache(obj)
    except:
        pass
    #
    # Generate errors
    #
    if obj.__class__ in native_types:
        raise TypeError("Cannot treat the value '%s' as a constant" % str(obj))
    raise TypeError(
        "Cannot treat the value '%s' as a constant because it has unknown "
        "type '%s'" % (str(obj), type(obj).__name__))


def check_if_numeric_type_and_cache(obj):
    """Test if the argument is a numeric type by checking if we can add
    zero to it.  If that works, then we cache the value and return a
    NumericConstant object.

    """
    obj_class = obj.__class__
    if obj_class is (obj + 0).__class__:
        #
        # Coerce the value to a float, if possible
        #
        try:
            obj = float(obj)
        except:
            pass
        #
        # obj may (or may not) be hashable, so we need this try
        # block so that things proceed normally for non-hashable
        # "numeric" types
        #
        retval = NumericConstant(obj)
        try:
            #
            # Create the numeric constant and add to the 
            # list of known constants.
            #
            # Note: we don't worry about the size of the
            # cache here, since we need to confirm that the
            # object is hashable.
            #
            _KnownConstants[obj] = retval
            #
            # If we get here, this is a reasonably well-behaving
            # numeric type: add it to the native numeric types
            # so that future lookups will be faster.
            #
            native_numeric_types.add(obj_class)
            native_types.add(obj_class)
            nonpyomo_leaf_types.add(obj_class)
            #
            # Generate a warning, since Pyomo's management of third-party
            # numeric types is more robust when registering explicitly.
            #
            logger.warning(
                """Dynamically registering the following numeric type:
    %s
Dynamic registration is supported for convenience, but there are known
limitations to this approach.  We recommend explicitly registering
numeric types using the following functions:
    RegisterNumericType(), RegisterIntegerType(), RegisterBooleanType()."""
                % (obj_class.__name__,))
        except:
            pass
        return retval


class NumericValue(PyomoObject):
    """
    This is the base class for numeric values used in Pyomo.
    """

    __slots__ = ()

    # This is required because we define __eq__
    __hash__ = None

    def __getstate__(self):
        """
        Prepare a picklable state of this instance for pickling.

        Nominally, __getstate__() should execute the following::

            state = super(Class, self).__getstate__()
            for i in Class.__slots__:
                state[i] = getattr(self,i)
            return state

        However, in this case, the (nominal) parent class is 'object',
        and object does not implement __getstate__.  So, we will
        check to make sure that there is a base __getstate__() to
        call.  You might think that there is nothing to check, but
        multiple inheritance could mean that another class got stuck
        between this class and "object" in the MRO.

        Further, since there are actually no slots defined here, the
        real question is to either return an empty dict or the
        parent's dict.
        """
        _base = super(NumericValue, self)
        if hasattr(_base, '__getstate__'):
            return _base.__getstate__()
        else:
            return {}

    def __setstate__(self, state):
        """
        Restore a pickled state into this instance

        Our model for setstate is for derived classes to modify
        the state dictionary as control passes up the inheritance
        hierarchy (using super() calls).  All assignment of state ->
        object attributes is handled at the last class before 'object',
        which may -- or may not (thanks to MRO) -- be here.
        """
        _base = super(NumericValue, self)
        if hasattr(_base, '__setstate__'):
            return _base.__setstate__(state)
        else:
            for key, val in state.items():
                # Note: per the Python data model docs, we explicitly
                # set the attribute using object.__setattr__() instead
                # of setting self.__dict__[key] = val.
                object.__setattr__(self, key, val)

    def getname(self, fully_qualified=False, name_buffer=None):
        """
        If this is a component, return the component's name on the owning
        block; otherwise return the value converted to a string
        """
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

    def is_numeric_type(self):
        """Return True if this class is a Pyomo numeric object"""
        return True

    def is_constant(self):
        """Return True if this numeric value is a constant value"""
        return False

    def is_fixed(self):
        """Return True if this is a non-constant value that has been fixed"""
        return False

    def is_potentially_variable(self):
        """Return True if variables can appear in this expression"""
        return False

    def is_relational(self):
        """
        Return True if this numeric value represents a relational expression.
        """
        return False

    def is_indexed(self):
        """Return True if this numeric value is an indexed object"""
        return False

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            :const:`None`
        """
        return self._compute_polynomial_degree(None)

    def _compute_polynomial_degree(self, values):
        """
        Compute the polynomial degree of this expression given
        the degree values of its children.

        Args:
            values (list): A list of values that indicate the degree
                of the children expression.

        Returns:
            :const:`None`
        """
        return None

    def __bool__(self):
        """Coerce the value to a bool

        Numeric values can be coerced to bool only if the value /
        expression is constant.  Fixed (but non-constant) or variable
        values will raise an exception.

        Raises:
            PyomoException

        """
        # Note that we want to implement __bool__, as scalar numeric
        # components (e.g., Param, Var) implement __len__ (since they
        # are implicit containers), and Python falls back on __len__ if
        # __bool__ is not defined.
        if self.is_constant():
            return bool(self())
        raise PyomoException("""
Cannot convert non-constant Pyomo numeric value (%s) to bool.
This error is usually caused by using a Var, unit, or mutable Param in a
Boolean context such as an "if" statement. For example,
    >>> m.x = Var()
    >>> if not m.x:
    ...     pass
would cause this exception.""".strip() % (self,))

    def __float__(self):
        """Coerce the value to a floating point

        Numeric values can be coerced to float only if the value /
        expression is constant.  Fixed (but non-constant) or variable
        values will raise an exception.

        Raises:
            TypeError

        """
        if self.is_constant():
            return float(self())
        raise TypeError("""
Implicit conversion of Pyomo numeric value (%s) to float is disabled.
This error is often the result of using Pyomo components as arguments to
one of the Python built-in math module functions when defining
expressions. Avoid this error by using Pyomo-provided math functions or
explicitly resolving the numeric value using the Pyomo value() function.
""".strip() % (self,))

    def __int__(self):
        """Coerce the value to an integer

        Numeric values can be coerced to int only if the value /
        expression is constant.  Fixed (but non-constant) or variable
        values will raise an exception.

        Raises:
            TypeError

        """
        if self.is_constant():
            return int(self())
        raise TypeError("""
Implicit conversion of Pyomo numeric value (%s) to int is disabled.
This error is often the result of using Pyomo components as arguments to
one of the Python built-in math module functions when defining
expressions. Avoid this error by using Pyomo-provided math functions or
explicitly resolving the numeric value using the Pyomo value() function.
""".strip() % (self,))

    def __lt__(self,other):
        """
        Less than operator

        This method is called when Python processes statements of the form::
        
            self < other
            other > self
        """
        return _generate_relational_expression(_lt, self, other)

    def __gt__(self,other):
        """
        Greater than operator

        This method is called when Python processes statements of the form::
        
            self > other
            other < self
        """
        return _generate_relational_expression(_lt, other, self)

    def __le__(self,other):
        """
        Less than or equal operator

        This method is called when Python processes statements of the form::
        
            self <= other
            other >= self
        """
        return _generate_relational_expression(_le, self, other)

    def __ge__(self,other):
        """
        Greater than or equal operator

        This method is called when Python processes statements of the form::
        
            self >= other
            other <= self
        """
        return _generate_relational_expression(_le, other, self)

    def __eq__(self,other):
        """
        Equal to operator

        This method is called when Python processes the statement::
        
            self == other
        """
        return _generate_relational_expression(_eq, self, other)

    def __add__(self,other):
        """
        Binary addition

        This method is called when Python processes the statement::
        
            self + other
        """
        return _generate_sum_expression(_add,self,other)

    def __sub__(self,other):
        """
        Binary subtraction

        This method is called when Python processes the statement::
        
            self - other
        """
        return _generate_sum_expression(_sub,self,other)

    def __mul__(self,other):
        """
        Binary multiplication

        This method is called when Python processes the statement::
        
            self * other
        """
        return _generate_mul_expression(_mul,self,other)

    def __div__(self,other):
        """
        Binary division

        This method is called when Python processes the statement::
        
            self / other
        """
        return _generate_mul_expression(_div,self,other)

    def __truediv__(self,other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::
        
            self / other
        """
        return _generate_mul_expression(_div,self,other)

    def __pow__(self,other):
        """
        Binary power

        This method is called when Python processes the statement::
        
            self ** other
        """
        return _generate_other_expression(_pow,self,other)

    def __radd__(self,other):
        """
        Binary addition

        This method is called when Python processes the statement::
        
            other + self
        """
        return _generate_sum_expression(_radd,self,other)

    def __rsub__(self,other):
        """
        Binary subtraction

        This method is called when Python processes the statement::
        
            other - self
        """
        return _generate_sum_expression(_rsub,self,other)

    def __rmul__(self,other):
        """
        Binary multiplication

        This method is called when Python processes the statement::
        
            other * self

        when other is not a :class:`NumericValue <pyomo.core.expr.numvalue.NumericValue>` object.
        """
        return _generate_mul_expression(_rmul,self,other)

    def __rdiv__(self,other):
        """Binary division

        This method is called when Python processes the statement::
        
            other / self
        """
        return _generate_mul_expression(_rdiv,self,other)

    def __rtruediv__(self,other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::
        
            other / self
        """
        return _generate_mul_expression(_rdiv,self,other)

    def __rpow__(self,other):
        """
        Binary power

        This method is called when Python processes the statement::
        
            other ** self
        """
        return _generate_other_expression(_rpow,self,other)

    def __iadd__(self,other):
        """
        Binary addition

        This method is called when Python processes the statement::
        
            self += other
        """
        return _generate_sum_expression(_iadd,self,other)

    def __isub__(self,other):
        """
        Binary subtraction

        This method is called when Python processes the statement::

            self -= other
        """
        return _generate_sum_expression(_isub,self,other)

    def __imul__(self,other):
        """
        Binary multiplication

        This method is called when Python processes the statement::

            self *= other
        """
        return _generate_mul_expression(_imul,self,other)

    def __idiv__(self,other):
        """
        Binary division

        This method is called when Python processes the statement::
        
            self /= other
        """
        return _generate_mul_expression(_idiv,self,other)

    def __itruediv__(self,other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::
        
            self /= other
        """
        return _generate_mul_expression(_idiv,self,other)

    def __ipow__(self,other):
        """
        Binary power

        This method is called when Python processes the statement::
        
            self **= other
        """
        return _generate_other_expression(_ipow,self,other)

    def __neg__(self):
        """
        Negation

        This method is called when Python processes the statement::
        
            - self
        """
        return _generate_sum_expression(_neg, self, None)

    def __pos__(self):
        """
        Positive expression

        This method is called when Python processes the statement::
        
            + self
        """
        return self

    def __abs__(self):
        """ Absolute value

        This method is called when Python processes the statement::
        
            abs(self)
        """
        return _generate_other_expression(_abs,self, None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NumericNDArray.__array_ufunc__(
            None, ufunc, method, *inputs, **kwargs)

    def to_string(self, verbose=None, labeler=None, smap=None,
                  compute_values=False):
        """
        Return a string representation of the expression tree.

        Args:
            verbose (bool): If :const:`True`, then the the string 
                representation consists of nested functions.  Otherwise,
                the string representation is an algebraic equation.
                Defaults to :const:`False`.
            labeler: An object that generates string labels for 
                variables in the expression tree.  Defaults to :const:`None`.

        Returns:
            A string representation for the expression tree.
        """
        if compute_values and self.is_fixed():
            try:
                return str(self())
            except:
                pass        
        if not self.is_constant():
            if smap is not None:
                return smap.getSymbol(self, labeler)
            elif labeler is not None:
                return labeler(self)
        return str(self)


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

    def _compute_polynomial_degree(self, result):
        return 0

    def __str__(self):
        return str(self.value)

    def __call__(self, exception=True):
        """Return the constant value"""
        return self.value

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:         #pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))


pyomo_constant_types.add(NumericConstant)

# We use as_numeric() so that the constant is also in the cache
ZeroConstant = as_numeric(0)

#
# Note: the "if numpy_available" in the class definition also ensures
# that the numpy types are registered if numpy is in fact available
#
class NumericNDArray(np.ndarray if numpy_available else object):
    """An ndarray subclass that stores Pyomo numeric expressions"""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            # Convert all incoming types to ndarray (to prevent recursion)
            args = [np.asarray(i) for i in inputs]
            # Set the return type to be an 'object'.  This prevents the
            # logical operators from casting the result to a bool.  This
            # requires numpy >= 1.6
            kwargs['dtype'] = object

        # Delegate to the base ufunc, but return an instance of this
        # class so that additional operators hit this method.
        ans = getattr(ufunc, method)(*args, **kwargs)
        if isinstance(ans, np.ndarray):
            if ans.size == 1:
                return ans[0]
            return ans.view(NumericNDArray)
        else:
            return ans
