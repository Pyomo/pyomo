#Below is copied from numvalue.py

__all__ = ('value', 'is_constant', 'is_fixed', 'is_variable_type',
           'is_potentially_variable', 'LogicalValue', 'TrueConstant',
           'FalseConstant', 'native_logical_types', 'native_types',
           'native_logical_values')

import sys
import logging
from six import iteritems, PY3, string_types, text_type, binary_type

from pyomo.core.expr.expr_common import \
    (_add, _sub, _mul, _div, _pow,
     _neg, _abs, _inplace, _radd,
     _rsub, _rmul, _rdiv, _rpow,
     _iadd, _isub, _imul, _idiv,
     _ipow, _lt, _le, _eq)

from pyomo.core.expr.expr_errors import TemplateExpressionError

logger = logging.getLogger('pyomo.core')

#keep it for now?
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

#0-0 check this
class NonLogicalValue(object):
    """An object that contains a non-logical value

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


nonpyomo_leaf_types = set([NonLogicalValue])



#start
native_logical_types = set([bool])
native_types = set([bool, int ,None]) #0-0 to be chekced
native_logical_values = set([True, False, 1, 0])
#0-0 is this a good idea
#the problem here would be '(1.5) in native__types' gives True

#tbc

def RegisterLogicalType(new_type):
    """
    A utility function for updating the set of types that are
    recognized as handling boolean values. This function does not
    register the type of integer or numeric.

    The argument should be a class (e.g., numpy.bool_).
    """
    global native_logical_types
    global native_types
    native_logical_types.add(new_type)
    native_types.add(new_type)
    #nonpyomo_leaf_types.add(new_type)

def value(obj, exception=True):
    #0-0 Also, not really sure how the exception work
    """
    A utility function that returns the value of a Pyomo object or
    expression.
    # 0-0 do we need to care about this

    Args:
        obj: The argument to evaluate. If it is None, a
            string, or any other primative numeric type,
            then this function simply returns the argument.
            Otherwise, if the argument is a LogicalValue
            then the __call__ method is executed.
        exception (bool): If :const:`True`, then an exception should
            be raised when instances of NumericValue fail to
            evaluate due to one or more objects not being
            initialized to a numeric value (e.g, one or more
            variabvalue(les in an algebraic expression having the
            value None). If :const:`False`, then the function
            returns :const:`None` when an exception occurs.
            Default is True.
    Returns: A numeric value or None.
    """
    #0-0 what should native_types contain, a questions again?
    # or should we even have it at all
    if obj.__class__ in native_types:
        return bool(obj)
    if obj.__class__ is LogicalConstant:
        #   
        # do not expect LogicalConstant with value None.
        #
        #if exception and obj.value is None:
        #    raise ValueError(
        #        "No value for uninitialized LogicalConstant object %s"
        #        % (obj.name,))
        return obj.value
    #
    # Test if we have a duck types for Pyomo expressions
    #
    try:
        obj.is_expression_type()
    except AttributeError:
        #
        # If not, then try to coerce this into a logical constant.  If that
        # works, then return the object
        #
        try:
            check_if_logical_type_and_cache(obj)
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
                    "No value for uninitialized LogicalcValue object %s"
                    % (obj.name,))
            return tmp
        except TemplateExpressionError:
            # Template expressions work by catching this error type. So
            # we should defer this error handling and not log an error
            # message.
            raise
        except:
            logger.error(
                "evaluating object as logical value: %s\n    (object: %s)\n%s"
                % (obj, type(obj), sys.exc_info()[1]))
            raise
    else:
        #
        # Here, we do not try to catch the exception
        #
        return obj(exception=False)

    #0-0 check the above later


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
        check_if_logical_type_and_cache(obj)
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
        check_if_logical_type_and_cache(obj)
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
    #do we need this 0-0
    if (obj.__class__ is 1 ) or (obj.__class is 0):
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

#0-0 is_logical_data
def is_logical_data(obj):
    """
    A utility function that returns a boolean indicating
    whether the input object is logical and not potentially
    variable.
    """
    if obj.__class__ in native_logical_types:
        return True
    elif obj.__class__ in native_types:
        # 0-0 what do we do about int?
        return False
    try:
        # Test if this is an expression object that 
        # is not potentially variable
        return not obj.is_potentially_variable()
    except AttributeError:
        pass
    try:
        # Now we need to confirm that we have an unknown logical type
        check_if_logical_type_and_cache(obj)
       #0-0 this function needs modifying
        return True
    except:
        pass
    return False


_KnownConstants = {}
#tbc

def as_logical(obj):
    # raise error for anything other than {0,1,True,False}
    """
    A function that creates a LogicalConstant object that
    wraps Python logical values.

    Args:
        obj: The logical value that may be wrapped.

    Raises: TypeError if the object is in native_types and not in 
        native_logical_types

    Returns: A true or false LogicalConstant or the original object
    """
    #0-0 concatanate int's?
    #if obj.__class__ in native_logical_types or obj is 1 or obj is 0:
    if obj in native_logical_values: 
        val = _KnownConstants.get(obj, None)
        if val is not None:

            return val 
        #
        # Create the logical constant.  This really
        # should be the only place in the code
        # where these objects are constructed.
        #
        retval = LogicalConstant(obj)
        #
        # Cache the numeric constants.  We used a bounded cache size
        # to avoid unexpectedly large lists of constants.  There are
        # typically a small number of constants that need to be cached.
        #
        # NOTE:  A LFU policy might be more sensible here, but that
        # requires a more complex cache.  It's not clear that that
        # is worth the extra cost.
         
        return retval
        #the rest should be errors
    #
    # Ignore objects that are duck types to work with Pyomo expressions
    #
    try:
        obj.is_expression_type()
        return obj
        #0-0 should be fine for now
    except AttributeError:
        pass
    #
    #check later 0-0
    #try:
    #    return check_if_logical_type_and_cache(obj)
    #except:
    #    pass
    #
    # Generate errors
    #
    if obj.__class__ in native_types:
        raise TypeError("Cannot treat the value '%s' as a constant" % str(obj))
    raise TypeError(
        "Cannot treat the value '%s' as a constant because it has unknown "
        "type '%s'" % (str(obj), type(obj).__name__))

#0-0 I think it's time to look at this one
def check_if_logical_type_and_cache(obj):
   pass
#cut



#the following is a logical version for Logical value

class LogicalValue(object):
    #an abstract class 
    #
    #__slots__ = ('value',)
    __slots__ = ()
    __hash__ = None
    def __getstate__(self):
    #delete the docs for an identation error
    #0-0
        _base = super(LogicalValue, self)
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
        _base = super(LogicalValue, self)
        if hasattr(_base, '__setstate__'):
            return _base.__setstate__(state)
        else:
            for key, val in iteritems(state):
                # Note: per the Python data model docs, we explicitly
                # set the attribute using object.__setattr__() instead
                # of setting self.__dict__[key] = val.
                object.__setattr__(self, key, val)

    def getname(self, fully_qualified=False, name_buffer=None):
        """
        If this is a component, return the component's name on the owning
        block; otherwise return the value converted to a string
        """
        _base = super(LogicalValue, self)
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
        """Return True if this Logical value is a constant value"""
        return False

    def is_fixed(self):
        """Return True if this is a non-constant value that has been fixed"""
        return False

    def is_parameter_type(self):
        """Return False unless this class is a parameter object"""
        return False

    def is_variable_type(self):
        """Return False unless this class is a variable object"""
        return False

    def is_potentially_variable(self):
        """Return True if variables can appear in this expression"""
        return True

    def is_named_expression_type(self):
        """Return True if this Logical value is a named expression"""
        return False

    #what do we about this
    def is_expression_type(self):
        """Return True if this Logical value is an expression"""
        return False

    def is_component_type(self):
        """Return True if this class is a Pyomo component"""
        return False

    def is_relational(self):
        """
        Return True if this Logical value represents a relational expression.
        """
        return False

    def is_indexed(self):
        """Return True if this Logical value is an indexed object"""
        return False


    def __float__(self):
        """
        Coerce the value to a floating point

        Raises:
            TypeError
        """
        raise TypeError(
        """Implicit conversion of Pyomo LogicalValue type `%s' to a float is
        disabled. This error is often the result of using Pyomo components as
        arguments to one of the Python built-in math module functions when
        defining expressions. Avoid this error by using Pyomo-provided math
        functions.""" % (self.name,))

    def __int__(self):
        """
        Coerce the value to an integer

        Raises:
            TypeError
        """
        raise TypeError(
        """Implicit conversion of Pyomo LogicalValue type `%s' to an integer is
        disabled. This error is often the result of using Pyomo components as
        arguments to one of the Python built-in math module functions when
        defining expressions. Avoid this error by using Pyomo-provided math
        functions.""" % (self.name,))

    #tbc    
    def __lt__(self,other):
        """
        Less than operator

        Should not be used for logical values
        """
        return TypeError(
        """Unable to do comparison between logical values. Avoid this error by
        using boolean variable.""")

    def __gt__(self,other):
        """
        Greater than operator

        Should not be used for logical values
        """
        return TypeError(
        """Unable to do comparison between logical values. Avoid this error by
        using boolean variable.""")

    def __le__(self,other):
        """
        Less than or equal operator

        Should not be used for logical values
        """
        return TypeError(
        """Unable to do comparison between logical values. Avoid this error by
        using boolean variable.""")

    def __ge__(self,other):
        """
        Greater than or equal operator

        Should not be used for logical values
        """
        return TypeError(
        """Unable to do comparison between logical values. Avoid this error by
        using boolean variable.""")

    #tbd
    def __eq__(self,other):
        """
        
        Keep it for now 0-0
        """


        #return Equivalence_expression(self,other)
        return True
    
    def __add__(self,other):
        """
        Binary addition

        Should not be used for logical values
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __sub__(self,other):
        """
        Binary subtraction

        This method is called when Python processes the statement::
        
            self - other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __mul__(self,other):
        """
        Binary multiplication

        This method is called when Python processes the statement::
        
            self * other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __div__(self,other):
        """
        Binary division

        This method is called when Python processes the statement::
        
            self / other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __truediv__(self,other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::
        
            self / other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __pow__(self,other):
        """
        Binary power

        This method is called when Python processes the statement::
        
            self ** other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __radd__(self,other):
        """
        Binary addition

        This method is called when Python processes the statement::
        
            other + self
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __rsub__(self,other):
        """
        Binary subtraction

        This method is called when Python processes the statement::
        
            other - self
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __rmul__(self,other):
        """
        Binary multiplication

        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __rdiv__(self,other):
        """Binary division

        This method is called when Python processes the statement::
        
            other / self
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __rtruediv__(self,other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::
        
            other / self
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __rpow__(self,other):
        """
        Binary power

        This method is called when Python processes the statement::
        
            other ** self
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __iadd__(self,other):
        """
        Binary addition

        This method is called when Python processes the statement::
        
            self += other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __isub__(self,other):
        """
        Binary subtraction

        This method is called when Python processes the statement::

            self -= other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __imul__(self,other):
        """
        Binary multiplication

        This method is called when Python processes the statement::

            self *= other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __idiv__(self,other):
        """
        Binary division

        This method is called when Python processes the statement::
        
            self /= other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __itruediv__(self,other):
        """
        Binary division (when __future__.division is in effect)

        This method is called when Python processes the statement::
        
            self /= other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    def __ipow__(self,other):
        """
        Binary power

        This method is called when Python processes the statement::
        
            self **= other
        """
        return TypeError(
        """Unable to perform arithmetic operations between logical values. Avoid this error by
        using boolean variable.""")

    #tbc   0-0 
    def __neg__(self):
        """
        Negation

        This method is called when Python processes the statement::
        
            - self

        Expected to be used as a LogicalExpression
        """
        return TypeError(
        """Unable to take negative of a logical value. Please use the negation
        Logical Expression instead""")# 0-0 add the exact expression after finished
    #keep this one?
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
        return TypeError(
        """Unable to take absolute of a logical value. Avoid this error by
        using boolean variable.""")

    # 0-0
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
        if compute_values:
            try:
                return str(self())
            except:
                pass        
        if not self.is_constant():
            if smap:
                return smap.getSymbol(self, labeler)
            elif labeler is not None:
                return labeler(self)
        return self.__str__()


class LogicalConstant(LogicalValue):
    """An object that contains a constant Logical value.

    Constructor Arguments:
        value           The initial value.
    """

    __slots__ = ('value',)

    #0-0 impose restriction on initialization?
    def __init__(self, value):
        #fine like this?
        if value not in native_logical_values:
            raise TypeError('Not a valid LogicalValue. Unable to create a logical constant')
        self.value = value

    def __getstate__(self):
        state = super(LogicalConstant, self).__getstate__()
        for i in LogicalConstant.__slots__:
            state[i] = getattr(self,i)
        return state

    def is_constant(self):
        return True

    def is_fixed(self):
        return True

    #def is_potentially_variable(self):
    #    return False

    def __str__(self):
        return str(self.value)

    #RaiseTypeError ("value of x?")    
    def __nonzero__(self):
        raise ValueError("Do you mean value of this logical constant : '%s'"
            % (self.name,))

    __bool__ = __nonzero__

    def __call__(self, exception=True):
        """Return the constant value"""
        return self.value

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:         #pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))


# We use as_logical() so that the constant is also in the cache
TrueConstant = as_logical(True)
FalseConstant = as_logical(False)




