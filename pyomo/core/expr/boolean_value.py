__all__ = (
    'BooleanValue', 'TrueConstant', 'FalseConstant',
    'native_logical_values', 'BooleanConstant', 'as_boolean')

import sys
import logging
from six import iteritems

from pyomo.core.expr.expr_errors import TemplateExpressionError
from pyomo.core.expr.numvalue import native_types, native_logical_types
from pyomo.core.expr.expr_common import _and, _or, _equiv, _inv, _xor, _impl

logger = logging.getLogger('pyomo.core')
native_logical_values = {True, False, 1, 0}


def _generate_logical_proposition(etype, _self, _other):
    raise RuntimeError("Incomplete import of Pyomo expression system")  #pragma: no cover


def as_boolean(obj):
    """
    A function that creates a BooleanConstant object that
    wraps Python Boolean values.

    Args:
        obj: The logical value that may be wrapped.

    Raises: TypeError if the object is in native_types and not in 
        native_logical_types

    Returns: A true or false BooleanConstant or the original object
    """
    if obj.__class__ in native_logical_types:
        return BooleanConstant(obj)
    #
    # Ignore objects that are duck types to work with Pyomo expressions
    #
    try:
        obj.is_expression_type()
        return obj
    except AttributeError:
        pass
    #
    # Generate errors
    #
    if obj.__class__ in native_types:
        raise TypeError("Cannot treat the value '%s' as a logical constant" % str(obj))
    raise TypeError(
        "Cannot treat the value '%s' as a logical constant because it has unknown "
        "type '%s'" % (str(obj), type(obj).__name__))


class BooleanValue(object):
    """
    This is the base class for Boolean values used in Pyomo.
    """
    __slots__ = ()
    __hash__ = None

    def __getstate__(self):
        _base = super(BooleanValue, self)
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
        _base = super(BooleanValue, self)
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
        _base = super(BooleanValue, self)
        if hasattr(_base, 'getname'):
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
        return True

    def is_indexed(self):
        """Return True if this Logical value is an indexed object"""
        return False

    def is_numeric_type(self):
        """Boolean values are not numeric."""
        return False

    def __float__(self):
        raise TypeError(
            "Implicit conversion of Pyomo BooleanValue type "
            "'%s' to a float is disabled." % (self.name,))

    def __int__(self):
        raise TypeError(
            "Implicit conversion of Pyomo BooleanValue type "
            "'%s' to an integer is disabled." % (self.name,))

    def __lt__(self, other):
        raise TypeError(
            "Numeric comparison with BooleanValue %s is not allowed." % self.name)

    def __gt__(self, other):
        raise TypeError(
            "Numeric comparison with BooleanValue %s is not allowed." % self.name)

    def __le__(self, other):
        raise TypeError(
            "Numeric comparison with BooleanValue %s is not allowed." % self.name)

    def __ge__(self, other):
        raise TypeError(
            "Numeric comparison with BooleanValue %s is not allowed." % self.name)
    
    def __add__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __sub__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __mul__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __div__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __truediv__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __pow__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __radd__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __rsub__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __rmul__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __rdiv__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __rtruediv__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __rpow__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __iadd__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __isub__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __imul__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __idiv__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __itruediv__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __ipow__(self, other):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __neg__(self):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __pos__(self):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __abs__(self):
        raise TypeError("Unable to perform arithmetic operations between Boolean values.")

    def __bool__(self):
        """Evaluation as a boolean (using if, and, or keywords)"""
        raise TypeError("Use value() for finding the value of a BooleanValue.")

    def __eq__(self, other):
        """
        Construct an EquivalenceExpression using operator '=='
        """
        return _generate_logical_proposition(_equiv, self, other)

    def equivalent_to(self, other):
        """
        Construct an EquivalenceExpression using method "equivalent_to"
        """
        return _generate_logical_proposition(_equiv, self, other)

    def __and__(self, other):
        """
        Construct an AndExpression using operator '&'
        """
        return _generate_logical_proposition(_and, self, other)

    def __rand__(self, other):
        """
        Construct an AndExpression using operator '&'
        """
        return _generate_logical_proposition(_and, other, self)

    def and_(self, other):
        """
        Construct an AndExpression using method "and_"

        Trailing underscore avoids name conflict with keyword 'and'.

        """
        return _generate_logical_proposition(_and, self, other)

    def __or__(self, other):
        """
        Construct an OrExpression using operator '|'
        """
        return _generate_logical_proposition(_or, self, other)

    def __ror__(self, other):
        """
        Construct an OrExpression using operator '|'
        """
        return _generate_logical_proposition(_or, other, self)

    def or_(self, other):
        """
        Construct an OrExpression using method "or_"

        Trailing underscore avoids name conflict with keyword 'or'.

        """
        return _generate_logical_proposition(_or, self, other)

    def __invert__(self):
        """
        Construct an XorExpression using operator '~'
        """
        return _generate_logical_proposition(_inv, self, None)

    def __xor__(self, other):
        """
        Construct an XorExpression using operator '^'
        """
        return _generate_logical_proposition(_xor, self, other)

    def xor(self, other):
        """
        Construct an EquivalenceExpression using method "xor"
        """
        return _generate_logical_proposition(_xor, self, other)

    def __lshift__(self, other):
        """
        Construct an ImplicationExpression using operator '<<'
        """
        return _generate_logical_proposition(_impl, other, self)

    def __rshift__(self, other):
        """
        Construct an ImplicationExpression using operator '>>'
        """
        return _generate_logical_proposition(_impl, self, other)

    def implies(self, other):
        """
        Construct an ImplicationExpression using method "implies"
        """
        return _generate_logical_proposition(_impl, self, other)

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


class BooleanConstant(BooleanValue):
    """An object that contains a constant Logical value.

    Constructor Arguments:
        value           The initial value.
    """

    __slots__ = ('value',)

    def __init__(self, value):
        if value not in native_logical_values:
            raise TypeError('Not a valid BooleanValue. Unable to create a logical constant')
        self.value = value

    def __getstate__(self):
        state = super(BooleanConstant, self).__getstate__()
        for i in BooleanConstant.__slots__:
            state[i] = getattr(self, i)
        return state

    def is_constant(self):
        return True

    def is_fixed(self):
        return True

    def is_potentially_variable(self):
        return False

    def __str__(self):
        return str(self.value)

    def __nonzero__(self):
        raise ValueError(
            "Boolean constant cannot be compared to zero: '%s'"
            % (self.name,))

    def __bool__(self):
        return self.value

    def __call__(self, exception=True):
        """Return the constant value"""
        return self.value

    def pprint(self, ostream=None, verbose=False):
        if ostream is None:         #pragma:nocover
            ostream = sys.stdout
        ostream.write(str(self))


# We use as_boolean() so that the constant is also in the cache
TrueConstant = as_boolean(True)
FalseConstant = as_boolean(False)
