#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("IVariable",
           "variable",
           "variable_list",
           "variable_dict")

import abc

import pyutilib.math

from pyomo.core.base.component_interface import \
    (IComponent,
     _IActiveComponent,
     _IActiveComponentContainer,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.base.component_dict import ComponentDict
from pyomo.core.base.component_list import ComponentList
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)

import six

_infinity = pyutilib.math.infinity
_negative_infinity = -pyutilib.math.infinity

class IVariable(IComponent, NumericValue):
    """
    The interface for optimization variables.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    domain_type = _abstract_readwrite_property()
    lb = _abstract_readwrite_property()
    ub = _abstract_readwrite_property()
    value = _abstract_readwrite_property()
    fixed = _abstract_readwrite_property()
    stale = _abstract_readwrite_property()

    #
    # Interface
    #

    @property
    def bounds(self):
        """Returns the tuple (lower bound, upper bound)."""
        return (self.lb, self.ub)
    @bounds.setter
    def bounds(self, bounds_tuple):
        self.lb, self.ub = bounds_tuple

    def is_integer(self):
        """Returns True when the domain class is IntegerSet."""
        return issubclass(self.domain_type, IntegerSet)

    def is_binary(self):
        """Returns True when the domain class is BooleanSet."""
        return self.is_integer() and \
            (self.lb >= 0) and \
            (self.ub <= 1)

# TODO?
#    def is_semicontinuous(self):
#        """
#        Returns True when the domain class is SemiContinuous.
#        """
#        return issubclass(self.domain_type, SemiRealSet)

#    def is_semiinteger(self):
#        """
#        Returns True when the domain class is SemiInteger.
#        """
#        return issubclass(self.domain_type, SemiIntegerSet)

    def is_continuous(self):
        """Returns True when the domain is an instance of RealSet."""
        return issubclass(self.domain_type, RealSet)

    #
    # Implement the NumericValue abstract methods
    #

    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def _polynomial_degree(self, result):
        """
        If the variable is fixed, it represents a constant
        is a polynomial with degree 0. Otherwise, it has
        degree 1. This method is used in expressions to
        compute polynomial degree.
        """
        if self.fixed:
            return 0
        return 1

    def __nonzero__(self):
        """
        Return True if the value is defined and non-zero.
        """
        if self.value:
            return True
        if self.value is None:
            raise ValueError("Var value is undefined")
        return False

    def __call__(self, exception=True):
        """Compute the value of this variable."""
        return self.value

    __bool__ = __nonzero__

    def fix(self, *val):
        """
        Set the fixed indicator to True. Value argument is optional,
        indicating the variable should be fixed at its current value.
        """
        raise NotImplementedError

    def unfix(self):
        """Sets the fixed indicator to False."""
        raise NotImplementedError

    free=unfix

class variable(IVariable):
    """A decision variable"""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "domain_type",
                 "lb",
                 "ub",
                 "value",
                 "fixed",
                 "stale",
                 "__weakref__")
    def __init__(self,
                 domain_type=RealSet,
                 lb=_negative_infinity,
                 ub=_infinity,
                 value=None,
                 fixed=False):
        self._parent = None
        self.domain_type = domain_type
        self.lb = lb
        self.ub = ub
        self.value = value
        self.fixed = fixed
        self.stale = True

class variable_list(ComponentList):
    """A list-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_list, self).__init__(*args, **kwds)

class variable_dict(ComponentDict):
    """A dict-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_dict, self).__init__(*args, **kwds)
