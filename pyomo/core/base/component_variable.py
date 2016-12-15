#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("variable",
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
                                       IntegerSet,
                                       BooleanSet,
                                       RealInterval,
                                       IntegerInterval)

import six
from six.moves import xrange

def _extract_domain_type_and_bounds(domain_type,
                                    domain,
                                    lb, ub):
    if domain is not None:
        if domain_type is not None:
            raise ValueError(
                "At most one of the 'domain' and "
                "'domain_type' keywords can be changed "
                "from their default value when "
                "initializing a variable.")
        domain_type = type(domain)
        # handle some edge cases
        if domain_type is BooleanSet:
            domain_type = IntegerSet
        elif domain_type is RealInterval:
            domain_type = RealSet
        elif domain_type is IntegerInterval:
            domain_type = IntegerSet
        domain_lb, domain_ub = domain.bounds()
        if domain_lb is not None:
            if lb is not None:
                raise ValueError(
                    "The 'lb' keyword can not be used "
                    "to initialize a variable when the "
                    "domain lower bound is finite.")
            lb = domain_lb
        if domain_ub is not None:
            if ub is not None:
                raise ValueError(
                    "The 'ub' keyword can not be used "
                    "to initialize a variable when the "
                    "domain upper bound is finite.")
            ub = domain_ub
    elif domain_type is None:
        domain_type = RealSet
    if domain_type not in IVariable._valid_domain_types:
        raise ValueError(
            "Domain type '%s' is not valid. Must be "
            "one of: %s" % (domain_type,
                            IVariable._valid_domain_types))

    return domain_type, lb, ub


class IVariable(IComponent, NumericValue):
    """
    The interface for optimization variables.
    """
    __slots__ = ()

    _valid_domain_types = (RealSet, IntegerSet)

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    domain_type = _abstract_readwrite_property(
        doc=("The domain type of the variable "
             "(RealSet or IntegerSet)"))
    lb = _abstract_readwrite_property(
        doc="The lower bound of the variable")
    ub = _abstract_readwrite_property(
        doc="The upper bound of the variable")
    value = _abstract_readwrite_property(
        doc="The value of the variable")
    fixed = _abstract_readwrite_property(
        doc="The fixed status of the variable")
    stale = _abstract_readwrite_property(
        doc="The stale status of the variable")

    #
    # Interface
    #

    @property
    def bounds(self):
        """Get/Set the bounds as a tuple (lb, ub)."""
        return (self.lb, self.ub)
    @bounds.setter
    def bounds(self, bounds_tuple):
        self.lb, self.ub = bounds_tuple

    def fix(self, *val):
        """
        Fix the variable. Sets the fixed indicator to
        True. An optional value argument will update the
        variable's value before fixing.
        """
        if len(val) == 1:
            self.value = val[0]
        elif len(val) > 1:
            raise TypeError("fix expected at most 1 arguments, "
                            "got %d" % (len(val)))
        self.fixed = True

    def unfix(self):
        """Free the variable. Sets the fixed indicator to False."""
        self.fixed = False

    free=unfix

    #
    # Convenience methods mainly used by the solver
    # interfaces
    #

    def is_continuous(self):
        """Returns True when the domain type is RealSet."""
        return issubclass(self.domain_type, RealSet)

    # this could be expanded to include semi-continuous
    # where as is_integer would not
    def is_discrete(self):
        """Returns True when the domain type is IntegerSet."""
        return issubclass(self.domain_type, IntegerSet)

    def is_integer(self):
        """Returns True when the domain type is IntegerSet."""
        return issubclass(self.domain_type, IntegerSet)

    def is_binary(self):
        """Returns True when the domain type is IntegerSet
        and the bounds are within [0,1]."""
        return self.is_integer() and \
            (self.lb is not None) and \
            (self.ub is not None) and \
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

    #
    # Implement the NumericValue abstract methods
    #

    def is_fixed(self):
        """Returns True if this variable is fixed, otherwise returns False."""
        return self.fixed

    def is_constant(self):
        """Returns False because this is not a constant in an expression."""
        return False

    def _potentially_variable(self):
        """Returns True because this is a variable."""
        return True

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

    def __call__(self, exception=True):
        """Return the value of this variable."""
        return self.value

class variable(IVariable):
    """A decision variable"""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "_domain_type",
                 "lb",
                 "ub",
                 "value",
                 "fixed",
                 "stale",
                 "__weakref__")

    def __init__(self,
                 domain_type=None,
                 domain=None,
                 lb=None,
                 ub=None,
                 value=None,
                 fixed=False):
        self._parent = None
        self._domain_type = None
        self.lb = lb
        self.ub = ub
        self.value = value
        self.fixed = fixed
        self.stale = True
        self._domain_type, self.lb, self.ub = \
            _extract_domain_type_and_bounds(domain_type,
                                            domain,
                                            lb, ub)
    @property
    def domain_type(self):
        """Return the domain type"""
        return self._domain_type
    @domain_type.setter
    def domain_type(self, domain_type):
        """Set the domain type"""
        if domain_type not in IVariable._valid_domain_types:
            raise ValueError(
                "Domain type '%s' is not valid. Must be "
                "one of: %s" % (self.domain_type,
                                IVariable._valid_domain_types))
        self._domain_type = domain_type

    def _set_domain(self, domain):
        """Set the domain. This method updates the
        domain_type property and sets the upper and
        lower bounds to the domain bounds."""
        self.domain_type, self.lb, self.ub = \
            _extract_domain_type_and_bounds(None,
                                            domain,
                                            None, None)
    domain = property(fset=_set_domain,
                      doc=_set_domain.__doc__)

class variable_list(ComponentList):
    """A list-style container for variables."""
    # To avoid a circular import, for the time being, this
    # property will be set in var.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
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
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        super(variable_dict, self).__init__(*args, **kwds)
