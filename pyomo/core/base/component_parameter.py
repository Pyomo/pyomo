#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("parameter",
           "parameter_list",
           "parameter_dict")

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
from pyomo.core.base.numvalue import (NumericValue,
                                      is_fixed,
                                      as_numeric)

import six

class IParameter(IComponent, NumericValue):
    """
    The interface for reusable parameters.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    value = _abstract_readwrite_property(
        doc="Access the stored value")

    #
    # Implement the NumericValue abstract methods
    #

    def __call__(self, exception=True):
        """Compute the value of this parameter."""
        return self.value

    def is_constant(self):
        """A boolean indicating that this parameter is constant."""
        return False

    def is_fixed(self):
        """A boolean indicating that this parameter is fixed."""
        return True

class parameter(IParameter):
    """A placeholder for a numeric value in an expression."""
    # To avoid a circular import, for the time being, this
    # property will be set in param.py
    _ctype = None
    __slots__ = ("_parent",
                 "_value",
                 "__weakref__")
    def __init__(self, value=None):
        self._parent = None
        self._value = value

    #
    # Define the IParameter abstract methods
    #

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, value):
        self._value = value

class parameter_list(ComponentList):
    """A list-style container for parameters."""
    # To avoid a circular import, for the time being, this
    # property will be set in param.py
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
        super(parameter_list, self).__init__(*args, **kwds)

class parameter_dict(ComponentDict):
    """A dict-style container for parameters."""
    # To avoid a circular import, for the time being, this
    # property will be set in param.py
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
        super(parameter_dict, self).__init__(*args, **kwds)
