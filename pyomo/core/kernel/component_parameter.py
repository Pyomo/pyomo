#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.kernel.component_interface import \
    (IComponent,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.kernel.component_dict import ComponentDict
from pyomo.core.kernel.component_tuple import ComponentTuple
from pyomo.core.kernel.component_list import ComponentList
from pyomo.core.kernel.numvalue import NumericValue

import six

class IParameter(IComponent, NumericValue):
    """
    The interface for mutable parameters.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    value = _abstract_readwrite_property(
        doc="The value of the parameter")

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

    def _potentially_variable(self):
        """Returns :const:`False` because this object can
        never reference variables."""
        return False

    def polynomial_degree(self):
        """Always return zero because we always validate
        that the stored expression can never reference
        variables."""
        return 0

class parameter(IParameter):
    """A placeholder for a mutable, numeric value."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
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
        """The value of the paramater"""
        return self._value
    @value.setter
    def value(self, value):
        self._value = value

class parameter_tuple(ComponentTuple):
    """A tuple-style container for parameters."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
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
        super(parameter_tuple, self).__init__(*args, **kwds)

class parameter_list(ComponentList):
    """A list-style container for parameters."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
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
    # property will be set externally
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
