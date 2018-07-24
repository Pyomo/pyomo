#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.core.expr
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.kernel.container_utils import \
    define_simple_containers

import six

class IParameter(ICategorizedObject, NumericValue):
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

    def is_parameter_type(self):
        """A boolean indicating that this is a parameter object."""
        #
        # The semantics of ParamData and parameter are different.
        # By default, ParamData are immutable.  Hence, we treat the
        # parameter objects as non-parameter data ... for now.
        #
        return False

    def is_variable_type(self):
        """A boolean indicating that this is a variable object."""
        return False

    def is_fixed(self):
        """A boolean indicating that this parameter is fixed."""
        return True

    def is_potentially_variable(self):
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
    _ctype = IParameter
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_value",
                 "__weakref__")
    def __init__(self, value=None):
        self._parent = None
        self._storage_key = None
        self._active = True
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

# inserts class definitions for simple _tuple, _list, and
# _dict containers into this module
define_simple_containers(globals(),
                         "parameter",
                         IParameter)
