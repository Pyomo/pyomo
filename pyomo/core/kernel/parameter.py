#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import (is_numeric_data,
                                      NumericValue)
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.container_utils import define_simple_containers


class IParameter(ICategorizedObject, NumericValue):
    """The interface for mutable numeric data."""
    __slots__ = ()

    def __call__(self, exception=True):
        """Computes the numeric value of this object."""
        raise NotImplementedError     #pragma:nocover

    #
    # Implement the NumericValue abstract methods
    #

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
    """A object for storing a mutable, numeric value that
    can be used to build a symbolic expression."""
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
    # Implement the IParameter abstract methods
    #

    def __call__(self, exception=True):
        """Computes the numeric value of this object."""
        return self.value

    #
    # Interface
    #

    @property
    def value(self):
        """The value of the paramater"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class functional_value(IParameter):
    """An object for storing a numeric function that can be
    used in a symbolic expression.

    Note that models making use of this object may require
    the dill module for serialization.
    """
    _ctype = IParameter
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_fn",
                 "__weakref__")
    def __init__(self, fn=None):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._fn = fn

    #
    # Implement the IParameter abstract methods
    #

    def __call__(self, exception=True):
        if self._fn is None:
            return None
        try:
            val = self._fn()
        except Exception as e:
            if exception:
                raise e
            else:
                return None
        # this exception should never be masked
        if not is_numeric_data(val):
            raise TypeError(
                "Functional value is not numeric data")
        return val

    #
    # Interface
    #

    @property
    def fn(self):
        """The function stored with this object"""
        return self._fn

    @fn.setter
    def fn(self, fn):
        self._fn = fn


# inserts class definitions for simple _tuple, _list, and
# _dict containers into this module
define_simple_containers(globals(),
                         "parameter",
                         IParameter)
