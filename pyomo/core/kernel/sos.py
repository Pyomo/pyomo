#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     _abstract_readwrite_property)
from pyomo.core.kernel.container_utils import \
    define_simple_containers

from six.moves import zip

class ISOS(ICategorizedObject):
    """
    The interface for Special Ordered Sets.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    variables = _abstract_readwrite_property(
        doc="The sos variables")
    weights = _abstract_readwrite_property(
        doc="The sos variables")
    level = _abstract_readwrite_property(
        doc="The sos level (e.g., 1,2,...)")

    #
    # Interface
    #

    def items(self):
        """Iterator over the sos variables and weights as tuples"""
        return zip(self.variables, self.weights)

    def __contains__(self, v):
        """Check if the sos contains the variable v"""
        for x in self.variables:
            if id(x) == id(v):
                return True

    def __len__(self):
        """The number of members in the set"""
        return len(self.variables)

class sos(ISOS):
    """A Special Ordered Set of type n."""
    _ctype = ISOS
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_variables",
                 "_weights",
                 "_level",
                 "__weakref__")
    def __init__(self, variables, weights=None, level=1):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._variables = tuple(variables)
        self._weights = None
        self._level = level
        if weights is None:
            self._weights = tuple(range(1,len(self._variables)+1))
        else:
            self._weights = tuple(weights)
            for w in self._weights:
                if not is_numeric_data(w):
                    raise ValueError(
                        "Weights for Special Ordered Sets must be "
                        "expressions restricted to numeric data")

        assert len(self._variables) == len(self._weights)
        assert self._level >= 1

    #
    # Define the ISOS abstract methods
    #

    @property
    def variables(self): return self._variables
    @property
    def weights(self): return self._weights
    @property
    def level(self): return self._level

def sos1(variables, weights=None):
    """A Special Ordered Set of type 1.

    This is an alias for sos(..., level=1)"""
    return sos(variables, weights=weights, level=1)

def sos2(variables, weights=None):
    """A Special Ordered Set of type 2.

    This is an alias for sos(..., level=2).
    """
    return sos(variables, weights=weights, level=2)

# inserts class definitions for simple _tuple, _list, and
# _dict containers into this module
define_simple_containers(globals(),
                         "sos",
                         ISOS)
