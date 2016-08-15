#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("IObjective",
           "objective",
           "objective_list",
           "objective_dict")

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
from pyomo.core.base.numvalue import as_numeric
from pyomo.core.base.component_expression import IExpression
from pyomo.core.base import minimize, maximize

import six

class IObjective(IExpression):
    """
    The interface for optimization objectives.
    """
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    sense = _abstract_readwrite_property(
        doc="The optimization direction for the objective.")

class objective(IObjective):
    """An optimization objective."""
    # To avoid a circular import, for the time being, this
    # property will be set in objective.py
    _ctype = None
    __slots__ = ("_parent",
                 "_expr",
                 "_sense",
                 "__weakref__")
    def __init__(self, expr=None, sense=minimize):
        self._parent = None
        self._expr = None
        self._sense = None

        # call the setters
        self.sense = sense
        self.expr = expr

    #
    # Define the IObjective abstract methods
    #

    @property
    def expr(self):
        return self._expr
    @expr.setter
    def expr(self, expr):
        self._expr = as_numeric(expr) if (expr is not None) else None

    #
    # Define the IExpression abstract methods
    #

    @property
    def sense(self):
        return self._sense
    @sense.setter
    def sense(self, sense):
        """Set the sense (direction) of this objective."""
        if (sense == minimize) or \
           (sense == maximize):
            self._sense = sense
        else:
            raise ValueError("Objective sense must be set to one of: "
                             "[minimize (%s), maximize (%s)]. Invalid "
                             "value: %s'" % (minimize, maximize, sense))

class objective_list(ComponentList):
    """A list-style container for objectives."""
    # To avoid a circular import, for the time being, this
    # property will be set in objective.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(objective_list, self).__init__(*args, **kwds)

class objective_dict(ComponentDict):
    """A dict-style container for objectives."""
    # To avoid a circular import, for the time being, this
    # property will be set in objective.py
    _ctype = None
    __slots__ = ("_parent",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
    def __init__(self, *args, **kwds):
        self._parent = None
        super(objective_dict, self).__init__(*args, **kwds)
