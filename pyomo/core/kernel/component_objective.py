#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.kernel.component_interface import \
    (IComponent,
     _ActiveObjectMixin,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.kernel.component_dict import ComponentDict
from pyomo.core.kernel.component_tuple import ComponentTuple
from pyomo.core.kernel.component_list import ComponentList
from pyomo.core.kernel.component_expression import IExpression

import six

# Constants used to define the optimization sense
minimize=1
maximize=-1

class IObjective(IExpression, _ActiveObjectMixin):
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
        doc=("The optimization direction for the "
             "objective (minimize or maximize)"))

    #
    # Interface
    #

    def is_minimizing(self):
        return self.sense == minimize

class objective(IObjective):
    """An optimization objective."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_expr",
                 "_sense",
                 "__weakref__")
    def __init__(self, expr=None, sense=minimize):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._expr = None
        self._sense = None

        # call the setters
        self.sense = sense
        self.expr = expr

    #
    # Define the IExpression abstract methods
    #

    @property
    def expr(self):
        return self._expr
    @expr.setter
    def expr(self, expr):
        self._expr = as_numeric(expr) if (expr is not None) else None

    #
    # Define the IObjective abstract methods
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
            raise ValueError(
                "Objective sense must be set to one of: "
                "[minimize (%s), maximize (%s)]. Invalid "
                "value: %s'" % (minimize, maximize, sense))

class objective_tuple(ComponentTuple, _ActiveObjectMixin):
    """A tuple-style container for objectives."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.

    def __init__(self, *args, **kwds):
        self._parent = None
        self._storage_key = None
        self._active = True
        super(objective_tuple, self).__init__(*args, **kwds)

class objective_list(ComponentList, _ActiveObjectMixin):
    """A list-style container for objectives."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_data")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.

    def __init__(self, *args, **kwds):
        self._parent = None
        self._storage_key = None
        self._active = True
        super(objective_list, self).__init__(*args, **kwds)

class objective_dict(ComponentDict, _ActiveObjectMixin):
    """A dict-style container for objectives."""
    # To avoid a circular import, for the time being, this
    # property will be set externally
    _ctype = None
    __slots__ = ("_parent",
                 "_storage_key",
                 "_active",
                 "_data")
    if six.PY3:
        # This has to do with a bug in the abc module
        # prior to python3. They forgot to define the base
        # class using empty __slots__, so we shouldn't add a slot
        # for __weakref__ because the base class has a __dict__.
        __slots__ = list(__slots__) + ["__weakref__"]

    def __init__(self, *args, **kwds):
        self._parent = None
        self._storage_key = None
        self._active = True
        super(objective_dict, self).__init__(*args, **kwds)
