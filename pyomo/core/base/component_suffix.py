#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("suffix",)

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
from pyomo.core.base.component_map import ComponentMap

import six

_infinity = pyutilib.math.infinity
_negative_infinity = -pyutilib.math.infinity

class suffix(IComponent, _IActiveComponent, ComponentMap):
    """A container for storing extranious model data that
    can be imported to or exported from a solver."""
    # To avoid a circular import, for the time being, this
    # property will be set in suffix.py
    _ctype = None
    __slots__ = ("_parent",
                 "_direction",
                 "_datatype")
    if six.PY3:
        __slots__ = list(__slots__) + ["__weakref__"]

    # neither sent to solver or received from solver
    LOCAL  = 0
    # sent to solver or other external location
    EXPORT = 1
    # obtained from solver or other external source
    IMPORT = 2
    # both
    IMPORT_EXPORT = 3

    _directions = {LOCAL: 'suffix.LOCAL',
                   EXPORT: 'suffix.EXPORT',
                   IMPORT: 'suffix.IMPORT',
                   IMPORT_EXPORT: 'suffix.IMPORT_EXPORT'}

    # datatypes (numbers are compatible with ASL bitcodes)
    FLOAT = 4
    INT = 0
    _datatypes = {FLOAT: 'suffix.FLOAT',
                  INT: 'suffix.INT',
                  None: str(None)}

    def __init__(self, *args, **kwds):
        self._parent = None
        self._direction = None
        self._datatype = None

        # call the setters
        self.direction = kwds.pop('direction', suffix.LOCAL)
        self.datatype = kwds.pop('direction', suffix.FLOAT)

        super(suffix, self).__init__(*args, **kwds)

    def export_enabled(self):
        """
        Returns True when this suffix is enabled for export to
        solvers.
        """
        return bool(self._direction & suffix.EXPORT)

    def import_enabled(self):
        """
        Returns True when this suffix is enabled for import from
        solutions.
        """
        return bool(self._direction & suffix.IMPORT)

    @property
    def datatype(self):
        """Return the suffix datatype."""
        return self._datatype
    @datatype.setter
    def datatype(self, datatype):
        """Set the suffix datatype."""
        if datatype not in self._datatypes:
            raise ValueError(
                "Suffix datatype must be one of: %s. \n"
                "Value given: %s"
                % (list(self._datatypes.values()),
                   datatype))
        self._datatype = datatype

    @property
    def direction(self):
        """Return the suffix direction."""
        return self._direction
    @direction.setter
    def direction(self, direction):
        """Set the suffix direction."""
        if not direction in self._directions:
            raise ValueError(
                "Suffix direction must be one of: %s. \n"
                "Value given: %s"
                % (list(self._directions.values()),
                   direction))
        self._direction = direction

    #
    # Methods that are deprecated
    #

    def set_all_values(self, value):
        logger.warning("DEPRECATION WARNING: suffix.set_all_values "
                       "will be removed in the future.")
        for ndx in self:
            self[ndx] = value

    def clear_value(self, component, expand=True):
        logger.warning("DEPRECATION WARNING: suffix.clear_value "
                       "will be removed in the future. Use "
                       "'del suffix[key]' instead.")
        if expand and component.is_indexed():
            for component_ in itervalues(component):
                try:
                    del self[component_]
                except KeyError:
                    pass
        else:
            try:
                del self[component]
            except KeyError:
                pass

    def clear_all_values(self):
        logger.warning(
            "DEPRECATION WARNING: suffix.clear_all_values "
            "is replaced with suffix.clear")
        self.clear()

    def get_datatype(self):
        logger.warning(
            "DEPRECATION WARNING: suffix.get_datatype is replaced "
            "with the property suffix.datatype")
        return self.datatype

    def set_datatype(self, datatype):
        logger.warning(
            "DEPRECATION WARNING: suffix.set_datatype is replaced "
            "with the property setter suffix.datatype")
        self.datatype = datatype

    def get_direction(self):
        logger.warning(
            "DEPRECATION WARNING: suffix.get_direction is replaced "
            "with the property suffix.direction")
        return self.direction

    def set_direction(self, direction):
        logger.warning(
            "DEPRECATION WARNING: suffix.set_direction is replaced "
            "with the property setter suffix.direction")
        self.direction = direction

