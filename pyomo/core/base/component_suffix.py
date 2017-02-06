#  _________________________________________________________________________
#
#  PyUtilib: A Python utility library.
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  _________________________________________________________________________

__all__ = ("suffix",)

import logging
import abc

import pyutilib.math

from pyomo.core.base.component_interface import \
    (IComponent,
     _IActiveComponentMixin,
     _abstract_readwrite_property,
     _abstract_readonly_property)
from pyomo.core.base.numvalue import NumericValue
from pyomo.core.base.set_types import (RealSet,
                                       IntegerSet)
from pyomo.core.base.component_map import ComponentMap

import six

logger = logging.getLogger('pyomo.core')

_noarg = object()

class suffix(ComponentMap, IComponent, _IActiveComponentMixin):
    """
    A container for storing extranious model data that can
    be imported to or exported from a solver.
    """
    # To avoid a circular import, for the time being, this
    # property will be set in suffix.py
    _ctype = None
    __slots__ = ("_parent",
                 "_active",
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
        self._active = True
        self._direction = None
        self._datatype = None

        # call the setters
        self.direction = kwds.pop('direction', suffix.LOCAL)
        self.datatype = kwds.pop('datatype', suffix.FLOAT)
        super(suffix, self).__init__(*args, **kwds)

    #
    # Interface
    #

    @property
    def export_enabled(self):
        """
        Returns True when this suffix is enabled for export to
        solvers.
        """
        return bool(self._direction & suffix.EXPORT)

    @property
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

    def clear_value(self, component):
        logger.warning("DEPRECATION WARNING: suffix.clear_value "
                       "will be removed in the future. Use "
                       "'del suffix[key]' instead.")
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

# A list of convenient suffix generators, including:
#   - export_suffix_generator
#       **(used by problem writers)
#   - import_suffix_generator
#       **(used by OptSolver and PyomoModel._load_solution)
#   - local_suffix_generator
#   - suffix_generator

def export_suffix_generator(a_block,
                            datatype=_noarg,
                            active=None,
                            descend_into=True):
    for name, suf in filter(lambda x: (x[1].export_enabled and \
                                       ((datatype is _noarg) or \
                                        (x[1].datatype is datatype))),
                            a_block.components(ctype=suffix._ctype,
                                               return_key=True,
                                               active=active,
                                               descend_into=descend_into)):
        yield name, suf

def import_suffix_generator(a_block,
                            datatype=_noarg,
                            active=None,
                            descend_into=True):
    for name, suf in filter(lambda x: (x[1].import_enabled and \
                                       ((datatype is _noarg) or \
                                        (x[1].datatype is datatype))),
                            a_block.components(ctype=suffix._ctype,
                                               return_key=True,
                                               active=active,
                                               descend_into=descend_into)):
        yield name, suf

def local_suffix_generator(a_block,
                           datatype=_noarg,
                           active=None,
                           descend_into=True):
    for name, suf in filter(lambda x: (x[1].direction is suffix.LOCAL and \
                                       ((datatype is _noarg) or \
                                        (x[1].datatype is datatype))),
                            a_block.components(ctype=suffix._ctype,
                                               return_key=True,
                                               active=active,
                                               descend_into=descend_into)):
        yield name, suf

def suffix_generator(a_block,
                     datatype=_noarg,
                     active=None,
                     descend_into=True):
    for name, suf in filter(lambda x: ((datatype is _noarg) or \
                                       (x[1].datatype is datatype)),
                            a_block.components(ctype=suffix._ctype,
                                               return_key=True,
                                               active=active,
                                               descend_into=descend_into)):
        yield name, suf
