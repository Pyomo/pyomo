#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.collections import ComponentMap
from pyomo.common.deprecation import deprecated
from pyomo.core.kernel.base import (
    ICategorizedObject, _abstract_readonly_property
)
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.container_utils import (
    define_homogeneous_container_type
)

import six

logger = logging.getLogger('pyomo.core')

_noarg = object()

# Note: ComponentMap is first in the inheritance chain
#       because its __getstate__ / __setstate__ methods
#       contain some special hacks that allow it to be used
#       for the AML-Suffix object as well (hopefully,
#       temporary). As a result, we need to override the
#       __str__ method on this class so that suffix behaves
#       like ICategorizedObject instead of ComponentMap
class ISuffix(ComponentMap,
              ICategorizedObject):
    """The interface for suffixes."""
    __slots__ = ()

    #
    # Implementations can choose to define these
    # properties as using __slots__, __dict__, or
    # by overriding the @property method
    #

    direction = _abstract_readonly_property(
        doc="The suffix direction")
    datatype = _abstract_readonly_property(
        doc="The suffix datatype")

    #
    # Interface
    #

    def __str__(self):
        return ICategorizedObject.__str__(self)

class suffix(ISuffix):
    """A container for storing extraneous model data that
    can be imported to or exported from a solver."""
    _ctype = ISuffix
    __slots__ = ("_parent",
                 "_storage_key",
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
        self._storage_key = None
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
        """Returns :const:`True` when this suffix is enabled
        for export to solvers."""
        return bool(self._direction & suffix.EXPORT)

    @property
    def import_enabled(self):
        """Returns :const:`True` when this suffix is enabled
        for import from solutions."""
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

    @deprecated("suffix.set_all_values will be removed in the future.",
                version='5.3')
    def set_all_values(self, value):
        for ndx in self:
            self[ndx] = value

    @deprecated("suffix.clear_value will be removed in the future. "
                "Use 'del suffix[key]' instead.", version='5.3')
    def clear_value(self, component):
        try:
            del self[component]
        except KeyError:
            pass

    @deprecated("suffix.clear_all_values is replaced with suffix.clear",
                version='5.3')
    def clear_all_values(self):
        self.clear()

    @deprecated("suffix.get_datatype is replaced with the property "
                "suffix.datatype", version='5.3')
    def get_datatype(self):
        return self.datatype

    @deprecated("suffix.set_datatype is replaced with the property "
                "setter suffix.datatype", version='5.3')
    def set_datatype(self, datatype):
        self.datatype = datatype

    @deprecated("suffix.get_direction is replaced with the property "
                "suffix.direction", version='5.3')
    def get_direction(self):
        return self.direction

    @deprecated("suffix.set_direction is replaced with the property "
                "setter suffix.direction", version='5.3')
    def set_direction(self, direction):
        self.direction = direction

# A list of convenient suffix generators, including:
#   - export_suffix_generator
#       **(used by problem writers)
#   - import_suffix_generator
#       **(used by OptSolver and PyomoModel._load_solution)
#   - local_suffix_generator
#   - suffix_generator

def export_suffix_generator(blk,
                            datatype=_noarg,
                            active=True,
                            descend_into=True):
    """
    Generates an efficient traversal of all suffixes that
    have been declared for exporting data.

    Args:
        blk: A block object.
        datatype: Restricts the suffixes included in the
            returned generator to those matching the
            provided suffix datatype.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend_into (bool, function): Indicates whether or
            not to descend into a heterogeneous
            container. Default is True, which is equivalent
            to `lambda x: True`, meaning all heterogeneous
            containers will be descended into.

    Returns:
        iterator of suffixes
    """
    for suf in filter(lambda x: (x.export_enabled and \
                                 ((datatype is _noarg) or \
                                  (x.datatype is datatype))),
                      blk.components(ctype=suffix._ctype,
                                     active=active,
                                     descend_into=descend_into)):
        yield suf

def import_suffix_generator(blk,
                            datatype=_noarg,
                            active=True,
                            descend_into=True):
    """
    Generates an efficient traversal of all suffixes that
    have been declared for importing data.

    Args:
        blk: A block object.
        datatype: Restricts the suffixes included in the
            returned generator to those matching the
            provided suffix datatype.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend_into (bool, function): Indicates whether or
            not to descend into a heterogeneous
            container. Default is True, which is equivalent
            to `lambda x: True`, meaning all heterogeneous
            containers will be descended into.

    Returns:
        iterator of suffixes
    """
    for suf in filter(lambda x: (x.import_enabled and \
                                 ((datatype is _noarg) or \
                                  (x.datatype is datatype))),
                      blk.components(ctype=suffix._ctype,
                                     active=active,
                                     descend_into=descend_into)):
        yield suf

def local_suffix_generator(blk,
                           datatype=_noarg,
                           active=True,
                           descend_into=True):
    """
    Generates an efficient traversal of all suffixes that
    have been declared local data storage.

    Args:
        blk: A block object.
        datatype: Restricts the suffixes included in the
            returned generator to those matching the
            provided suffix datatype.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend_into (bool, function): Indicates whether or
            not to descend into a heterogeneous
            container. Default is True, which is equivalent
            to `lambda x: True`, meaning all heterogeneous
            containers will be descended into.

    Returns:
        iterator of suffixes
    """
    for suf in filter(lambda x: (x.direction is suffix.LOCAL and \
                                 ((datatype is _noarg) or \
                                  (x.datatype is datatype))),
                      blk.components(ctype=suffix._ctype,
                                     active=active,
                                     descend_into=descend_into)):
        yield suf

def suffix_generator(blk,
                     datatype=_noarg,
                     active=True,
                     descend_into=True):
    """
    Generates an efficient traversal of all suffixes that
    have been declared.

    Args:
        blk: A block object.
        datatype: Restricts the suffixes included in the
            returned generator to those matching the
            provided suffix datatype.
        active (:const:`True`/:const:`None`): Controls
            whether or not to filter the iteration to
            include only the active part of the storage
            tree. The default is :const:`True`. Setting this
            keyword to :const:`None` causes the active
            status of objects to be ignored.
        descend_into (bool, function): Indicates whether or
            not to descend into a heterogeneous
            container. Default is True, which is equivalent
            to `lambda x: True`, meaning all heterogeneous
            containers will be descended into.

    Returns:
        iterator of suffixes
    """
    for suf in filter(lambda x: ((datatype is _noarg) or \
                                 (x.datatype is datatype)),
                      blk.components(ctype=suffix._ctype,
                                     active=active,
                                     descend_into=descend_into)):
        yield suf

# inserts class definition for simple a
# simple suffix_dict into this module
define_homogeneous_container_type(
    globals(),
    "suffix_dict",
    DictContainer,
    ISuffix,
    doc=("A dict-style container for objects "
         "with category type "+ISuffix.__name__),
    use_slots=True)
