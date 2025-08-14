#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.collections import ComponentMap
from pyomo.common.config import In
from pyomo.common.deprecation import deprecated
from pyomo.common.enums import IntEnum
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.block import BlockData
from pyomo.core.base.component import ActiveComponent, ModelComponentFactory
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import Initializer

logger = logging.getLogger('pyomo.core')

_SUFFIX_API = (
    ('__contains__', 'test membership in'),
    ('__iter__', 'iterate over'),
    '__getitem__',
    '__setitem__',
    'set_value',
    'set_all_values',
    'clear_value',
    'clear_all_values',
    'update_values',
)

# A list of convenient suffix generators, including:
#   - active_export_suffix_generator
#       **(used by problem writers)
#   - export_suffix_generator
#   - active_import_suffix_generator
#       **(used by OptSolver and PyomoModel._load_solution)
#   - import_suffix_generator
#   - active_local_suffix_generator
#   - local_suffix_generator
#   - active_suffix_generator
#   - suffix_generator


def suffix_generator(a_block, datatype=NOTSET, direction=NOTSET, active=None):
    _iter = (
        (s.local_name, s)
        for s in a_block.component_data_objects(
            Suffix, active=active, descend_into=False
        )
    )
    if direction is not NOTSET:
        direction = _SuffixDirectionDomain(direction)
        if not direction:
            _iter = filter(lambda item: item[1].direction == direction, _iter)
        else:
            _iter = filter(lambda item: item[1].direction & direction, _iter)
    if datatype is not NOTSET:
        _iter = filter(lambda item: item[1].datatype == datatype, _iter)
    return _iter


def active_export_suffix_generator(a_block, datatype=NOTSET):
    return suffix_generator(a_block, datatype, SuffixDirection.EXPORT, True)


def export_suffix_generator(a_block, datatype=NOTSET):
    return suffix_generator(a_block, datatype, SuffixDirection.EXPORT)


def active_import_suffix_generator(a_block, datatype=NOTSET):
    return suffix_generator(a_block, datatype, SuffixDirection.IMPORT, True)


def import_suffix_generator(a_block, datatype=NOTSET):
    return suffix_generator(a_block, datatype, SuffixDirection.IMPORT)


def active_local_suffix_generator(a_block, datatype=NOTSET):
    return suffix_generator(a_block, datatype, SuffixDirection.LOCAL, True)


def local_suffix_generator(a_block, datatype=NOTSET):
    return suffix_generator(a_block, datatype, SuffixDirection.LOCAL)


def active_suffix_generator(a_block, datatype=NOTSET):
    return suffix_generator(a_block, datatype, active=True)


class SuffixDataType(IntEnum):
    """Suffix data types

    AMPL only supports two data types for Suffixes: int and float.  The
    numeric values here are specific to the NL file format and should
    not be changed without checking/updating the NL writer.

    """

    INT = 0
    FLOAT = 4


class SuffixDirection(IntEnum):
    """Suffix data flow definition.

    This identifies if the specific Suffix is to be sent to the solver,
    read from the solver output, both, or neither:

    - LOCAL: Suffix is local to Pyomo and should not be sent to or read
      from the solver.

    - EXPORT: Suffix should be sent to the solver as supplemental model
      information.

    - IMPORT: Suffix values will be returned from the solver and should
      be read from the solver output.

    - IMPORT_EXPORT: The Suffix is both an EXPORT and IMPORT suffix.

    """

    LOCAL = 0
    EXPORT = 1
    IMPORT = 2
    IMPORT_EXPORT = 3


_SuffixDataTypeDomain = In(SuffixDataType)
_SuffixDirectionDomain = In(SuffixDirection)


@ModelComponentFactory.register("Declare a container for extraneous model data")
class Suffix(ComponentMap, ActiveComponent):
    """A model suffix, representing extraneous model data"""

    """
    Constructor Arguments:
        direction   The direction of information flow for this suffix.
                        By default, this is LOCAL, indicating that no
                        suffix data is exported or imported.
        datatype    A variable type associated with all values of this
                        suffix.
    """

    #
    # The following local (class) aliases are provided for convenience
    # and backwards compatibility with The Book, 3rd ed
    #

    # Suffix Directions:
    # - neither sent to solver or received from solver
    LOCAL = SuffixDirection.LOCAL
    # - sent to solver or other external location
    EXPORT = SuffixDirection.EXPORT
    # - obtained from solver or other external source
    IMPORT = SuffixDirection.IMPORT
    # - both import and export
    IMPORT_EXPORT = SuffixDirection.IMPORT_EXPORT

    FLOAT = SuffixDataType.FLOAT
    INT = SuffixDataType.INT

    def __new__(cls, *args, **kwargs):
        if cls is not Suffix:
            return super().__new__(cls)
        return super().__new__(AbstractSuffix)

    def __setstate__(self, state):
        super().__setstate__(state)
        # As the concrete class *is* the "Suffix" base class, the normal
        # implementation of deepcopy (through get/setstate) will create
        # the new Suffix, and __new__ will map it to AbstractSuffix.  We
        # need to map constructed Suffixes back to Suffix:
        if self._constructed and self.__class__ is AbstractSuffix:
            self.__class__ = Suffix

    @overload
    def __init__(
        self,
        *,
        direction=LOCAL,
        datatype=FLOAT,
        initialize=None,
        rule=None,
        name=None,
        doc=None,
    ): ...

    def __init__(self, **kwargs):
        # Suffix type information
        self._direction = None
        self._datatype = None
        self._rule = None

        # The suffix direction (note the setter performs error checking)
        self.direction = kwargs.pop('direction', Suffix.LOCAL)

        # The suffix datatype (note the setter performs error checking)
        self.datatype = kwargs.pop('datatype', Suffix.FLOAT)

        # The suffix construction rule
        # TODO: deprecate the use of 'rule'
        self._rule = Initializer(
            self._pop_from_kwargs('Suffix', kwargs, ('rule', 'initialize'), None),
            treat_sequences_as_mappings=False,
            allow_generators=True,
        )

        # Initialize base classes
        kwargs.setdefault('ctype', Suffix)
        ActiveComponent.__init__(self, **kwargs)
        ComponentMap.__init__(self)

        if self._rule is None:
            self.construct()

    def construct(self, data=None):
        """
        Constructs this component, applying rule if it exists.
        """
        if is_debug_set(logger):
            logger.debug(f"Constructing %s '%s'", self.__class__.__name__, self.name)

        if self._constructed is True:
            return

        timer = ConstructionTimer(self)
        self._constructed = True

        if self._rule is not None:
            rule = self._rule
            if rule.contains_indices():
                # The rule contains explicit indices (e.g., is a dict).
                # Iterate over the indices, expand them, and store the
                # result
                block = self.parent_block()
                for index in rule.indices():
                    self.set_value(index, rule(block, index), expand=True)
            else:
                self.update_values(rule(self.parent_block(), None), expand=True)
        timer.report()

    @property
    def datatype(self):
        """Return the suffix datatype."""
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        """Set the suffix datatype."""
        if datatype is not None:
            datatype = _SuffixDataTypeDomain(datatype)
        self._datatype = datatype

    @property
    def direction(self):
        """Return the suffix direction."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set the suffix direction."""
        self._direction = _SuffixDirectionDomain(direction)

    def export_enabled(self):
        """
        Returns True when this suffix is enabled for export to
        solvers.
        """
        return bool(self._direction & Suffix.EXPORT)

    def import_enabled(self):
        """
        Returns True when this suffix is enabled for import from
        solutions.
        """
        return bool(self._direction & Suffix.IMPORT)

    def update_values(self, data, expand=True):
        """
        Updates the suffix data given a list of component,value
        tuples. Provides an improvement in efficiency over calling
        set_value on every component.
        """
        if expand:
            try:
                items = data.items()
            except AttributeError:
                items = data

            for component, value in items:
                self.set_value(component, value, expand=expand)

        else:
            # As implemented by MutableMapping
            self.update(data)

    def set_value(self, component, value, expand=True):
        """
        Sets the value of this suffix on the specified component.

        When expand is True (default), array components are handled by
        storing a reference and value for each index, with no
        reference being stored for the array component itself. When
        expand is False (this is the case for __setitem__), this
        behavior is disabled and a reference to the array component
        itself is kept.
        """
        if expand and component.is_indexed():
            for component_ in component.values():
                self[component_] = value
        else:
            self[component] = value

    def set_all_values(self, value):
        """
        Sets the value of this suffix on all components.
        """
        for ndx in self:
            self[ndx] = value

    def clear_value(self, component, expand=True):
        """
        Clears suffix information for a component.
        """
        if expand and component.is_indexed():
            for component_ in component.values():
                self.pop(component_, None)
        else:
            self.pop(component, None)

    def clear_all_values(self):
        """
        Clears all suffix data.
        """
        self.clear()

    @deprecated(
        'Suffix.set_datatype is replaced with the Suffix.datatype property',
        version='6.7.1',
    )
    def set_datatype(self, datatype):
        """
        Set the suffix datatype.
        """
        self.datatype = datatype

    @deprecated(
        'Suffix.get_datatype is replaced with the Suffix.datatype property',
        version='6.7.1',
    )
    def get_datatype(self):
        """
        Return the suffix datatype.
        """
        return self.datatype

    @deprecated(
        'Suffix.set_direction is replaced with the Suffix.direction property',
        version='6.7.1',
    )
    def set_direction(self, direction):
        """
        Set the suffix direction.
        """
        self.direction = direction

    @deprecated(
        'Suffix.get_direction is replaced with the Suffix.direction property',
        version='6.7.1',
    )
    def get_direction(self):
        """
        Return the suffix direction.
        """
        return self.direction

    def _pprint(self):
        return (
            [
                ('Direction', str(self._direction.name)),
                ('Datatype', getattr(self._datatype, 'name', 'None')),
            ],
            ((str(k), v) for k, v in self._dict.values()),
            ("Value",),
            lambda k, v: [v],
        )

    #
    # Override a few methods to make sure the ActiveComponent versions are
    # called. We can't just switch the inheritance order due to
    # complications with __setstate__
    #

    def pprint(self, *args, **kwds):
        return ActiveComponent.pprint(self, *args, **kwds)

    def __str__(self):
        return ActiveComponent.__str__(self)


@disable_methods(_SUFFIX_API)
class AbstractSuffix(Suffix):
    pass


class SuffixFinder(object):
    def __init__(self, name, default=None, context=None):
        """This provides an efficient utility for finding suffix values on a
        (hierarchical) Pyomo model.

        Parameters
        ----------
        name: str

            Name of Suffix to search for.

        default:

            Default value to return from `.find()` if no matching Suffix
            is found.

        context: BlockData

            The root of the Block hierarchy to use when searching for
            Suffix components.  Suffixes outside this hierarchy will not
            be interrogated and components that are queried (with
            :py:meth:`find(component_data)` will return the default
            value.

        """
        self.name = name
        self.default = default
        self.all_suffixes = []
        self._context = context
        self._suffixes_by_block = ComponentMap()
        self._suffixes_by_block[self._context] = []
        if context is not None:
            s = context.component(name)
            if s is not None and s.ctype is Suffix and s.active:
                self._suffixes_by_block[context].append(s)
                self.all_suffixes.append(s)

    def find(self, component_data):
        """Find suffix value for a given component data object in model tree

        Suffixes are searched by traversing the model hierarchy in three passes:

        1. Search for a Suffix matching the specific component_data,
           starting at the `root` and descending down the tree to
           the component_data.  Return the first match found.
        2. Search for a Suffix matching the component_data's container,
           starting at the `root` and descending down the tree to
           the component_data.  Return the first match found.
        3. Search for a Suffix with key `None`, starting from the
           component_data and working up the tree to the `root`.
           Return the first match found.
        4. Return the default value

        Parameters
        ----------
        component_data: ComponentDataBase

            Component or component data object to find suffix value for.

        Returns
        -------
        The value for Suffix associated with component data if found, else None.

        """
        # Walk parent tree and search for suffixes
        if isinstance(component_data, BlockData):
            _block = component_data
        else:
            _block = component_data.parent_block()
        try:
            suffixes = self._get_suffix_list(_block)
        except AttributeError:
            # Component was outside the context (eventually parent
            # becomes None and parent.parent_block() raises an
            # AttributeError): we will return the default value
            return self.default
        # Pass 1: look for the component_data, working root to leaf
        for s in suffixes:
            if component_data in s:
                return s[component_data]
        # Pass 2: look for the component container, working root to leaf
        parent_comp = component_data.parent_component()
        if parent_comp is not component_data:
            for s in suffixes:
                if parent_comp in s:
                    return s[parent_comp]
        # Pass 3: look for None, working leaf to root
        for s in reversed(suffixes):
            if None in s:
                return s[None]
        return self.default

    def _get_suffix_list(self, parent):
        if parent in self._suffixes_by_block:
            return self._suffixes_by_block[parent]
        suffixes = list(self._get_suffix_list(parent.parent_block()))
        self._suffixes_by_block[parent] = suffixes
        s = parent.component(self.name)
        if s is not None and s.ctype is Suffix and s.active:
            suffixes.append(s)
            self.all_suffixes.append(s)
        return suffixes
