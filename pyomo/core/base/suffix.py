#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('Suffix', 'active_export_suffix_generator', 'active_import_suffix_generator')

import enum
import logging

from pyomo.common.collections import ComponentMap
from pyomo.common.config import In
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import ActiveComponent, ModelComponentFactory
from pyomo.core.base.initializer import Initializer

logger = logging.getLogger('pyomo.core')

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
    _iter = a_block.component_map(Suffix, active=active).items()
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


class SuffixDataType(enum.IntEnum):
    INT = 0
    FLOAT = 4


class SuffixDirection(enum.IntEnum):
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

    @overload
    def __init__(
        self,
        *,
        direction=LOCAL,
        datatype=FLOAT,
        initialize=None,
        rule=None,
        name=None,
        doc=None
    ):
        ...

    def __init__(self, **kwargs):
        # Suffix type information
        self._direction = None
        self._datatype = None
        self._rule = None

        # The suffix direction (note the setter performs error chrcking)
        self.direction = kwargs.pop('direction', Suffix.LOCAL)

        # The suffix datatype (note the setter performs error chrcking)
        self.datatype = kwargs.pop('datatype', Suffix.FLOAT)

        # The suffix construction rule
        # TODO: deprecate the use of 'rule'
        self._rule = Initializer(
            self._pop_from_kwargs('Suffix', kwargs, ('rule', 'initialize'), None),
            treat_sequences_as_mappings=False,
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
            logger.debug("Constructing suffix %s", self.name)

        if self._constructed is True:
            return

        timer = ConstructionTimer(self)
        self._constructed = True

        if self._rule is not None:
            rule = self._rule
            block = self.parent_block()
            if rule.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in rule.indices():
                    self[index] = rule(block, index)
            else:
                self.update_values(rule(block, None))
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
        if direction is not None:
            direction = _SuffixDirectionDomain(direction)
        self._direction = direction

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
        """
        Clears all suffix data.
        """
        self.clear()

    @deprecated(
        'Suffix.set_datatype is replaced with the Suffix.datatype property',
        version='6.7.1.dev0',
    )
    def set_datatype(self, datatype):
        """
        Set the suffix datatype.
        """
        self.datatype = datatype

    @deprecated(
        'Suffix.get_datatype is replaced with the Suffix.datatype property',
        version='6.7.1.dev0',
    )
    def get_datatype(self):
        """
        Return the suffix datatype.
        """
        return self.datatype

    @deprecated(
        'Suffix.set_direction is replaced with the Suffix.direction property',
        version='6.7.1.dev0',
    )
    def set_direction(self, direction):
        """
        Set the suffix direction.
        """
        self.direction = direction

    @deprecated(
        'Suffix.set_direction is replaced with the Suffix.direction property',
        version='6.7.1.dev0',
    )
    def get_direction(self):
        """
        Return the suffix direction.
        """
        return self.direction

    def _pprint(self):
        return (
            [('Direction', str(self._direction)), ('Datatype', str(self._datatype))],
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

    #
    # Override NotImplementedError messages on ComponentMap base class
    #

    def __eq__(self, other):
        """Not implemented."""
        raise NotImplementedError("Suffix components are not comparable")

    def __ne__(self, other):
        """Not implemented."""
        raise NotImplementedError("Suffix components are not comparable")


class SuffixFinder(object):
    def __init__(self, name, default=None):
        """This provides an efficient utility for finding suffix values on a
        (hierarchical) Pyomo model.

        Parameters
        ----------
        name: str

            Name of Suffix to search for.

        default:

            Default value to return from `.find()` if no matching Suffix
            is found.

        """
        self.name = name
        self.default = default
        self.all_suffixes = []
        self._suffixes_by_block = {None: []}

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
        suffixes = self._get_suffix_list(component_data.parent_block())
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
