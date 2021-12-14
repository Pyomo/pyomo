#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ('Suffix',
           'active_export_suffix_generator',
           'active_import_suffix_generator')

import logging
from typing import overload

from pyomo.common.collections import ComponentMap
from pyomo.common.log import is_debug_set
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import ActiveComponent, ModelComponentFactory

from pyomo.common.deprecation import deprecated

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


def active_export_suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            if suffix.export_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            if (suffix.export_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix


def export_suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix).items():
            if suffix.export_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix).items():
            if (suffix.export_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix


def active_import_suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            if suffix.import_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            if (suffix.import_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix


def import_suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix).items():
            if suffix.import_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix).items():
            if (suffix.import_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix


def active_local_suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            if suffix.get_direction() is Suffix.LOCAL:
                yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            if (suffix.get_direction() is Suffix.LOCAL) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix


def local_suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix).items():
            if suffix.get_direction() is Suffix.LOCAL:
                yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix).items():
            if (suffix.get_direction() is Suffix.LOCAL) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix


def active_suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix, active=True).items():
            if suffix.get_datatype() is datatype:
                yield name, suffix


def suffix_generator(a_block, datatype=False):
    if (datatype is False):
        for name, suffix in a_block.component_map(Suffix).items():
            yield name, suffix
    else:
        for name, suffix in a_block.component_map(Suffix).items():
            if suffix.get_datatype() is datatype:
                yield name, suffix

# Note: The order of inheritance here is important so that
#       __setstate__ works correctly on the ActiveComponent base class.


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
    # If more directions are added be sure to update the error message
    # in the setDirection method
    # neither sent to solver or received from solver
    LOCAL = 0
    # sent to solver or other external location
    EXPORT = 1
    # obtained from solver or other external source
    IMPORT = 2
    IMPORT_EXPORT = 3  # both

    SuffixDirections = (LOCAL, EXPORT, IMPORT, IMPORT_EXPORT)
    SuffixDirectionToStr = {LOCAL: 'Suffix.LOCAL',
                            EXPORT: 'Suffix.EXPORT',
                            IMPORT: 'Suffix.IMPORT',
                            IMPORT_EXPORT: 'Suffix.IMPORT_EXPORT'}
    # Suffix Datatypes
    FLOAT = 4
    INT = 0
    SuffixDatatypes = (FLOAT, INT, None)
    SuffixDatatypeToStr = {FLOAT: 'Suffix.FLOAT',
                           INT: 'Suffix.INT',
                           None: str(None)}

    @overload
    def __init__(self, *, direction=LOCAL, datatype=FLOAT,
                 initialize=None, rule=None, name=None, doc=None): ...

    def __init__(self, **kwds):

        # Suffix type information
        self._direction = None
        self._datatype = None
        self._rule = None

        # The suffix direction
        direction = kwds.pop('direction', Suffix.LOCAL)

        # The suffix datatype
        datatype = kwds.pop('datatype', Suffix.FLOAT)

        # The suffix construction rule
        # TODO: deprecate the use of 'rule'
        self._rule = kwds.pop('rule', None)
        self._rule = kwds.pop('initialize', self._rule)

        # Check that keyword values make sense (these function have
        # internal error checking).
        self.set_direction(direction)
        self.set_datatype(datatype)

        # Initialize base classes
        kwds.setdefault('ctype', Suffix)
        ActiveComponent.__init__(self, **kwds)
        ComponentMap.__init__(self)

        if self._rule is None:
            self.construct()

    def __setstate__(self, state):
        """
        This method must be defined for deepcopy/pickling because this
        class relies on component ids.
        """
        ActiveComponent.__setstate__(self, state)
        ComponentMap.__setstate__(self, state)

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
            self.update_values(self._rule(self._parent()))
        timer.report()

    @deprecated('Suffix.exportEnabled is replaced with Suffix.export_enabled.',
                version='4.1.10486')
    def exportEnabled(self):
        return self.export_enabled()

    def export_enabled(self):
        """
        Returns True when this suffix is enabled for export to
        solvers.
        """
        return bool(self._direction & Suffix.EXPORT)

    @deprecated('Suffix.importEnabled is replaced with Suffix.import_enabled.',
                version='4.1.10486')
    def importEnabled(self):
        return self.import_enabled()

    def import_enabled(self):
        """
        Returns True when this suffix is enabled for import from
        solutions.
        """
        return bool(self._direction & Suffix.IMPORT)

    @deprecated('Suffix.updateValues is replaced with Suffix.update_values.',
                version='4.1.10486')
    def updateValues(self, data, expand=True):
        return self.update_values(data, expand)

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

    @deprecated('Suffix.setValue is replaced with Suffix.set_value.',
                version='4.1.10486')
    def setValue(self, component, value, expand=True):
        return self.set_value(component, value, expand)

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

    @deprecated('Suffix.setAllValues is replaced with Suffix.set_all_values.',
                version='4.1.10486')
    def setAllValues(self, value):
        return self.set_all_values(value)

    def set_all_values(self, value):
        """
        Sets the value of this suffix on all components.
        """
        for ndx in self:
            self[ndx] = value

    @deprecated('Suffix.clearValue is replaced with Suffix.clear_value.',
                version='4.1.10486')
    def clearValue(self, component, expand=True):
        return self.clear_value(component, expand)

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

    @deprecated('Suffix.clearAllValues is replaced with '
                'Suffix.clear_all_values.',
                version='4.1.10486')
    def clearAllValues(self):
        return self.clear_all_values()

    def clear_all_values(self):
        """
        Clears all suffix data.
        """
        self.clear()

    @deprecated('Suffix.setDatatype is replaced with Suffix.set_datatype.',
                version='4.1.10486')
    def setDatatype(self, datatype):
        return self.set_datatype(datatype)

    def set_datatype(self, datatype):
        """
        Set the suffix datatype.
        """
        if datatype not in self.SuffixDatatypes:
            raise ValueError("Suffix datatype must be one of: %s. \n"
                             "Value given: %s"
                             % (list(Suffix.SuffixDatatypeToStr.values()),
                                datatype))
        self._datatype = datatype

    @deprecated('Suffix.getDatatype is replaced with Suffix.get_datatype.',
                version='4.1.10486')
    def getDatatype(self):
        return self.get_datatype()

    def get_datatype(self):
        """
        Return the suffix datatype.
        """
        return self._datatype

    @deprecated('Suffix.setDirection is replaced with Suffix.set_direction.',
                version='4.1.10486')
    def setDirection(self, direction):
        return self.set_direction(direction)

    def set_direction(self, direction):
        """
        Set the suffix direction.
        """
        if direction not in self.SuffixDirections:
            raise ValueError("Suffix direction must be one of: %s. \n"
                             "Value given: %s"
                             % (list(self.SuffixDirectionToStr.values()),
                                direction))
        self._direction = direction

    @deprecated('Suffix.getDirection is replaced with Suffix.get_direction.',
                version='4.1.10486')
    def getDirection(self):
        return self.get_direction()

    def get_direction(self):
        """
        Return the suffix direction.
        """
        return self._direction

    def __str__(self):
        """
        Return a string representation of the suffix.  If the name
        attribute is None, then return ''
        """
        name = self.name
        if name is None:
            return ''
        return name

    def _pprint(self):
        return (
            [('Direction', self.SuffixDirectionToStr[self._direction]),
             ('Datatype', self.SuffixDatatypeToStr[self._datatype]),
             ],
            ((str(k), v) for k, v in self._dict.values()),
            ("Value",),
            lambda k, v: [v]
        )

    # TODO: delete
    @deprecated('Suffix.getValue is replaced with '
                'the dict-interface method Suffix.get.',
                version='4.1.10486')
    def getValue(self, component, *args):
        """
        Returns the current value of this suffix for the specified
        component.
        """
        # As implemented by MutableMapping
        return self.get(component, *args)

    # TODO: delete
    @deprecated('Suffix.extractValues() is replaced with '
                'the dict-interface method Suffix.items().',
                version='4.1.10486')
    def extractValues(self):
        """
        Extract all data stored on this Suffix into a list of
        component, value tuples.
        """
        # As implemented by MutableMapping
        return list(self.items())

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

