#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ['ComponentMap','Suffix','active_export_suffix_generator','active_import_suffix_generator']

import logging
from collections import MutableMapping
import pprint

from pyomo.core.base.component import ActiveComponent, register_component

from six import iteritems, itervalues

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

def active_export_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            if suffix.export_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            if (suffix.export_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix

def export_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            if suffix.export_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            if (suffix.export_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix

def active_import_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            if suffix.import_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            if (suffix.import_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix

def import_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            if suffix.import_enabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            if (suffix.import_enabled() is True) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix

def active_local_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            if suffix.get_direction() is Suffix.LOCAL:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            if (suffix.get_direction() is Suffix.LOCAL) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix

def local_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            if suffix.get_direction() is Suffix.LOCAL:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            if (suffix.get_direction() is Suffix.LOCAL) and \
               (suffix.get_datatype() is datatype):
                yield name, suffix

def active_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix, active=True)):
            if suffix.get_datatype() is datatype:
                yield name, suffix

def suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            yield name, suffix
    else:
        for name, suffix in iteritems(a_block.component_map(Suffix)):
            if suffix.get_datatype() is datatype:
                yield name, suffix

#
# This class acts as replacement for dict that allows Pyomo modeling
# components (and anything else) to be used as entry keys. The
# underlying mapping is based on the Python id() of the object, so
# that hashing can still take place when the object is mutable,
# unhashable, etc.
#
# A reference to the object is kept around as long as it has a
# corresponding entry in the container so that we don't need to worry
# about id() clashes.
#
# We also override __setstate__ so that we can rebuild the container
# based on possibly updated object id()'s after a deepcopy or
# unpickle.
#
# *** An instance of this class should never be deepcopied/pickled
# unless it is done so along with the components for which it contains
# map entries. ***
#

class ComponentMap(MutableMapping):

    def __init__(self, *args, **kwds):

        # maps id(obj) -> (obj,val)
        self._dict = {}
        # handle the dict-style initialization scenarios
        self.update(*args, **kwds)

    def __setstate__(self, state):
        """
        This method must be defined for deepcopy/pickling because this
        class relies on Python ids.
        """
        id_func = id
        # object id() may have changed after unpickling so we rebuild
        # the dictionary keys
        self._dict = \
            dict((id_func(component), (component,val)) \
                 for component, val in itervalues(state['_dict']))

    def __str__(self):
        """
        String representation of the mapping
        """
        tmp = '{' + \
              (', '.join(component.name+": "+str(val) \
                        for component, val \
                        in itervalues(self._dict))) + \
              '}'
        return tmp

    def pprint(self, stream=None, indent=1, width=80, depth=None, verbose=False):
        """
        Pretty-print a Python object to a stream [default is sys.stdout].
        """
        if verbose:
            tmp = dict((repr(component.name)+" (id="+str(id(component))+")", val)
                           for component, val \
                           in itervalues(self._dict))
        else:
            tmp = dict((repr(component.name)+" (id="+str(id(component))+")", val)
                           for component, val \
                           in itervalues(self._dict))
        pprint.pprint(tmp,
                      stream=stream,
                      indent=indent,
                      width=width,
                      depth=depth)

    #
    # MutableMapping Abstract Methods
    #

    def __getitem__(self, component):
        try:
            return self._dict[id(component)][1]
        except KeyError:
            cname = component.name
            raise KeyError("Component with name: "
                           +cname+
                           " (id=%s)" % id(component))

    def __setitem__(self, component, value):
        self._dict[id(component)] = (component,value)

    def __delitem__(self, component):
        try:
            del self._dict[id(component)]
        except KeyError:
            cname = component.name
            raise KeyError("Component with name: "
                           +cname+
                           " (id=%s)" % id(component))

    def __iter__(self):
        return (component \
                for component, value in \
                itervalues(self._dict))

    def __len__(self):
        return self._dict.__len__()

    #
    # Overload MutableMapping default implementations
    #

    def __eq__(self, other):
        raise NotImplementedError("ComponentMap is not comparable")

    def __ne__(self, other):
        raise NotImplementedError("ComponentMap is not comparable")

    #
    # The remaining methods have slow default implementations
    # for MutableMapping. In particular, they rely KeyError
    # catching, which is slow for this class because KeyError
    # messages use fully qualified names.
    #

    def __contains__(self, component):
        return id(component) in self._dict

    def clear(self):
        'D.clear() -> None.  Remove all items from D.'
        self._dict.clear()

    def get(self, key, default=None):
        'D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.'
        if key in self:
            return self[key]
        return default

    def setdefault(self, key, default=None):
        'D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D'
        if key in self:
            return self[key]
        else:
            self[key] = default
        return default


# Note: The order of inheritance here is important so that
#       __setstate__ works correctly on the ActiveComponent base class.
class Suffix(ComponentMap, ActiveComponent):
    """A model suffix, representing extranious model data"""

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
    LOCAL  = 0
    # sent to solver or other external location
    EXPORT = 1
    # obtained from solver or other external source
    IMPORT = 2
    IMPORT_EXPORT = 3 # both

    SuffixDirections = (LOCAL,EXPORT,IMPORT,IMPORT_EXPORT)
    SuffixDirectionToStr = {LOCAL:'Suffix.LOCAL',
                            EXPORT:'Suffix.EXPORT',
                            IMPORT:'Suffix.IMPORT',
                            IMPORT_EXPORT:'Suffix.IMPORT_EXPORT'}
    # Suffix Datatypes
    FLOAT = 4
    INT = 0
    SuffixDatatypes = (FLOAT,INT,None)
    SuffixDatatypeToStr = {FLOAT:'Suffix.FLOAT',
                           INT:'Suffix.INT',
                           None:str(None)}

    def __init__(self, **kwds):

        # Suffix type information
        self._direction = None
        self._datatype = None
        self._rule = None

        # The suffix direction
        direction = kwds.pop('direction',Suffix.LOCAL)

        # The suffix datatype
        datatype = kwds.pop('datatype',Suffix.FLOAT)

        # The suffix construction rule
        # TODO: deprecate the use of 'rule'
        self._rule = kwds.pop('rule',None)
        self._rule = kwds.pop('initialize',self._rule)

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
        ActiveComponent.__setstate__(self,state)
        ComponentMap.__setstate__(self,state)

    def construct(self, data=None):
        """
        Constructs this component, applying rule if it exists.
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing suffix %s",self.name)

        if self._constructed is True:
            return

        self._constructed = True

        if self._rule is not None:
            self.update_values(self._rule(self._parent()))

    def exportEnabled(self):
        logger.warning("DEPRECATION WARNING: Suffix.exportEnabled is replaced "
                       " with Suffix.export_enabled.")
        return self.export_enabled()

    def export_enabled(self):
        """
        Returns True when this suffix is enabled for export to
        solvers.
        """
        return bool(self._direction & Suffix.EXPORT)

    def importEnabled(self):
        logger.warning("DEPRECATION WARNING: Suffix.importEnabled is replaced "
                       " with Suffix.import_enabled.")
        return self.import_enabled()

    def import_enabled(self):
        """
        Returns True when this suffix is enabled for import from
        solutions.
        """
        return bool(self._direction & Suffix.IMPORT)

    def updateValues(self, data, expand=True):
        logger.warning("DEPRECATION WARNING: Suffix.updateValues is replaced "
                       " with Suffix.update_values.")
        return self.update_values(data,expand)

    def update_values(self, data, expand=True):
        """
        Updates the suffix data given a list of component,value
        tuples. Provides an improvement in efficiency over calling
        set_value on every component.
        """
        if expand:

            try:
                items = iteritems(data)
            except AttributeError:
                items = data

            for component, value in items:
                self.set_value(component, value, expand=expand)

        else:

            # As implemented by MutableMapping
            self.update(data)

    def setValue(self, component, value, expand=True):
        logger.warning("DEPRECATION WARNING: Suffix.setValue is replaced "
                       " with Suffix.set_value.")
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
            for component_ in itervalues(component):
                self[component_] = value
        else:
            self[component] = value

    def setAllValues(self, value):
        logger.warning("DEPRECATION WARNING: Suffix.setAllValues is replaced "
                       " with Suffix.set_all_values.")
        return self.set_all_values(value)

    def set_all_values(self, value):
        """
        Sets the value of this suffix on all components.
        """
        for ndx in self:
            self[ndx] = value

    def clearValue(self, component, expand=True):
        logger.warning("DEPRECATION WARNING: Suffix.clearValue is replaced "
                       " with Suffix.clear_value.")
        return self.clear_value(component, expand)

    def clear_value(self, component, expand=True):
        """
        Clears suffix information for a component.
        """
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

    def clearAllValues(self):
        logger.warning("DEPRECATION WARNING: Suffix.clearAllValues is replaced "
                       " with Suffix.clear_all_values.")
        return self.clear_all_values()

    def clear_all_values(self):
        """
        Clears all suffix data.
        """
        self.clear()

    def setDatatype(self, datatype):
        logger.warning("DEPRECATION WARNING: Suffix.setDatatype is replaced "
                       " with Suffix.set_datatype.")
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

    def getDatatype(self):
        logger.warning("DEPRECATION WARNING: Suffix.getDatatype is replaced "
                       " with Suffix.get_datatype.")
        return self.get_datatype()

    def get_datatype(self):
        """
        Return the suffix datatype.
        """
        return self._datatype

    def setDirection(self, direction):
        logger.warning("DEPRECATION WARNING: Suffix.setDirection is replaced "
                       " with Suffix.set_direction.")
        return self.set_direction(direction)

    def set_direction(self, direction):
        """
        Set the suffix direction.
        """
        if not direction in self.SuffixDirections:
            raise ValueError("Suffix direction must be one of: %s. \n"
                              "Value given: %s"
                             % (list(self.SuffixDirectionToStr.values()),
                                direction))
        self._direction = direction

    def getDirection(self):
        logger.warning("DEPRECATION WARNING: Suffix.getDirection is replaced "
                       " with Suffix.get_direction.")
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
            ( (str(k),v) for k,v in itervalues(self._dict) ),
            ("Value",),
            lambda k,v: [ v ]
            )

    # TODO: delete
    def getValue(self, component, *args):
        """
        Returns the current value of this suffix for the specified
        component.
        """
        logger.warning("DEPRECATION WARNING: Suffix.getValue is replaced "
                       " with the dict-interface method Suffix.get.")
        # As implemented by MutableMapping
        return self.get(component, *args)

    # TODO: delete
    def extractValues(self):
        """
        Extract all data stored on this Suffix into a list of
        component, value tuples.
        """
        logger.warning("DEPRECATION WARNING: Suffix.extractValues() is replaced "
                       " with the dict-interface method Suffix.items().")
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
        raise NotImplementedError("Suffix components are not comparable")

    def __ne__(self, other):
        raise NotImplementedError("Suffix components are not comparable")


register_component(Suffix, "Declare a container for extraneous model data")
