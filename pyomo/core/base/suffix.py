#  _________________________________________________________________________
#
#  Pyomo: A COmmon Optimization Python Repository
#  Copyright (c) 2008 Sandia Corporation.
#  This software is distributed under the BSD License.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  For more information, see the Pyomo README.txt file.
#  _________________________________________________________________________

__all__ = ['ComponentMap','Suffix','active_export_suffix_generator','active_import_suffix_generator']

import sys
import logging
from collections import MutableMapping
import pprint

from six import iteritems, iterkeys, itervalues

from pyomo.core.base.component import Component, register_component

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
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            if suffix.exportEnabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            if (suffix.exportEnabled() is True) and \
               (suffix.getDatatype() is datatype):
                yield name, suffix

def export_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.components(Suffix)):
            if suffix.exportEnabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.components(Suffix)):
            if (suffix.exportEnabled() is True) and \
               (suffix.getDatatype() is datatype):
                yield name, suffix

def active_import_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            if suffix.importEnabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            if (suffix.importEnabled() is True) and \
               (suffix.getDatatype() is datatype):
                yield name, suffix

def import_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.components(Suffix)):
            if suffix.importEnabled() is True:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.components(Suffix)):
            if (suffix.importEnabled() is True) and \
               (suffix.getDatatype() is datatype):
                yield name, suffix

def active_local_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            if suffix.getDirection() is Suffix.LOCAL:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            if (suffix.getDirection() is Suffix.LOCAL) and \
               (suffix.getDatatype() is datatype):
                yield name, suffix

def local_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.components(Suffix)):
            if suffix.getDirection() is Suffix.LOCAL:
                yield name, suffix
    else:
        for name, suffix in iteritems(a_block.components(Suffix)):
            if (suffix.getDirection() is Suffix.LOCAL) and \
               (suffix.getDatatype() is datatype):
                yield name, suffix

def active_suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            yield name, suffix
    else:
        for name, suffix in iteritems(a_block.active_components(Suffix)):
            if suffix.getDatatype() is datatype:
                yield name, suffix

def suffix_generator(a_block,datatype=False):
    if (datatype is False):
        for name, suffix in iteritems(a_block.components(Suffix)):
            yield name, suffix
    else:
        for name, suffix in iteritems(a_block.components(Suffix)):
            if suffix.getDatatype() is datatype:
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
              (', '.join(str(component)+": "+str(val) \
                        for component, val \
                        in itervalues(self._dict))) + \
              '}'
        return tmp

    def pprint(self, stream=None, indent=1, width=80, depth=None, verbose=False):
        """
        Pretty-print a Python object to a stream [default is sys.stdout].
        """
        if verbose:
            tmp = dict((repr(str(component))+" (id="+str(id(component))+")", val)
                           for component, val \
                           in itervalues(self._dict))
        else:
            tmp = dict((repr(str(component))+" (id="+str(id(component))+")", val)
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
            raise KeyError("Component with name: "
                           +str(component)+
                           " (id=%s)" % id(component))

    def __setitem__(self, component, value):
        self._dict[id(component)] = (component,value)

    def __delitem__(self, component):
        try:
            del self._dict[id(component)]
        except KeyError:
            raise KeyError("Component with name: "
                           +str(component)+
                           " (id=%s)" % id(component))

    def __iter__(self):
        return (component \
                for component, value in \
                itervalues(self._dict))

    def __len__(self):
        return self._dict.__len__()

    def __contains__(self, component):
        return id(component) in self._dict

    def __eq__(self, other):
        raise NotImplementedError("ComponentMap is not comparable")

    def __ne__(self, other):
        raise NotImplementedError("ComponentMap is not comparable")

# Note: The order of inheritance here is important so that
#       __setstate__ works correctly on the Component base class.
class Suffix(ComponentMap, Component):
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

    def __init__(self, *args, **kwds):

        # Suffix type information
        self._direction = None
        self._datatype = None
        self._rule = None

        # The suffix direction
        direction = kwds.pop('direction',Suffix.LOCAL)

        # The suffix datatype
        datatype = kwds.pop('datatype',Suffix.FLOAT)

        # The suffix construction rule
        self._rule = kwds.pop('rule',None)

        # Check that keyword values make sense (these function have
        # internal error checking).
        self.setDirection(direction)
        self.setDatatype(datatype)

        # Initialize base classes
        kwds.setdefault('ctype', Suffix)
        Component.__init__(self, *args, **kwds)
        ComponentMap.__init__(self)

        if self._rule is None:
            self.construct()

    def __setstate__(self, state):
        """
        This method must be defined for deepcopy/pickling because this
        class relies on component ids.
        """
        Component.__setstate__(self,state)
        ComponentMap.__setstate__(self,state)

    def reset(self):
        """
        Reconstructs this component by clearing all values and
        re-calling construction rule if it exists.
        """
        self.clearAllValues()
        self._constructed = False
        self.construct()

    def construct(self, data=None):
        """
        Constructs this component, applying rule if it exists.
        """
        if __debug__ and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Constructing suffix %s",self.cname())

        if self._constructed is True:
            return

        self._constructed = True

        if self._rule is not None:
            self.updateValues(self._rule(self._parent()))

    def exportEnabled(self):
        """
        Returns True when this suffix is enabled for export to
        solvers.
        """
        return bool(self._direction & Suffix.EXPORT)

    def importEnabled(self):
        """
        Returns True when this suffix is enabled for import from
        solutions.
        """
        return bool(self._direction & Suffix.IMPORT)

    def updateValues(self, data, expand=True):
        """
        Updates the suffix data given a list of component,value
        tuples. Provides an improvement in efficiency over calling
        setValue on every component.
        """
        if expand:

            try:
                items = iteritems(data)
            except AttributeError:
                items = data

            for component, value in items:
                self.setValue(component, value, expand=expand)

        else:

            # As implemented by MutableMapping
            self.update(data)

    def setValue(self, component, value, expand=True):
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
        """
        Sets the value of this suffix on all components.
        """
        for ndx in self:
            self[ndx] = value

    def clearValue(self, component, expand=True):
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
        """
        Clears all suffix data.
        """
        # As implemented by MutableMapping
        self.clear()

    def setDatatype(self, datatype):
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
        """
        Return the suffix datatype.
        """
        return self._datatype

    def setDirection(self, direction):
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
        """
        Return the suffix direction.
        """
        return self._direction

    def __str__(self):
        """
        Return a string representation of the suffix.  If the name
        attribute is None, then return ''
        """
        if self.cname() is None:
            return ""
        else:
            return self.cname()

    def _pprint(self):
        return (
            [('Direction', self.SuffixDirectionToStr[self._direction]),
             ('Datatype', self.SuffixDatatypeToStr[self._datatype]),
             ],
            ( (str(k),v) for k,v in itervalues(self._dict) ),
            ("Key","Value"),
            lambda k,v: [ k, v ]
            )

    # TODO: delete
    def getValue(self, component, *args):
        """
        Returns the current value of this suffix for the specified
        component.
        """
        logger.warn("DEPRECATION WARNING: Suffix.getValue is replaced "
                    " with the dict-interface method Suffix.get.")
        # As implemented by MutableMapping
        return self.get(component, *args)

    # TODO: delete
    def extractValues(self):
        """
        Extract all data stored on this Suffix into a list of
        component, value tuples.
        """
        logger.warn("DEPRECATION WARNING: Suffix.extractValues() is replaced "
                    " with the dict-interface method Suffix.items().")
        # As implemented by MutableMapping
        return list(self.items())

    #
    # Override a few methods to make sure the Component versions are
    # called. We can't just switch the inheritance order due to
    # complications with __setstate__
    #

    def pprint(self, *args, **kwds):
        return Component.pprint(self, *args, **kwds)

    def __str__(self):
        return Component.__str__(self)

    #
    # Override NotImplementedError messages on ComponentMap base class
    #

    def __eq__(self, other):
        raise NotImplementedError("Suffix components are not comparable")

    def __ne__(self, other):
        raise NotImplementedError("Suffix components are not comparable")

register_component(Suffix, "Declare a container for extraneous model data")
