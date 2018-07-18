#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import collections
import logging
try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

from pyomo.core.kernel.component_interface import \
    (IComponentContainer,
     _SimpleContainerMixin)

import six
from six import itervalues, iteritems

logger = logging.getLogger('pyomo.core')

# Note that prior to Python 3, collections.MutableMappping
# is not defined with an empty __slots__
# attribute. Therefore, in Python 2, all implementations of
# this class will have a __dict__ member whether or not they
# declare __slots__. I don't believe it is worth trying to
# code a work around for the Python 2 case as we are moving
# closer to a Python 3-only world and indexed component
# storage containers are not really memory bottlenecks.
class ComponentDict(_SimpleContainerMixin,
                    IComponentContainer,
                    collections.MutableMapping):
    """
    A partial implementation of the IComponentContainer
    interface that presents dict-like storage functionality.

    Complete implementations need to set the _ctype property
    at the class level, declare the remaining required
    abstract properties of the IComponentContainer base
    class, and declare a slot or attribute named _data.

    Note that this implementation allows nested storage of
    other IComponentContainer implementations that are
    defined with the same ctype.

    The optional keyword 'ordered' can be set to :const:`True`/:const:`False`
    to enable/disable the use of an OrderedDict as the
    underlying storage dictionary (default is :const:`True`).
    """
    __slots__ = ()

    def __init__(self, *args, **kwds):
        ordered = kwds.pop('ordered', True)
        if len(kwds):
            raise ValueError("Unexpected keywords used "
                             "to initialize class: %s"
                             % (str(list(kwds.keys()))))
        if ordered:
            self._data = OrderedDict()
        else:
            self._data = {}
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "%s expected at most 1 arguments, "
                    "got %s" % (self.__class__.__name__,
                                len(args)))
            self.update(args[0])

    #
    # Define the IComponentContainer abstract methods
    #

    def child(self, key):
        """Get the child object associated with a given
        storage key for this container.

        Raises:
            KeyError: if the argument is not a storage key
                for any children of this container
        """
        return self[key]

    def children(self):
        """A generator over the children of this container."""
        return itervalues(self._data)

    def _fast_insert(self, key, item):
        self._prepare_for_add(key, item)
        self._data[key] = item

    #
    # Define the MutableMapping abstract methods
    #

    def __setitem__(self, key, item):
        if item.ctype == self.ctype:
            if item._parent is None:
                if key in self._data:
                    logger.warning(
                        "Implicitly replacing the entry %s (type=%s) "
                        "with a new object (type=%s). This is usually "
                        "indicative of a modeling error. To avoid this "
                        "warning, delete the original object from the "
                        "container before assigning a new object."
                        % (self[key].name,
                           self[key].__class__.__name__,
                           item.__class__.__name__))
                    self._prepare_for_delete(
                        self._data[key])
                self._fast_insert(key, item)
                return
            elif (key in self._data) and (self._data[key] is item):
                # a very special case that makes sense to handle
                # because the implied order should be: (1) delete
                # the object at the current index, (2) insert the
                # the new object. This performs both without any
                # actions, but it is an extremely rare case, so
                # it should go last.
                return
            # see note about allowing components to live in more than
            # one container
            raise ValueError(
                "Invalid assignment to %s type with name '%s' "
                "at key %s. A parent container has already been "
                "assigned to the component being inserted: %s"
                % (self.__class__.__name__,
                   self.name,
                   key,
                   item.parent.name))
        else:
            raise TypeError(
                "Invalid assignment to type %s with index %s. "
                "The component being inserted has the wrong "
                "component type: %s"
                % (self.__class__.__name__,
                   key,
                   item.ctype))


    def __delitem__(self, key):
        obj = self._data[key]
        self._prepare_for_delete(obj)
        del self._data[key]

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return self._data.__iter__()

    def __len__(self):
        return self._data.__len__()

    #
    # Override a few default implementations on MutableMapping
    #

    def __contains__(self, key):
        return self._data.__contains__(key)

    # We want to avoid generating Pyomo expressions due to
    # comparison of values, so we convert both objects to a
    # plain dictionary mapping key->(type(val), id(val)) and
    # compare that instead.
    def __eq__(self, other):
        if not isinstance(other, collections.Mapping):
            return False
        return dict((key, (type(val), id(val)))
                    for key, val in self.items()) == \
               dict((key, (type(val), id(val)))
                    for key, val in other.items())

    def __ne__(self, other):
        return not (self == other)

def create_component_dict(container, type_, keys, *args, **kwds):
    """A utility function for constructing a ComponentDict
    container of objects with the same initialization data.

    Note that this function bypasses a few safety checks
    when adding the objects into the container, so it should
    only be used in cases where this is okay.

    Args:
        container: The container type. Must be a subclass of
            ComponentDict.
        type_: The object type to populate the container
            with. Must have the same ctype as the container
            argument.
        keys: The set of keys to used to populate the
            ComponentDict.
        *args: arguments used to construct the objects
            placed in the container.
        **kwds: keywords used to construct the objects
            placed in the container.

    Returns:
        A fully populated container.
    """
    assert container.ctype == type_.ctype
    assert issubclass(container, ComponentDict)
    cdict = container()
    for key in keys:
        cdict._fast_insert(key, type_(*args, **kwds))
    return cdict
