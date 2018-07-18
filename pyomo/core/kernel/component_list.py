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

from pyomo.core.kernel.component_tuple import ComponentTuple

from six.moves import xrange as range

logger = logging.getLogger('pyomo.core')

class ComponentList(ComponentTuple,
                    collections.MutableSequence):
    """
    A partial implementation of the IComponentContainer
    interface that presents list-like storage functionality.

    Complete implementations need to set the _ctype property
    at the class level, declare the remaining required
    abstract properties of the IComponentContainer base
    class, and declare a slot or attribute named _data.

    Note that this implementation allows nested storage of
    other IComponentContainer implementations that are
    defined with the same ctype.
    """
    __slots__ = ()

    def __init__(self, *args):
        self._data = []
        self._init(args)

    #
    # Define the MutableSequence abstract methods
    #

    def __setitem__(self, i, item):
        if item.ctype == self.ctype:
            if item._parent is None:
                # be sure the current object is properly
                # removed
                logger.warning(
                    "Implicitly replacing the entry %s (type=%s) "
                    "with a new object (type=%s). This is usually "
                    "indicative of a modeling error. To avoid this "
                    "warning, delete the original object from the "
                    "container before assigning a new object."
                    % (self[i].name,
                       self[i].__class__.__name__,
                       item.__class__.__name__))
                self._prepare_for_delete(self._data[i])
                self._prepare_for_add(i, item)
                self._data[i] = item
                return
            elif self._data[i] is item:
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
                "at index %s. A parent container has already been "
                "assigned to the component being inserted: %s"
                % (self.__class__.__name__,
                   self.name,
                   i,
                   item.parent.name))
        else:
            raise TypeError(
                "Invalid assignment to type %s with index %s. "
                "The component being inserted has the wrong "
                "component type: %s"
                % (self.__class__.__name__,
                   i,
                   item.ctype))

    def insert(self, i, item):
        """S.insert(index, object) -- insert object before index"""
        self._insert(i, item)

    def __delitem__(self, i):
        obj = self._data[i]
        self._prepare_for_delete(obj)
        del self._data[i]

    #
    # Override a few default implementations on Sequence
    #
    # We want to avoid issues with parent ownership
    # as we shuffle items

    def reverse(self):
        """S.reverse() -- reverse *IN PLACE*"""
        n = len(self)
        data = self._data
        for i in range(n//2):
            data[i], data[n-i-1] = data[n-i-1], data[i]

def create_component_list(container, type_, size, *args, **kwds):
    """A utility function for constructing a ComponentList
    container of objects with the same initialization data.

    Note that this function bypasses a few safety checks
    when adding the objects into the container, so it should
    only be used in cases where this is okay.

    Args:
        container: The container type. Must be a subclass of
            ComponentList.
        type_: The object type to populate the container
            with. Must have the same ctype as the container
            argument.
        size (int): The number of objects to place in the
            ComponentList.
        *args: arguments used to construct the objects
            placed in the container.
        **kwds: keywords used to construct the objects
            placed in the container.

    Returns:
        A fully populated container.
    """
    assert size >= 0
    assert container.ctype == type_.ctype
    assert issubclass(container, ComponentList)
    clist = container()
    for i in range(size):
        clist._fast_insert(i, type_(*args, **kwds))
    return clist
