#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

__all__ = ("ComponentList",)

import weakref
import collections

from pyomo.core.base.component_interface import \
    (IComponentContainer,
     _SimpleContainerMixin,
     _abstract_readwrite_property)
from pyomo.core.base.component_map import ComponentMap

class ComponentList(_SimpleContainerMixin,
                    IComponentContainer,
                    collections.MutableSequence):
    """
    A partial implementation of the IComponentContainer
    interface that presents list-like storage functionality.

    Complete implementations need to set the _ctype property
    at the class level, declare the remaining required
    abstract properties of the IComponentContainer base
    class, and declare an slot named _data if using __slots__.

    Note that this implementation allows nested storage of
    other IComponentContainer implementations that are
    defined with the same ctype.
    """
    __slots__ = ()

    def __init__(self, *args):
        self._data = []
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "%s expected at most 1 arguments, "
                    "got %s" % (self.__class__.__name__,
                                len(args)))
            for item in args[0]:
                self.insert(len(self), item)

    #
    # Define the IComponentContainer abstract methods
    #

    def child_key(self, child):
        return self.index(child)

    def child(self, key):
        try:
            return self.__getitem__(key)
        except (IndexError, TypeError):
            raise KeyError(str(key))

    def children(self, return_key=False):
        """Iterate over the children of this container.

        Args:
            return_key (bool): Set to True to indicate that
                the return type should be a 2-tuple
                consisting the child storage key and the
                child object. By default, only the child
                objects are returned.

        Returns: an iterator of objects or (key,object) tuples
        """
        if return_key:
            return enumerate(self._data)
        else:
            return self._data.__iter__()

    #
    # Define the MutableSequence abstract methods
    #

    def __setitem__(self, i, item):
        if item.ctype == self.ctype:
            if item._parent is None:
                # be sure the current object is properly
                # removed
                self._prepare_for_delete(self._data[i])
                self._prepare_for_add(item)
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
        if item.ctype == self.ctype:
            if item._parent is None:
                self._prepare_for_add(item)
                self._data.insert(i, item)
                return
            # see note about allowing components to live
            # in more than one container
            raise ValueError(
                "Invalid assignment to type %s with index %s. "
                "A parent container has already been "
                "assigned to the component being inserted: %s"
                % (self.__class__.__name__,
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

    def __delitem__(self, i):
        obj = self._data[i]
        self._prepare_for_delete(obj)
        del self._data[i]

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return self._data.__len__()

    #
    # Extend the interface to allow for equality comparison
    #
    # We want to avoid generating Pyomo expressions due to
    # comparison of values.

    # Convert both objects to a plain list of (type(val),
    # id(val)) tuples and compare that instead.
    def __eq__(self, other):
        if not isinstance(other, (collections.Set,
                                  collections.Sequence)):
            return False
        return tuple((type(val), id(val))
                     for val in self) == \
               tuple((type(val), id(val))
                     for val in other)

    def __ne__(self, other):
        return not (self == other)


    #
    # Override a few default implementations on MutableSequence
    #
    # We want to avoid generating Pyomo expressions due to
    # comparison of values.

    def __iter__(self):
        return self._data.__iter__()

    def __contains__(self, item):
        item_id = id(item)
        return any(item_id == id(_v) for _v in self._data)

    def index(self, item, start=0, stop=None):
        """S.index(value, [start, [stop]]) -> integer -- return first index of value.

           Raises ValueError if the value is not present.
        """
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)

        i = start
        while stop is None or i < stop:
            try:
                if self[i] is item:
                    return i
            except IndexError:
                break
            i += 1
        raise ValueError

    def count(self, item):
        'S.count(value) -> integer -- return number of occurrences of value'
        item_id = id(item)
        cnt = sum(1 for _v in self._data if id(_v) == item_id)
        assert cnt == 1
        return cnt

    def reverse(self):
        'S.reverse() -- reverse *IN PLACE*'
        n = len(self)
        data = self._data
        for i in range(n//2):
            data[i], data[n-i-1] = data[n-i-1], data[i]
