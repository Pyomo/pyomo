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

from pyomo.core.kernel.component_interface import \
    (IComponentContainer,
     _SimpleContainerMixin)

from six.moves import xrange as range

class ComponentTuple(_SimpleContainerMixin,
                     IComponentContainer,
                     collections.Sequence):
    """
    A partial implementation of the IComponentContainer
    interface that presents tuple-like storage
    functionality.

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
        self._data = tuple(self._data)

    def _init(self, args):
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError(
                    "%s expected at most 1 arguments, "
                    "got %s" % (self.__class__.__name__,
                                len(args)))
            for item in args[0]:
                self._insert(len(self), item)

    def _fast_insert(self, i, item):
        self._prepare_for_add(i, item)
        self._data.insert(i, item)

    def _insert(self, i, item):
        if item.ctype == self.ctype:
            if item._parent is None:
                self._fast_insert(i, item)
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
        try:
            return self.__getitem__(key)
        except (IndexError, TypeError):
            raise KeyError(str(key))

    def children(self):
        """A generator over the children of this container."""
        return self._data.__iter__()

    #
    # Define the Sequence abstract methods
    #

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return self._data.__len__()

    #
    # Extend the interface to allow for equality comparison
    #
    # We want to avoid generating Pyomo expressions due to
    # comparison of values.

    # Convert both objects to a plain tuple of (type(val),
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
    # Override a few default implementations on Sequence
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

def create_component_tuple(container, type_, size, *args, **kwds):
    """A utility function for constructing a ComponentTuple
    container of objects with the same initialization data.

    Note that this function bypasses a few safety checks
    when adding the objects into the container, so it should
    only be used in cases where this is okay.

    Args:
        container: The container type. Must be a subclass of
            ComponentTuple.
        type_: The object type to populate the container
            with. Must have the same ctype as the container
            argument.
        size (int): The number of objects to place in the
            ComponentTuple.
        *args: arguments used to construct the objects
            placed in the container.
        **kwds: keywords used to construct the objects
            placed in the container.

    Returns:
        A fully populated container.
    """
    assert size >= 0
    assert container.ctype == type_.ctype
    assert issubclass(container, ComponentTuple)
    ctuple = container()
    ctuple._data = []
    for i in range(size):
        ctuple._fast_insert(i, type_(*args, **kwds))
    ctuple._data = tuple(ctuple._data)
    return ctuple
