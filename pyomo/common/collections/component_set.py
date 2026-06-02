# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from collections.abc import MutableSet, Set
from functools import partial

from pyomo.common.autoslots import AutoSlots
from pyomo.common.formatting import tostr

from ._hasher import hasher


def _rehash_keys(keygen, encode, val):
    if encode:
        # TBD [JDS 2/2024]: if we
        #
        # return list(val.values())
        #
        # here, then we get a strange failure when deepcopying
        # ComponentSets containing an _ImplicitAny domain.  We could
        # track it down to the implementation of
        # autoslots.fast_deepcopy, but couldn't find an obvious bug.
        # There is no error if we just return the original dict, or if
        # we return a tuple(val.values)
        return tuple(val.values())
    else:
        # object id() may have changed after unpickling,
        # so we rebuild the dictionary keys
        return {keygen(obj): obj for obj in val}


class ComponentSet(AutoSlots.Mixin, MutableSet):
    """Set that admits unhashable objects.

    This class is a replacement for :py:`set` that allows Pyomo modeling
    components to be used as entries. The underlying hash is based on
    the Python :py:`id()` of the object, which gets around the problem
    of hashing subclasses of :py:class:`NumericValue`. This class is
    meant for creating sets of Pyomo components.

    References to objects are kept around as long as they
    are entries in the container, so there is no need to
    worry about id() collisions.

    This class leverages :py:class:`AutoSlots` to update any id() keys
    during pickling, restoration, or deepcopying.

    .. warning::

       An instance of this class should never be deepcopied/pickled
       unless it is done so along with its component entries (e.g., as
       part of a block).

    """

    __slots__ = ("_data",)
    __autoslot_mappers__ = {"_data": partial(_rehash_keys, hasher.__call__)}
    # Expose a "public" interface to the global hasher dict (for
    # backwards compatibility)
    hasher = hasher

    def __init__(self, iterable=None):
        # maps id_hash(obj) -> obj
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __str__(self):
        """String representation of the set."""
        tmp = (tostr(k) for k in self._data.values())
        return f"{self.__class__.__name__}({', '.join(tmp)})"

    def update(self, *iterables):
        """Update a set with the union of itself and others."""
        for iterable in *iterables:
            if iterable.__class__ is self.__class__:
                self._data.update(iterable._data)
            else:
                for val in iterable:
                    self.add(val)

    #
    # Implement MutableSet abstract methods
    #

    def __contains__(self, val):
        return hasher[val.__class__](val) in self._data

    def __iter__(self):
        return iter(self._data.values())

    def __len__(self):
        return self._data.__len__()

    def add(self, val):
        """Add an element."""
        self._data[hasher[val.__class__](val)] = val

    def discard(self, val):
        """Remove an element. Do not raise an exception if absent."""
        try:
            del self._data[hasher[val.__class__](val)]
        except KeyError:
            pass

    #
    # Overload MutableSet default implementations
    #

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Set) or len(self._data) != len(other):
            return False
        if other.__class__ is self.__class__:
            return all(key in self._data for key in other._data)
        else:
            return all(map(self.__contains__, other))

    def __ne__(self, other):
        return not (self == other)

    #
    # The remaining MutableSet methods have slow default
    # implementations.
    #

    def clear(self):
        """Remove all elements from this set."""
        self._data.clear()

    def remove(self, val):
        """Remove an element. If not a member, raise a :class:`KeyError`."""
        try:
            del self._data[hasher[val.__class__](val)]
        except KeyError:
            raise KeyError(val)


class ObjectIdSet(ComponentSet):
    """A faster version of :py:class:`ComponentSet`

    :py:class:`ObjectIdSet` is a lighter-weight version of
    :py:class:`ComponentSet`.  By unconditionally using :py:`id()` to
    hash all members, this class performs approximately 50% faster than
    :py:class:`ComponentSet` at the expense of being slightly more
    fragile.

    It is _strongly_ recommended to only store Pyomo components in
    :class:`ObjectIdSet` containers.

    .. warning::

       **DO NOT** store objects that do not return persistent
       :py:func:`id()` values.  In particular, avoid certain immutable
       data types like :class:`tuple` or other immutable objects,
       strings, and long integers.  Doing so may result in failed
       lookups or duplicate entries.

       If you want to mix immutable data types with other unhashable
       objects (like Pyomo :class:`Var` or :class:`Param` components),
       please use :class:`ComponentSet`.

    """

    __slots__ = ()
    __autoslot_mappers__ = {"_data": partial(_rehash_keys, id)}

    def __contains__(self, val):
        return id(val) in self._data

    def add(self, val):
        """Add an element."""
        self._data[id(val)] = val

    def discard(self, val):
        """Remove an element. Do not raise an exception if absent."""
        try:
            del self._data[id(val)]
        except KeyError:
            pass

    def remove(self, val):
        """Remove an element. If not a member, raise a KeyError."""
        try:
            del self._data[id(val)]
        except KeyError:
            raise KeyError(val) from None
