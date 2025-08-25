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

from collections.abc import MutableSet, Set

from pyomo.common.autoslots import AutoSlots

from ._hasher import hasher


def _rehash_keys(encode, val):
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
        return val
    else:
        # object id() may have changed after unpickling,
        # so we rebuild the dictionary keys
        return {hasher[obj.__class__](obj): obj for obj in val.values()}


class ComponentSet(AutoSlots.Mixin, MutableSet):
    """
    This class is a replacement for set that allows Pyomo
    modeling components to be used as entries. The
    underlying hash is based on the Python id() of the
    object, which gets around the problem of hashing
    subclasses of NumericValue. This class is meant for
    creating sets of Pyomo components. The use of non-Pyomo
    components as entries should be avoided (as the behavior
    is undefined).

    References to objects are kept around as long as they
    are entries in the container, so there is no need to
    worry about id() clashes.

    We also override __setstate__ so that we can rebuild the
    container based on possibly updated object ids after
    a deepcopy or pickle.

    *** An instance of this class should never be
    deepcopied/pickled unless it is done so along with
    its component entries (e.g., as part of a block). ***
    """

    __slots__ = ("_data",)
    __autoslot_mappers__ = {"_data": _rehash_keys}
    # Expose a "public" interface to the global _hasher dict
    hasher = hasher

    def __init__(self, iterable=None):
        # maps id_hash(obj) -> obj
        self._data = {}
        if iterable is not None:
            self.update(iterable)

    def __str__(self):
        """String representation of the mapping."""
        tmp = [f"{v} (key={k})" for k, v in self._data.items()]
        return f"ComponentSet({tmp})"

    def update(self, iterable):
        """Update a set with the union of itself and others."""
        if isinstance(iterable, ComponentSet):
            self._data.update(iterable._data)
        else:
            self._data.update((hasher[val.__class__](val), val) for val in iterable)

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
        _id = hasher[val.__class__](val)
        if _id in self._data:
            del self._data[_id]

    #
    # Overload MutableSet default implementations
    #

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Set):
            return False
        return len(self) == len(other) and all(
            hasher[val.__class__](val) in self._data for val in other
        )

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
        """Remove an element. If not a member, raise a KeyError."""
        try:
            del self._data[hasher[val.__class__](val)]
        except KeyError:
            _id = hasher[val.__class__](val)
            raise KeyError(f"{val} (key={_id})") from None
