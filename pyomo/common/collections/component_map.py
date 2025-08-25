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

from collections.abc import Mapping, MutableMapping

from pyomo.common.autoslots import AutoSlots

from ._hasher import hasher


def _rehash_keys(encode, val):
    if encode:
        return val
    else:
        # object id() may have changed after unpickling,
        # so we rebuild the dictionary keys
        return {hasher[obj.__class__](obj): (obj, v) for obj, v in val.values()}


class ComponentMap(AutoSlots.Mixin, MutableMapping):
    """
    This class is a replacement for dict that allows Pyomo
    modeling components to be used as entry keys. The
    underlying mapping is based on the Python id() of the
    object, which gets around the problem of hashing
    subclasses of NumericValue. This class is meant for
    creating mappings from Pyomo components to values. The
    use of non-Pyomo components as entry keys should be
    avoided.

    A reference to the object is kept around as long as it
    has a corresponding entry in the container, so there is
    no need to worry about id() clashes.

    We also override __setstate__ so that we can rebuild the
    container based on possibly updated object ids after
    a deepcopy or pickle.

    *** An instance of this class should never be
    deepcopied/pickled unless it is done so along with the
    components for which it contains map entries (e.g., as
    part of a block). ***
    """

    __slots__ = ("_dict",)
    __autoslot_mappers__ = {"_dict": _rehash_keys}
    # Expose a "public" interface to the global _hasher dict
    hasher = hasher

    def __init__(self, *args, **kwds):
        # maps id_hash(obj) -> (obj,val)
        self._dict = {}
        # handle the dict-style initialization scenarios
        self.update(*args, **kwds)

    def __str__(self):
        """String representation of the mapping."""
        tmp = {f"{v[0]} (key={k})": v[1] for k, v in self._dict.items()}
        return f"ComponentMap({tmp})"

    #
    # Implement MutableMapping abstract methods
    #

    def __getitem__(self, obj):
        try:
            return self._dict[hasher[obj.__class__](obj)][1]
        except KeyError:
            _id = hasher[obj.__class__](obj)
            raise KeyError(f"{obj} (key={_id})") from None

    def __setitem__(self, obj, val):
        self._dict[hasher[obj.__class__](obj)] = (obj, val)

    def __delitem__(self, obj):
        try:
            del self._dict[hasher[obj.__class__](obj)]
        except KeyError:
            _id = hasher[obj.__class__](obj)
            raise KeyError(f"{obj} (key={_id})") from None

    def __iter__(self):
        return (obj for obj, val in self._dict.values())

    def __len__(self):
        return self._dict.__len__()

    #
    # Overload MutableMapping default implementations
    #

    # We want a specialization of update() to avoid unnecessary calls to
    # id() when copying / merging ComponentMaps
    def update(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and isinstance(args[0], ComponentMap):
            return self._dict.update(args[0]._dict)
        return super().update(*args, **kwargs)

    # We want to avoid generating Pyomo expressions due to comparing the
    # keys, so look up each entry from other in this dict.
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        # Note we have already verified the dicts are the same size
        for key, val in other.items():
            other_id = hasher[key.__class__](key)
            if other_id not in self._dict:
                return False
            self_val = self._dict[other_id][1]
            # Note: check "is" first to help avoid creation of Pyomo
            # expressions (for the case that the values contain the same
            # Pyomo component)
            if self_val is not val and self_val != val:
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    #
    # The remaining methods have slow default
    # implementations for MutableMapping. In particular,
    # they rely KeyError catching, which is slow for this
    # class because KeyError messages use fully qualified
    # names.
    #

    def __contains__(self, obj):
        return hasher[obj.__class__](obj) in self._dict

    def clear(self):
        "D.clear() -> None.  Remove all items from D."
        self._dict.clear()

    def get(self, key, default=None):
        "D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."
        if key in self:
            return self[key]
        return default

    def setdefault(self, key, default=None):
        "D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D"
        if key in self:
            return self[key]
        else:
            self[key] = default
        return default


class DefaultComponentMap(ComponentMap):
    """A :py:class:`defaultdict` admitting Pyomo Components as keys

    This class is a replacement for defaultdict that allows Pyomo
    modeling components to be used as entry keys. The base
    implementation builds on :py:class:`ComponentMap`.

    """

    __slots__ = ("default_factory",)

    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = default_factory

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = ans = self.default_factory()
        return ans

    def __getitem__(self, obj):
        _key = hasher[obj.__class__](obj)
        if _key in self._dict:
            return self._dict[_key][1]
        else:
            return self.__missing__(obj)
