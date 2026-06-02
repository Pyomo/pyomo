# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from collections.abc import Set, Mapping, MutableMapping
from functools import partial
from operator import itemgetter

from pyomo.common.autoslots import AutoSlots
from pyomo.common.formatting import tostr

from ._hasher import hasher


def _rehash_keys(keygen, encode, val):
    if encode:
        return list(val.values())
    else:
        # object id() may have changed after unpickling,
        # so we rebuild the dictionary keys
        return {keygen(v[0]): v for v in val}


class ComponentMap_keys(Set):
    __slots__ = ('_cm',)

    def __init__(self, cm):
        self._cm = cm

    def __iter__(self):
        return iter(map(itemgetter(0), self._cm._dict.values()))

    def __contains__(self, key):
        return self._cm.__contains__(key)

    def __len__(self):
        return self._cm.__len__()


class ComponentMap_items(Set):
    __slots__ = ('_cm',)

    def __init__(self, cm):
        self._cm = cm

    def __iter__(self):
        return iter(self._cm._dict.values())

    def __contains__(self, item):
        try:
            key, val = item
        except (TypeError, ValueError):
            return False
        return key in self._cm and self._cm._value_eq(val, self._cm[key])

    def __len__(self):
        return self._cm.__len__()


class ComponentMap_values(Set):
    __slots__ = ('_cm',)

    def __init__(self, cm):
        self._cm = cm

    def __iter__(self):
        return iter(map(itemgetter(1), self._cm._dict.values()))

    def __contains__(self, val):
        """Returns True if `val` appears as a value in this ComponentMap

        .. warining::

           This method is provided for API compatibility and is NOT
           efficient (it is a linear scan through the underlying
           `dict`).  We *do not* recommend using it in large
           applications of when performance matters.

        """
        return any(self._cm._value_eq(v, val) for v in self.__iter__())

    def __len__(self):
        return self._cm.__len__()


class ComponentMap(AutoSlots.Mixin, MutableMapping):
    """Mapping that admits unhashable objects as keys

    This class is a replacement for :py:`dict` that allows Pyomo
    modeling components to be used as keys. The underlying mapping is
    based on the Python :py:`id()` of the object, which gets around the
    problem of hashing subclasses of :py:class:`NumericValue`. This
    class is meant for creating mappings from Pyomo components to
    values.

    A reference to the object is kept around as long as it
    has a corresponding entry in the container, so there is
    no need to worry about id() collisions.

    This class leverages :py:class:`AutoSlots` to update any id() keys
    during pickling, restoration, or deepcopying.

    .. warning::

       An instance of this class should never be deepcopied/pickled
       unless it is done so along with its component entries (e.g., as
       part of a block).

    """

    __slots__ = ("_dict",)
    __autoslot_mappers__ = {"_dict": partial(_rehash_keys, hasher.__call__)}
    # Expose a "public" interface to the global hasher dict (for
    # backwards compatibility)
    hasher = hasher

    def __init__(self, *args, **kwargs):
        # maps id_hash(obj) -> (obj,val)
        self._dict = {}
        # handle the dict-style initialization scenarios
        if args or kwargs:
            self.update(*args, **kwargs)

    def __str__(self):
        """String representation of the mapping."""
        tmp = ', '.join(f"{tostr(v[0])}: {tostr(v[1])}" for v in self._dict.values())
        return f"{self.__class__.__name__}({tmp})"

    #
    # Implement MutableMapping abstract methods
    #

    def __getitem__(self, obj):
        try:
            return self._dict[hasher[obj.__class__](obj)][1]
        except KeyError:
            raise KeyError(obj) from None

    def __setitem__(self, obj, val):
        self._dict[hasher[obj.__class__](obj)] = (obj, val)

    def __delitem__(self, obj):
        try:
            del self._dict[hasher[obj.__class__](obj)]
        except KeyError:
            raise KeyError(obj) from None

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return self._dict.__len__()

    #
    # Overload MutableMapping default implementations
    #

    # We want a specialization of update() to avoid unnecessary calls to
    # the hasher when copying / merging ComponentMaps
    def update(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and args[0].__class__ is self.__class__:
            return self._dict.update(args[0]._dict)
        return super().update(*args, **kwargs)

    def _rekey_items(self, items):
        """Utility method for mapping key-value pairs into local hash keys"""
        return ((hasher[key.__class__](key), val) for key, val in items)

    @staticmethod
    def _value_eq(a, b):
        # Note: check "is" first to help avoid creation of Pyomo
        # expressions (for the case that the values contain the same
        # Pyomo component)
        if a is b:
            return True
        diff = a != b
        return not diff if diff.__class__ is bool else False

    # We want to avoid generating Pyomo expressions due to comparing the
    # keys, so look up each entry from other in this dict.
    def __eq__(self, other):
        """Return self==other."""
        if self is other:
            return True
        if not isinstance(other, Mapping) or len(self) != len(other):
            return False
        # Note we have already verified the dicts are the same size
        if other.__class__ is self.__class__:
            # shortcut for comparing ComponentMaps to each other: avoid
            # regenerating any keys
            other_items = ((key, val[1]) for key, val in other._dict.items())
        else:
            other_items = self._rekey_items(other.items())

        _dict = self._dict
        _eq = self._value_eq
        return all(key in _dict and _eq(val, _dict[key][1]) for key, val in other_items)

    def __ne__(self, other):
        """Return self!=other."""
        return not self.__eq__(other)

    #
    # The remaining methods have slow default implementations
    #

    def keys(self):
        return ComponentMap_keys(self)

    def values(self):
        return ComponentMap_values(self)

    def items(self):
        return ComponentMap_items(self)

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
        if default_factory is not None and not callable(default_factory):
            args = (default_factory,) + args
            default_factory = None
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


class ObjectIdMap(ComponentMap):
    """A faster version of :py:class:`ComponentMap`

    :py:class:`ObjectIdMap` is a lighter-weight version of
    :py:class:`ComponentMap`.  By unconditionally using :py:`id()` to
    generate all keys, this class performs approximately 25% faster than
    :py:class:`ComponentMap` at the expense of being slightly more
    fragile.

    It is _strongly_ recommended to only use Pyomo components as
    :py:class:`ObjectIdMap` keys.

    .. warning::

       Do not store keys that do not return persistent :py:func:`id()`
       values.  In particular, avoid certain immutable data types like
       :py:`tuple` objects, strings, and long integers.  Doing so may
       result in failed lookups or duplicate entries.

       If you want to mix these keys with other unhashable objects (like
       Pyomo :py:class:`Var` or :py:class:`Param` components), please
       use :py:class:`ComponentMap`.

    """

    __slots__ = ()
    __autoslot_mappers__ = {"_dict": partial(_rehash_keys, id)}

    def __getitem__(self, obj):
        try:
            return self._dict[id(obj)][1]
        except KeyError:
            raise KeyError(obj) from None

    def __setitem__(self, obj, val):
        self._dict[id(obj)] = (obj, val)

    def __delitem__(self, obj):
        try:
            del self._dict[id(obj)]
        except KeyError:
            raise KeyError(obj) from None

    def __contains__(self, obj):
        return id(obj) in self._dict

    def _rekey_items(self, items):
        return ((id(key), val) for key, val in items)

    def __str__(self):
        """String representation of the mapping."""
        tmp = [f"{v[0]} (key={k}): {v[1]}" for k, v in self._dict.items()]
        return f"{self.__class__.__name__}({', '.join(tmp)})"
