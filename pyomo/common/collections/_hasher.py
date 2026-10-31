# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from collections import defaultdict


class _HashKey:
    """Utility class to support hashing by object id()

    This class should never be instantiated, and should never be
    accessed referenced by user code.  Instead this provides a simple
    :class:`type` that we can use as an internal flag to differentiate
    between an :class:`int` key and the result from :func:`id()`.

    """

    pass


class HashDispatcher(defaultdict):
    """Dispatch table for generating "universal" hashing of all Python objects.

    This class manages a dispatch table for providing hash functions for all Python
    types.  When an object is passed to the Hasher, it determines the appropriate
    hashing strategy based on the object's type:

      - If a custom hashing function is registered for the type, it is used.
      - If the object is natively hashable, the default hash is used.
      - If the object is unhashable, the object's :func:`id()` is used as a fallback.

    The Hasher also includes special handling for tuples by recursively applying the
    appropriate hashing strategy to each element within the tuple.
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(lambda: self._missing_impl, *args, **kwargs)
        self[tuple] = self._tuple

    def _missing_impl(self, val):
        # Inherit the hasher from a base class, if found
        for _type in val.__class__.__mro__[1:]:
            if _type in self:
                self[val.__class__] = ans = self[_type]
                return ans(val)
        try:
            hash(val)
            self[val.__class__] = self._hashable
        except:
            self[val.__class__] = self._unhashable
        return self[val.__class__](val)

    @staticmethod
    def _hashable(val):
        return val

    @staticmethod
    def _unhashable(val):
        return _HashKey, id(val)

    def _tuple(self, val):
        try:
            # if *this tuple* is hashable, then use it as the key
            hash(val)
            return val
        except:
            # duplicate the tuple, recursively processing all fields.
            # The use of val.__class__ ensures that derived things (like
            # namedtuples) have their class preserved.
            return val.__class__(self[i.__class__](i) for i in val)

    def hashable(self, obj, hashable=None):
        if isinstance(obj, type):
            cls = obj
        else:
            cls = type(obj)
        if hashable is None:
            fcn = self.get(cls, None)
            if fcn is None:
                raise KeyError(obj)
            return fcn is self._hashable
        self[cls] = self._hashable if hashable else self._unhashable

    def __call__(self, obj):
        # Make the dispatcher callable so that it can be used in place of id()
        return self[obj.__class__](obj)


#: The global 'hasher' instance for managing "universal" hashing.
#:
#: This instance of the :class:`HashDispatcher` is used by
#: :class:`~pyomo.common.collections.component_map.ComponentMap` and
#: :class:`~pyomo.common.collections.component_set.ComponentSet` for
#: generating hashes for all Python and Pyomo types.
hasher = HashDispatcher()
