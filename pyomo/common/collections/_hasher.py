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

from collections import defaultdict


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

    def __init__(self, *args, **kwargs):
        super().__init__(lambda: self._missing_impl, *args, **kwargs)
        self[tuple] = self._tuple

    def _missing_impl(self, val):
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
        return id(val)

    def _tuple(self, val):
        return tuple(self[i.__class__](i) for i in val)

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


#: The global 'hasher' instance for managing "universal" hashing.
#:
#: This instance of the :class:`HashDispatcher` is used by
#: :class:`~pyomo.common.collections.component_map.ComponentMap` and
#: :class:`~pyomo.common.collections.component_set.ComponentSet` for
#: generating hashes for all Python and Pyomo types.
hasher = HashDispatcher()
