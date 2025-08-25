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


class _Hasher(defaultdict):
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


hasher = _Hasher()
