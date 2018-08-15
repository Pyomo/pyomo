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
from six import iteritems, advance_iterator

from pyutilib.misc import flatten_tuple
from pyomo.core.base.sets import SetOf, _SetProduct
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import (
    _IndexedComponent_slice, _IndexedComponent_slice_iter
)

_NotSpecified = object()

class UnitentifiableWildcardSets(Exception):
    pass

class _fill_in_known_wildcards(object):
    def __init__(self, wildcard_values):
        self.base_key = wildcard_values
        self.key = list(wildcard_values)
        self.known_slices = set()

    def __call__(self, _slice):
        if _slice in self.known_slices:
            raise StopIteration()
        self.known_slices.add(_slice)

        if _slice.ellipsis is not None:
            raise RuntimeError(
                "Cannot lookup elements in a _ReferenceDict when the "
                "underlying slice object contains ellipsis")
        try:
            idx = tuple(
                _slice.fixed[i] if i in _slice.fixed else self.key.pop(0)
                for i in range(_slice.explicit_index_count))
        except IndexError:
            raise KeyError(
                "Insufficient values for slice of indexed component '%s'"
                "(found evaluating slice index %s)"
                % (_slice.component.name, self.base_key))

        if idx in _slice.component:
            _slice.last_index = idx
            return _slice.component[idx]
        elif len(idx) == 1 and idx[0] in _slice.component:
            _slice.last_index = idx
            return _slice.component[idx[0]]
        else:
            raise KeyError(
                "Index %s is not valid for indexed component '%s' "
                "(found evaluating slice index %s)"
                % (idx, _slice.component.name, self.base_key))

    def check_complete(self):
        if self.key:
            raise KeyError("Extra (unused) values for slice index %s"
                           % ( self.base_key, ))


class _ReferenceDict(collections.MutableMapping):
    def __init__(self, component_slice):
        self._slice = component_slice

    def __getitem__(self, key):
        try:
            return advance_iterator(self._get_iter(self._slice, key))
        except StopIteration:
            raise KeyError("KeyError: %s" % (key,))

    def __setitem__(self, key, val):
        tmp = self._slice.duplicate()
        op = tmp._call_stack[-1][0]
        if op == _IndexedComponent_slice.get_item:
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.set_item,
                tmp._call_stack[-1][1],
                val )
        elif op == _IndexedComponent_slice.slice_info:
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.set_item,
                tmp,
                val )
        elif op == _IndexedComponent_slice.get_attribute:
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.set_attribute,
                tmp._call_stack[-1][1],
                val )
        else:
            raise DeveloperError(
                "Unexpected slice _call_stack operation: %s" % op)
        try:
            advance_iterator(self._get_iter(tmp, key))
        except StopIteration:
            pass

    def __delitem__(self, key):
        tmp = self._slice.duplicate()
        op = tmp._call_stack[-1][0]
        if op == _IndexedComponent_slice.get_item:
            # If the last attribute of the slice gets an item,
            # change it to delete the item
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.del_item,
                tmp._call_stack[-1][1] )
        elif op == _IndexedComponent_slice.slice_info:
            assert len(tmp._call_stack) == 1
            _iter = self._get_iter(tmp, key)
            try:
                advance_iterator(_iter)
                del _iter._iter_stack[0].component[_iter.get_last_index()]
                return
            except StopIteration:
                raise KeyError("KeyError: %s" % (key,))
        elif op == _IndexedComponent_slice.get_attribute:
            # If the last attribute of the slice retrieves an attribute,
            # change it to delete the attribute
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.del_attribute,
                tmp._call_stack[-1][1] )
        else:
            raise DeveloperError(
                "Unexpected slice _call_stack operation: %s" % op)
        try:
            advance_iterator(self._get_iter(tmp, key))
        except StopIteration:
            pass

    def __iter__(self):
        return self._slice.wildcard_keys()

    def __len__(self):
        return sum(1 for i in self._slice)

    def __contains__(self, key):
        try:
            return super(_ReferenceDict, self).__contains__(key)
        except AttributeError, KeyError:
            return False

    def _get_iter(self, _slice, key):
        if key.__class__ not in (tuple, list):
            key = (key,)
        return _IndexedComponent_slice_iter(
            _slice, _fill_in_known_wildcards(flatten_tuple(key)))


class _ReferenceSet(collections.Set):
    def __init__(self, ref_dict):
        self._ref = ref_dict

    def __contains__(self, key):
        return key in self._ref

    def __iter__(self):
        return iter(self._ref)

    def __len__(self):
        return len(self._ref)


def _get_base_sets(_set):
    if isinstance(_set, _SetProduct):
        for subset in _set.set_tuple:
            for _ in _get_base_sets(subset):
                yield _
    else:
        yield _set

def _identify_wildcard_sets(iter_stack, index):
    if index is None:
        return index

    tmp = [None]*len(iter_stack)
    for i, level in enumerate(iter_stack):
        if level is not None:
            base_sets = list(_get_base_sets(level.component.index_set()))
            if len(base_sets) != level.explicit_index_count:
                return None
            wildcard_sets = { i: base_sets[i]
                              for i in range(len(base_sets))
                              if i not in level.fixed }
            tmp[i] = wildcard_sets
    if not index:
        return tmp

    if len(index) != len(tmp):
        return None
    for i, level in enumerate(tmp):
        if level is None:
            if index[i] is not None:
                return None
            continue
        else:
            if index[i] is None:
                return None
        if (index[i] is None) ^ (level is None):
            return None
        if len(index[i]) != len(level):
            return None
        if any(index[i].get(j,None) is not _set for j,_set in iteritems(level)):
            return None
    return index

def Reference(reference, ctype=_NotSpecified):
    _data = _ReferenceDict(reference)
    _iter = iter(reference)
    ctypes = set()
    index = []
    for obj in _iter:
        ctypes.add(obj.type())
        index = _identify_wildcard_sets(_iter._iter_stack, index)
    if index is None:
        index = SetOf(_ReferenceSet(_data))
    else:
        wildcards = sum((sorted(iteritems(lvl)) for lvl in index
                         if lvl is not None), [])
        index = wildcards[0][1]
        for idx in wildcards[1:]:
            index = index * idx[1]
    if ctype is _NotSpecified:
        if len(ctypes) == 1:
            ctype = ctypes.pop()
        else:
            ctype = IndexedComponent
    elif ctype is None:
        ctype = IndexedComponent

    obj = ctype(index, ctype=ctype)
    obj._constructed = True
    obj._data = _data
    return obj
