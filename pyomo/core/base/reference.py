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
from six import advance_iterator

from pyutilib.misc import flatten_tuple

class _fill_in_known_wildcards(object):
    def __init__(self, wildcard_values):
        self.key = wildcard_values
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
                "Insufficient wildcards to populate slice of indexed "
                "component '%s'" % (_slice.component.name,))

        if idx in _slice.component:
            _slice.last_index = idx
            return _slice.component[idx]
        elif len(idx) == 1 and idx[0] in _slice.component:
            _slice.last_index = idx
            return _slice.component[idx[0]]
        else:
            raise KeyError(
                "Index '%s' is not valid for indexed component '%s'"
                % (idx, _slice.component.name))


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
        key = list(flatten_tuple(key))
        return _IndexedComponent_slice_iter(
            _slice, _fill_in_known_wildcards(key))


class _ReferenceSet(collections.Set):
    def __init__(self, ref_dict):
        self._ref = ref_dict

    def __contains__(self, key):
        return key in self._ref

    def __iter__(self):
        return iter(self._ref)

    def __len__(self):
        return len(self._ref)
