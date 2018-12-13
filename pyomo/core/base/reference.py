#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyutilib.misc import flatten_tuple
from pyomo.common import DeveloperError
from pyomo.core.base.sets import SetOf, _SetProduct, _SetDataBase
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set
)
from pyomo.core.base.indexed_component_slice import (
    _IndexedComponent_slice, _IndexedComponent_slice_iter
)

import six
from six import iteritems, advance_iterator

if six.PY3:
    from collections.abc import MutableMapping as collections_MutableMapping
    from collections.abc import Set as collections_Set
else:
    from collections import MutableMapping as collections_MutableMapping
    from collections import Set as collections_Set

_NotSpecified = object()

class _fill_in_known_wildcards(object):
    """Variant of "six.advance_iterator" that substitutes wildcard values

    This object is initialized with a tuple of index values.  Calling
    the resulting object on a :py:class:`_slice_generator` will
    "advance" the iterator, substituting values from the tuple into the
    slice wildcards (":" indices), and returning the resulting object.
    The motivation for implementing this as an iterator is so that we
    can re-use all the logic from
    :py:meth:`_IndexedComponent_slice_iter.__next__` when looking up
    specific indices within the slice.

    Parameters
    ----------
    wildcard_values : tuple of index values
        a tuple containing index values to substitute into the slice wildcards
    """
    def __init__(self, wildcard_values):
        self.base_key = wildcard_values
        self.key = list(wildcard_values)
        self.known_slices = set()

    def __call__(self, _slice):
        """Advance the specified slice generator, substituting wildcard values

        This advances the passed :py:class:`_slice_generator
        <pyomo.core.base.indexed_component_slice._slice_generator>` by
        substituting values from the `wildcard_values` list for any
        wildcard slices ("`:`").

        Parameters
        ----------
        _slice : pyomo.core.base.indexed_component_slice._slice_generator
            the slice to advance
        """
        if _slice in self.known_slices:
            raise StopIteration()
        self.known_slices.add(_slice)

        if _slice.ellipsis is None:
            idx_count = _slice.explicit_index_count
        elif not _slice.component.is_indexed():
            idx_count = 1
        else:
            idx_count = _slice.component.index_set().dimen
            if idx_count is None:
                raise SliceEllipsisLookupError(
                    "Cannot lookup elements in a _ReferenceDict when the "
                    "underlying slice object contains ellipsis over a jagged "
                    "(dimen=None) Set")
        try:
            idx = tuple(
                _slice.fixed[i] if i in _slice.fixed else self.key.pop(0)
                for i in range(idx_count))
        except IndexError:
            raise KeyError(
                "Insufficient values for slice of indexed component '%s' "
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


class SliceEllipsisLookupError(Exception):
    pass

class _ReferenceDict(collections_MutableMapping):
    def __init__(self, component_slice):
        self._slice = component_slice

    def __getitem__(self, key):
        try:
            return advance_iterator(self._get_iter(self._slice, key))
        except StopIteration:
            raise KeyError("KeyError: %s" % (key,))
        except SliceEllipsisLookupError:
            if type(key) is tuple and len(key) == 1:
                key = key[0]
            # Brute force (linear time) lookup
            _iter = iter(self._slice)
            for item in _iter:
                if _iter.get_last_index_wildcards() == key:
                    return item
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
                tmp._call_stack[-1][1],
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
        except (AttributeError, KeyError):
            return False

    def iteritems(self):
        """Return the wildcard, value tuples for this ReferenceDict

        This method is necessary because the default implementation
        iterates over the keys and looks the values up in the
        dictionary.  Unfortunately some slices have structures that make
        looking up components by the wildcard keys very expensive
        (linear time; e.g., the use of elipses with jagged sets).  By
        implementing this method without using lookups, general methods
        that iterate over everything (like component.pprint()) will
        still be linear and not quadratic time.

        """
        return self._slice.wildcard_items()

    def itervalues(self):
        """Return the values for this ReferenceDict

        This method is necessary because the default implementation
        iterates over the keys and looks the values up in the
        dictionary.  Unfortunately some slices have structures that make
        looking up components by the wildcard keys very expensive
        (linear time; e.g., the use of elipses with jagged sets).  By
        implementing this method without using lookups, general methods
        that iterate over everything (like component.pprint()) will
        still be linear and not quadratic time.

        """
        return iter(self._slice)

    def _get_iter(self, _slice, key):
        if key.__class__ not in (tuple, list):
            key = (key,)
        return _IndexedComponent_slice_iter(
            _slice, _fill_in_known_wildcards(flatten_tuple(key)))

if six.PY3:
    _ReferenceDict.items = _ReferenceDict.iteritems
    _ReferenceDict.values = _ReferenceDict.itervalues

class _ReferenceSet(collections_Set):
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
    # if we have already decided that there isn't a comon index for the
    # slices, there is nothing more we can do.  Bail.
    if index is None:
        return index

    # Walk the iter_stack that led to the current item and try to
    # identify the component wildcard sets
    tmp = [None]*len(iter_stack)
    for i, level in enumerate(iter_stack):
        if level is not None:
            offset = 0
            wildcard_sets = {}
            for j,s in enumerate(_get_base_sets(level.component.index_set())):
                if s is UnindexedComponent_set:
                    wildcard_sets[j] = s
                    offset += 1
                    continue
                if s.dimen is None:
                    return None
                wild = sum( 1 for k in range(s.dimen)
                            if k+offset not in level.fixed )
                if wild == s.dimen:
                    wildcard_sets[j] = s
                elif wild != 0:
                    # This subset is "touched" by an explicit slice, but
                    # the whole set is not (i.e. there is a fixed
                    # component to this subset).  Therefore, as we
                    # cannot extract that subset, we quit.
                    return None
                offset += s.dimen
            # I believe that this check is not necessary: weirdnesses
            # with elipsis should get caught by the check for s.dimen
            # above.
            #
            #if offset != level.explicit_index_count:
            #    return None
            tmp[i] = wildcard_sets
    if not index:
        return tmp

    # Any of the following would preclude identifying common sets.
    # However, I can't see a way to actually create any of these
    # situations (i.e., I can't test them).  Assertions are left in for
    # defensive programming.
    assert len(index) == len(tmp)
    for i, level in enumerate(tmp):
        assert (index[i] is None) == (level is None)
        # No slices at this level in the slice
        if level is None:
            continue
        # if there are a differing number of subsets
        if len(index[i]) != len(level):
            return None
        # if any subset differs
        if any(index[i].get(j,None) is not _set for j,_set in iteritems(level)):
            return None
    return index

def Reference(reference, ctype=_NotSpecified):
    """Creates a component that references other components

    ``Reference`` generates a *reference component*; that is, an indexed
    component that does not contain data, but instead references data
    stored in other components as defined by a component slice.  The
    ctype parameter sets the :py:meth:`Component.type` of the resulting
    indexed component.  If the ctype parameter is not set and all data
    identified by the slice (at construction time) share a common
    :py:meth:`Component.type`, then that type is assumed.  If either the
    ctype parameter is ``None`` or the data has more than one ctype, the
    resulting indexed component will have a ctype of
    :py:class:`IndexedComponent`.

    If the indices associated with wildcards in the component slice all
    refer to the same :py:class:`Set` objects for all data identifed by
    the slice, then the resulting indexed component will be indexed by
    the product of those sets.  However, if all data do not share common
    set objects, or only a subset of indices in a multidimentional set
    appear as wildcards, then the resulting indexed component will be
    indexed by a :py:class:`SetOf` containing a
    :py:class:`_ReferenceSet` for the slice.

    Parameters
    ----------
    reference : :py:class:`_IndexedComponent_slice`
        component slice that defines the data to include in the
        Reference component

    ctype : :py:class:`type` [optional]
        the type used to create the resulting indexed component.  If not
        specified, the data's ctype will be used (if all data share a
        common ctype).  If multiple data ctypes are found or type is
        ``None``, then :py:class:`IndexedComponent` will be used.

    Examples
    --------

    .. doctest::

        >>> from pyomo.environ import *
        >>> m = ConcreteModel()
        >>> @m.Block([1,2],[3,4])
        ... def b(b,i,j):
        ...     b.x = Var(bounds=(i,j))
        ...
        >>> m.r1 = Reference(m.b[:,:].x)
        >>> m.r1.pprint()
        r1 : Size=4, Index=r1_index
            Key    : Lower : Value : Upper : Fixed : Stale : Domain
            (1, 3) :     1 :  None :     3 : False :  True :  Reals
            (1, 4) :     1 :  None :     4 : False :  True :  Reals
            (2, 3) :     2 :  None :     3 : False :  True :  Reals
            (2, 4) :     2 :  None :     4 : False :  True :  Reals

    Reference components may also refer to subsets of the original data:

    .. doctest::

        >>> m.r2 = Reference(m.b[:,3].x)
        >>> m.r2.pprint()
        r2 : Size=2, Index=b_index_0
            Key : Lower : Value : Upper : Fixed : Stale : Domain
              1 :     1 :  None :     3 : False :  True :  Reals
              2 :     2 :  None :     3 : False :  True :  Reals

    Reference components may have wildcards at multiple levels of the
    model hierarchy:

    .. doctest::

        >>> from pyomo.environ import *
        >>> m = ConcreteModel()
        >>> @m.Block([1,2])
        ... def b(b,i):
        ...     b.x = Var([3,4], bounds=(i,None))
        ...
        >>> m.r3 = Reference(m.b[:].x[:])
        >>> m.r3.pprint()
        r3 : Size=4, Index=r3_index
            Key    : Lower : Value : Upper : Fixed : Stale : Domain
            (1, 3) :     1 :  None :  None : False :  True :  Reals
            (1, 4) :     1 :  None :  None : False :  True :  Reals
            (2, 3) :     2 :  None :  None : False :  True :  Reals
            (2, 4) :     2 :  None :  None : False :  True :  Reals

    The resulting reference component may be used just like any other
    component.  Changes to the stored data will be reflected in the
    original objects:

    .. doctest::

        >>> m.r3[1,4] = 10
        >>> m.b[1].x.pprint()
        x : Size=2, Index=b[1].x_index
            Key : Lower : Value : Upper : Fixed : Stale : Domain
              3 :     1 :  None :  None : False :  True :  Reals
              4 :     1 :    10 :  None : False : False :  Reals

    """
    if isinstance(reference, _IndexedComponent_slice):
        pass
    elif isinstance(reference, Component):
        reference = reference[...]
    else:
        raise TypeError(
            "First argument to Reference constructors must be a "
            "component or component slice (received %s)"
            % (type(reference).__name__,))

    _data = _ReferenceDict(reference)
    _iter = iter(reference)
    if ctype is _NotSpecified:
        ctypes = set()
    else:
        # If the caller specified a ctype, then we will prepopulate the
        # list to improve our chances of avoiding a scan of the entire
        # Reference
        ctypes = set((1,2))
    index = []
    for obj in _iter:
        ctypes.add(obj.type())
        if not isinstance(obj, ComponentData):
            # This object is not a ComponentData (likely it is a pure
            # IndexedComponent container).  As the Reference will treat
            # it as if it *were* a ComponentData, we will skip ctype
            # identification and return a base IndexedComponent, thereby
            # preventing strange exceptions in the writers and with
            # things like pprint().  Of course, all of this logic is
            # skipped if the User knows better and forced a ctype on us.
            ctypes.add(0)
        if index is not None:
            index = _identify_wildcard_sets(_iter._iter_stack, index)
        # Note that we want to walk the entire slice, unless we can
        # prove that BOTH there aren't common indexing sets AND there is
        # more than one ctype.
        elif len(ctypes) > 1:
            break
    if index is None:
        index = SetOf(_ReferenceSet(_data))
    else:
        wildcards = sum((sorted(iteritems(lvl)) for lvl in index
                         if lvl is not None), [])
        index = wildcards[0][1]
        if not isinstance(index, _SetDataBase):
            index = SetOf(index)
        for lvl, idx in wildcards[1:]:
            if not isinstance(idx, _SetDataBase):
                idx = SetOf(idx)
            index = index * idx
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
