#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import copy
from six import iteritems, iterkeys, advance_iterator
from pyomo.common import DeveloperError

class IndexedComponent_slice(object):
    """Special class for slicing through hierarchical component trees

    The basic concept is to interrupt the normal slice generation
    procedure to return a specialized iterable class (this object).  This
    object supports simple getitem / getattr / call methods and caches
    them until it is time to actually iterate through the slice.  We
    then walk down the cached names / indices and resolve the final
    objects during the iteration process.  This works because all the
    calls to __getitem__ / __getattr__ / __call__ happen *before* the
    call to __iter__()
    """
    ATTR_MASK = 8
    ITEM_MASK = 16
    CALL_MASK = 32
    GET_MASK = 1
    SET_MASK = 2
    DEL_MASK = 4

    slice_info = 0
    get_attribute = ATTR_MASK | GET_MASK
    set_attribute = ATTR_MASK | SET_MASK
    del_attribute = ATTR_MASK | DEL_MASK
    get_item = ITEM_MASK | GET_MASK
    set_item = ITEM_MASK | SET_MASK
    del_item = ITEM_MASK | DEL_MASK
    call = CALL_MASK

    def __init__(self, component, fixed=None, sliced=None, ellipsis=None):
        """A "slice" over an _IndexedComponent hierarchy

        This class has two forms for the constructor.  The first form is
        the standard constructor that takes a base component and
        indexing information.  This form takes

           IndexedComponent_slice(component, fixed, sliced, ellipsis)

        The second form is a "copy constructor" that is used internally
        when building up the "call stack" for the hierarchical slice.  The
        copy constructor takes an IndexedComponent_slice and an
        optional "next term" in the slice construction (from get/set/del
        item/attr or call):

           IndexedComponent_slice(slice, next_term=None)

        Parameters
        ----------
        component: IndexedComponent
            The base component for this slice

        fixed: dict
            A dictionary indicating the fixed indices of component,
            mapping index position to value

        sliced: dict
            A dictionary indicating the sliced indices of component
            mapping the index position to the (python) slice object

        ellipsis: int
            The position of the ellipsis in the initial component slice

        """
        # Note that because we use a custom __setattr__, we need to
        # define actual instance attributes using the base class
        # __setattr__.
        set_attr = super(IndexedComponent_slice, self).__setattr__
        if type(component) is IndexedComponent_slice:
            # Copy constructor
            _len = component._len
            # For efficiency, we will only duplicate the call stack
            # list if this instance is not point to the end of the list.
            if _len == len(component._call_stack):
                set_attr('_call_stack', component._call_stack)
            else:
                set_attr('_call_stack', component._call_stack[:_len])
            set_attr('_len', _len)
            if fixed is not None:
                self._call_stack.append(fixed)
                self._len += 1
            set_attr('call_errors_generate_exceptions',
                     component.call_errors_generate_exceptions)
            set_attr('key_errors_generate_exceptions',
                     component.key_errors_generate_exceptions)
            set_attr('attribute_errors_generate_exceptions',
                     component.attribute_errors_generate_exceptions)
        else:
            # Normal constructor
            set_attr('_call_stack', [
                (IndexedComponent_slice.slice_info,
                 (component, fixed, sliced, ellipsis)) ])
            set_attr('_len', 1)
            # Since this is an object, users may change these flags
            # between where they declare the slice and iterate over it.
            set_attr('call_errors_generate_exceptions', True)
            set_attr('key_errors_generate_exceptions', True)
            set_attr('attribute_errors_generate_exceptions', True)

    def __getstate__(self):
        """Serialize this object.

        In general, we would not need to implement this (the object does
        not leverage ``__slots__``).  However, because we have a
        "blanket" implementation of :py:meth:`__getattr__`, we need to
        explicitly implement these to avoid "accidentally" extending or
        evaluating this slice."""
        return {k:getattr(self,k) for k in self.__dict__}

    def __setstate__(self, state):
        """Deserialize the state into this object. """
        set_attr = super(IndexedComponent_slice, self).__setattr__
        for k,v in iteritems(state):
            set_attr(k,v)

    def __deepcopy__(self, memo):
        """Deepcopy this object (leveraging :py:meth:`__getstate__`)"""
        ans = memo[id(self)] = self.__class__.__new__(self.__class__)
        ans.__setstate__(copy.deepcopy(self.__getstate__(), memo))
        return ans

    def __iter__(self):
        """Return an iterator over this slice"""
        return _IndexedComponent_slice_iter(self)

    def __getattr__(self, name):
        """Override the "." operator to defer resolution until iteration.

        Creating a slice of a component returns a
        IndexedComponent_slice object.  Subsequent attempts to resolve
        attributes hit this method.
        """
        return IndexedComponent_slice(self, (
            IndexedComponent_slice.get_attribute, name ) )

    def __setattr__(self, name, value):
        """Override the "." operator implementing attribute assignment

        This supports notation similar to:

            del model.b[:].c.x = 5

        and immediately evaluates the slice.
        """
        # Don't overload any pre-existing attributes
        if name in self.__dict__:
            return super(IndexedComponent_slice, self).__setattr__(name,value)

        # Immediately evaluate the slice and set the attributes
        for i in IndexedComponent_slice(self, (
                IndexedComponent_slice.set_attribute, name, value ) ):
            pass
        return None

    def __getitem__(self, idx):
        """Override the "[]" operator to defer resolution until iteration.

        Creating a slice of a component returns a
        IndexedComponent_slice object.  Subsequent attempts to query
        items hit this method.
        """
        return IndexedComponent_slice(self, (
            IndexedComponent_slice.get_item, idx ) )

    def __setitem__(self, idx, val):
        """Override the "[]" operator for setting item values.

        This supports notation similar to:

            model.b[:].c.x[1,:] = 5

        and immediately evaluates the slice.
        """
        # Immediately evaluate the slice and set the attributes
        for i in IndexedComponent_slice(self, (
                IndexedComponent_slice.set_item, idx, val ) ):
            pass
        return None

    def __delitem__(self, idx):
        """Override the "del []" operator for deleting item values.

        This supports notation similar to:

            del model.b[:].c.x[1,:]

        and immediately evaluates the slice.
        """
        # Immediately evaluate the slice and set the attributes
        for i in IndexedComponent_slice(self, (
                IndexedComponent_slice.del_item, idx ) ):
            pass
        return None

    def __call__(self, *idx, **kwds):
        """Special handling of the "()" operator for component slices.

        Creating a slice of a component returns a IndexedComponent_slice
        object.  Subsequent attempts to call items hit this method.  We
        handle the __call__ method separately based on the item (identifier
        immediately before the "()") being called:

        - if the item was 'component', then we defer resolution of this call
        until we are actually iterating over the slice.  This allows users
        to do operations like `m.b[:].component('foo').bar[:]`

        - if the item is anything else, then we will immediately iterate over
        the slice and call the item.  This allows "vector-like" operations
        like: `m.x[:,1].fix(0)`.
        """
        # There is a weird case in pypy3.6-7.2.0 where __name__ gets
        # called after retrieving an attribute that will be called.  I
        # don't know why that happens, but we will trap it here and
        # remove the getattr(__name__) from the call stack.
        _len = self._len
        if self._call_stack[_len-1][0] == IndexedComponent_slice.get_attribute \
           and self._call_stack[_len-1][1] == '__name__':
            self._len -= 1

        ans = IndexedComponent_slice(self, (
            IndexedComponent_slice.call, idx, kwds ) )
        # Because we just duplicated the slice and added a new entry, we
        # know that the _len == len(_call_stack)
        if ans._call_stack[-2][1] == 'component':
            return ans
        else:
            # Note: simply calling "list(self)" results in infinite
            # recursion in python2.6
            return list( i for i in ans )

    def __hash__(self):
        return hash(tuple(_freeze(x) for x in self._call_stack[:self._len]))

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is not IndexedComponent_slice:
            return False
        return tuple(_freeze(x) for x in self._call_stack[:self._len]) \
            == tuple(_freeze(x) for x in other._call_stack[:other._len])

    def __ne__(self, other):
        return not self.__eq__(other)

    def duplicate(self):
        ans = IndexedComponent_slice(self)
        ans._call_stack = ans._call_stack[:ans._len]
        return ans

    def index_wildcard_keys(self):
        _iter = _IndexedComponent_slice_iter(self, iter_over_index=True)
        return (_iter.get_last_index_wildcards() for _ in _iter)

    def wildcard_keys(self):
        _iter = self.__iter__()
        return (_iter.get_last_index_wildcards() for _ in _iter)

    def wildcard_items(self):
        _iter = self.__iter__()
        return ((_iter.get_last_index_wildcards(), _) for _ in _iter)

    def expanded_keys(self):
        _iter = self.__iter__()
        return (_iter.get_last_index() for _ in _iter)

    def expanded_items(self):
        _iter = self.__iter__()
        return ((_iter.get_last_index(), _) for _ in _iter)


def _freeze(info):
    if info[0] == IndexedComponent_slice.slice_info:
        return (
            info[0],
            id(info[1][0]),  # id of the Component
            tuple(iteritems(info[1][1])), # {idx: value} for fixed
            tuple(iterkeys(info[1][2])),  # {idx: slice} for slices
            info[1][3]  # elipsis index
        )
    elif info[0] & IndexedComponent_slice.ITEM_MASK:
        if type(info[1]) is not tuple:
            index = (info[1],)
        else:
            index = info[1]
        return (
            info[0],
            tuple( (x.start,x.stop,x.step) if type(x) is slice else x
                   for x in index ),
            info[2:],
        )
    else:
        return info



class _slice_generator(object):
    """Utility (iterator) for generating the elements of one slice

    Iterate through the component index and yield the component data
    values that match the slice template.
    """
    def __init__(self, component, fixed, sliced, ellipsis, iter_over_index):
        self.component = component
        self.fixed = fixed
        self.sliced = sliced
        self.ellipsis = ellipsis
        self.iter_over_index = iter_over_index

        self.tuplize_unflattened_index = (
            self.component._implicit_subsets is None
            or len(self.component._implicit_subsets) == 1 )

        self.explicit_index_count = len(fixed) + len(sliced)
        if iter_over_index:
            # This should be used to iterate over all the potential
            # indices of a sparse IndexedComponent.
            self.component_iter = component.index_set().__iter__()
        else:
            # The default behavior is to iterate over the component.
            self.component_iter = component.__iter__()

        # Cache for the most recent index returned. This is used to
        # iterate over keys of the slice (for instance, in a
        # _ReferenceDict).
        self.last_index = None

    def next(self):
        """__next__() iterator for Py2 compatibility"""
        return self.__next__()

    def __next__(self):
        # We have to defer this import to here to resolve circular
        # imports.  Ideally, we would move normalize_index to another
        # module to resolve this.
        from .indexed_component import normalize_index

        while 1:
            # Note: running off the end of the underlying iterator will
            # generate a StopIteration exception that will propagate up
            # and end this iterator.
            index = advance_iterator(self.component_iter)

            # We want a tuple of indices, so convert scalars to tuples
            if normalize_index.flatten:
                _idx = index if type(index) is tuple else (index,)
            elif self.tuplize_unflattened_index:
                _idx = (index,)
            else:
                _idx = index

            # Verify the number of indices: if there is a wildcard
            # slice, then there must be enough indices to at least match
            # the fixed indices.  Without the wildcard slice (ellipsis),
            # the number of indices must match exactly.
            if self.ellipsis is not None:
                if self.explicit_index_count > len(_idx):
                    continue
            elif len(_idx) != self.explicit_index_count:
                continue

            valid = True
            for key, val in iteritems(self.fixed):
                # If this index of the component does not match all
                # the specified fixed indices, don't return anything.
                if not val == _idx[key]:
                    valid = False
                    break
            if valid:
                # Remember the index tuple corresponding to the last
                # component data returned by this iterator. In this way
                # we can use the cached indices to iterate over "indices"
                # of a slice.
                #
                # last_index is the most recent index encountered, not 
                # the last index that will ever be encountered.
                self.last_index = _idx

                # Note: it is important to use __getitem__, as the
                # derived class may implement a non-standard storage
                # mechanism (e.g., Param)
                if (not self.iter_over_index) or index in self.component:
                    # If iter_over_index is False, we are iterating over
                    # the component ("filled-in" indices only).  Since
                    # `advance_iter` was called on the component iter,
                    # we already know index is in self.component.
                    return self.component[index]
                else:
                    # If iter_over_index is True, we need to return
                    # something even when index is not actually in
                    # self.component.  We will (arbitrarily) return
                    # None.
                    return None

# Backwards compatibility
_IndexedComponent_slice = IndexedComponent_slice

# Mock up a callable object with a "check_complete" method
def _advance_iter(_iter):
    return advance_iterator(_iter)
def _advance_iter_check_complete():
    pass
_advance_iter.check_complete = _advance_iter_check_complete

# A dummy class that we can use as a named entity below
class _NotIterable(object): pass


class _IndexedComponent_slice_iter(object):
    def __init__(self, component_slice, advance_iter=_advance_iter,
                 iter_over_index=False):
        # _iter_stack holds a list of elements X where X is either a
        # _slice_generator iterator (if this level in the hierarchy is a
        # slice) or None (if this level is either a SimpleComponent,
        # attribute, method, or is explicitly indexed).
        self._slice = component_slice
        self.advance_iter = advance_iter
        self._iter_over_index = iter_over_index
        call_stack = self._slice._call_stack
        call_stack_len = self._slice._len
        self._iter_stack = [None]*call_stack_len
        # Initialize the top of the `_iter_stack` (deepest part of the
        # model hierarchy):
        if call_stack[0][0] == IndexedComponent_slice.slice_info:
            # The root of the _iter_stack is a generator for the
            # "highest-level slice" (slice closest to the model() block)
            self._iter_stack[0] = _slice_generator(
                *call_stack[0][1], iter_over_index=self._iter_over_index)
            # call_stack[0][1] is a (fixed, sliced, ellipsis) tuple, where
            # fixed and sliced are dicts.
        elif call_stack[0][0] == IndexedComponent_slice.set_item:
            # This is a special case that happens when calling
            # `_ReferenceDict.__setitem__` when the call stack consists
            # of only a `set_item` entry.  We need to initialize the
            # root of _iter_stack to something other than None (so it
            # doesn't immediately get "popped" off.  However, set_item
            # is not an iterable thing, so we will will use a type flag
            # to signal this case to __next__below.
            assert call_stack_len == 1
            self._iter_stack[0] = _NotIterable # Something not None
        else:
            raise DeveloperError("Unexpected call_stack flag encountered: %s"
                                 % call_stack[0][0])

    def __iter__(self):
        """This class implements the iterator API"""
        return self

    def next(self):
        """__next__() iterator for Py2 compatibility"""
        return self.__next__()

    def __next__(self):
        """Return the next element in the slice."""
        # In each call to this function, idx will initially point
        # to the bottom of the stack.
        #
        # NOTE: We refer to this stack as growing "downward", just like
        # the model hierarchy to which it refers.
        idx = len(self._iter_stack)-1
        while True:
            # Flush out any non-slice levels.  Since we initialize
            # _iter_stack with None, in the first call this will
            # immediately walk up to the beginning of the _iter_stack
            #
            # On subsequent calls, we will walk up only as far as the
            # "deepest" active (non-exhausted) iterator. Higher-level
            # iterators could still be active as well, but those index
            # values will remain constant until we have exhausted this
            # "deepest" iterator.
            while self._iter_stack[idx] is None:
                idx -= 1
            # Get the next element in the deepest iterator (active slice)
            try:
                if self._iter_stack[idx] is _NotIterable:
                    # This happens when attempting a `set_item` call on
                    # a `_ReferenceDict` whose slice consists of only a 
                    # `slice_info` entry.
                    # E.g.
                    #     ref = Reference(m.x[:])
                    #     ref._data[1] = 2
                    # but not
                    #     ref = Reference(m.b[:].x[:])
                    #     ref._data['a',1] = 2
                    #
                    _comp = self._slice._call_stack[0][1][0]
                    # _comp is the component in the slice_info entry
                    # of the call stack
                else:
                    # Advance the "deepest active iterator"
                    _comp = self.advance_iter(self._iter_stack[idx])
                    # Note that if we are looking for a specific
                    # wildcard index, that data is stored in
                    # advance_iter() and will be automatically inserted.
                    #
                    # _comp is "local" part of the component we're
                    # looking for.  The rest of the component will be
                    # located using the remainder of the iter stack.
                    #
                    # Note that _comp is actually a component data, because
                    # the _slice_generator (_iter_stack[idx]) returns
                    # component datas rather than indices.
                    # The _slice_generator is able to know about its
                    # component because it was created from a "higher-
                    # level" component/slice in the call/iter stack.
                    # A higher-level iterator may still be active, and
                    # this _slice_generator will need to be regenerated
                    # when/if that iterator is advanced.
                    idx += 1
            except StopIteration:
                # We have exhausted the iterator at this level of the
                # stack
                if not idx:
                    # Top-level iterator is done.  We are done.
                    # (This is how the infinite loop terminates!)
                    raise
                # Reset the _slice_generator to None so that the next
                # iteration will walk up to - and advance - the
                # "next-highest level" iterator.
                self._iter_stack[idx] = None
                # Trivial optimization: we now know that the
                # _iter_stack[idx] is None, so we can preemptively
                # decrement idx in preparatioon for the next iteration
                # of this loop.
                idx -= 1
                continue

            # Walk to the end of the iter/call stacks, constructing a
            # component to return along the way.  The _iter_stack for
            # all of these levels from idx to the end of the list are
            # known to be None at this point.
            while idx < self._slice._len:
                _call = self._slice._call_stack[idx]
                if _call[0] == IndexedComponent_slice.get_attribute:
                    try:
                        # Attach attribute to our current component:
                        _comp = getattr(_comp, _call[1])
                    except AttributeError:
                        # Since we are slicing, we may only be interested in
                        # things that match.  We will allow users to
                        # (silently) ignore any attribute errors generated
                        # by concrete indices in the slice hierarchy...
                        if self._slice.attribute_errors_generate_exceptions \
                           and not self._iter_over_index:
                            raise
                        # Break from the inner loop; next action will be to
                        # advance the "highest-level iterator"
                        break
                elif _call[0] == IndexedComponent_slice.get_item:
                    try:
                        # Get the specified index for the current component:
                        _comp = _comp.__getitem__( _call[1] )
                    except LookupError:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.key_errors_generate_exceptions \
                           and not self._iter_over_index:
                            raise
                        break
                    # If the index defines a slice, add a slice generator
                    # to the iter_stack:
                    if _comp.__class__ is IndexedComponent_slice:
                        # Extract the _slice_generator (for
                        # efficiency... these are always 1-level slices,
                        # so we don't need the overhead of the
                        # IndexedComponent_slice object)
                        assert _comp._len == 1
                        self._iter_stack[idx] = _slice_generator(
                            *_comp._call_stack[0][1],
                            iter_over_index=self._iter_over_index
                        )
                        try:
                            # Advance to get the first component defined
                            # by this slice (so that we have a concrete
                            # context that we can use to decend further
                            # down the model hierarchy):
                            _comp = self.advance_iter(self._iter_stack[idx])
                            # Note that the iterator will remained
                            # cached for subsequent calls to __next__()
                            # (when it will eventually be exhausted).
                        except StopIteration:
                            # We got a slicer, but the slicer doesn't
                            # match anything.  We should break here,
                            # which (due to 'while True' above) will
                            # walk back up to the next iterator and move
                            # on
                            self._iter_stack[idx] = None
                            break
                    else:
                        # `_comp` is a fully qualified component data (i.e.,
                        # not a slice).  Record None in the _iter_stack
                        # so we note that this level in the stack is not
                        # defined by an iterator (i.e., subsequent calls
                        # should immediately "pop" this level off the
                        # stack and proceed to the next higher level.
                        self._iter_stack[idx] = None
                elif _call[0] == IndexedComponent_slice.call:
                    try:
                        # Assume the callable "comp" in our hierarchy
                        # returns a component:
                        _comp = _comp( *(_call[1]), **(_call[2]) )
                    except:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.call_errors_generate_exceptions \
                           and not self._iter_over_index:
                            raise
                        break
                elif _call[0] == IndexedComponent_slice.set_attribute:
                    # set_attribute should only appear at the deepest
                    # point (end) of the call stack
                    assert idx == self._slice._len - 1
                    try:
                        # set attribute of this component:
                        _comp = setattr(_comp, _call[1], _call[2])
                        # If we want to support __setattr__ with "vector"
                        # arguments, e.g.
                        # `m.b[:].v.value = [1,2,3]` or
                        # `m.b[:].v[1].value = m.b[:].v[0].value`,
                        # this will need to be modified to
                        # cache an iterator over _call[2].
                    except AttributeError:
                        # Since we are slicing, we may only be interested in
                        # things that match.  We will allow users to
                        # (silently) ignore any attribute errors generated
                        # by concrete indices in the slice hierarchy...
                        if self._slice.attribute_errors_generate_exceptions:
                            raise
                        break
                elif _call[0] == IndexedComponent_slice.set_item:
                    # `set_item` must always appear at the deepest
                    # point (end) of the call stack
                    assert idx == self._slice._len - 1
                    # We have a somewhat unusual situation when someone
                    # makes a _ReferenceDict to m.x[:] and then wants to
                    # set one of the items.  In that situation,
                    # there is only one level in the _call_stack, and we
                    # need to iterate over it here (so we do not allow
                    # the outer portion of this loop to handle the
                    # iteration).  This is indicated by setting the
                    # _iter_stack value to _NotIterable.
                    if self._iter_stack[idx] is _NotIterable:
                        _iter = _slice_generator(
                            *_call[1], iter_over_index=self._iter_over_index
                        )
                        while True:
                            # This ends when the _slice_generator raises
                            # a StopIteration exception
                            self.advance_iter(_iter)
                            # Check to make sure the custom iterator
                            # (i.e._fill_in_known_wildcards) is complete
                            self.advance_iter.check_complete()
                            _comp[_iter.last_index] = _call[2]

                    # The problem here is that _call[1] may be a slice.
                    # If it is, but we are in something like a
                    # _ReferenceDict, where the caller actually wants a
                    # specific index from the slice, we cannot simply
                    # set every element of the slice.  Instead, we will
                    # look for the component (generating a slice if
                    # appropriate).  If it returns a slice, we can use
                    # our current advance_iter to walk it and set only
                    # the appropriate keys
                    try:
                        _tmp = _comp.__getitem__( _call[1] )
                    except KeyError:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.key_errors_generate_exceptions \
                           and not self._iter_over_index:
                            raise
                        break
                    if _tmp.__class__ is IndexedComponent_slice:
                        # Extract the _slice_generator and evaluate it.
                        assert _tmp._len == 1
                        _iter = _IndexedComponent_slice_iter(
                            _tmp, self.advance_iter)
                        for _ in _iter:
                            # Check to make sure the custom iterator
                            # (i.e._fill_in_known_wildcards) is complete
                            self.advance_iter.check_complete()
                            _comp[_iter.get_last_index()] = _call[2]
                        break
                    else:
                        # Check to make sure the custom iterator
                        # (i.e._fill_in_known_wildcards) is complete
                        self.advance_iter.check_complete()
                        # No try-catch, since we know this key is valid
                        _comp[_call[1]] = _call[2]
                    # If we want to support vectorized set_item, e.g.
                    # `m.b[:].v[1] = m.b[:].v[0]`,
                    # we need to cache an iterator over _call[2].
                elif _call[0] == IndexedComponent_slice.del_item:
                    assert idx == self._slice._len - 1
                    # The problem here is that _call[1] may be a slice.
                    # If it is, but we are in something like a
                    # _ReferenceDict, where the caller actually wants a
                    # specific index from the slice, we cannot simply
                    # delete the slice from the component.  Instead, we
                    # will look for the component (generating a slice if
                    # appropriate).  If it returns a slice, we can use
                    # our current advance_iter to walk it and delete the
                    # appropriate keys
                    try:
                        _tmp = _comp.__getitem__( _call[1] )
                    except KeyError:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.key_errors_generate_exceptions:
                            raise
                        break
                    if _tmp.__class__ is IndexedComponent_slice:
                        # Extract the _slice_generator and evaluate it.
                        assert _tmp._len == 1
                        _iter = _IndexedComponent_slice_iter(
                            _tmp, self.advance_iter)
                        _idx_to_del = []
                        # Two passes, so that we don't edit the _data
                        # dicts while we are iterating over them
                        for _ in _iter:
                            _idx_to_del.append(_iter.get_last_index())
                        # Check to make sure the custom iterator
                        # (i.e._fill_in_known_wildcards) is complete
                        self.advance_iter.check_complete()
                        for _idx in _idx_to_del:
                            del _comp[_idx]
                        break
                    else:
                        # No try-catch, since we know this key is valid
                        del _comp[_call[1]]
                elif _call[0] == IndexedComponent_slice.del_attribute:
                    assert idx == self._slice._len - 1
                    try:
                        _comp = delattr(_comp, _call[1])
                    except AttributeError:
                        # Since we are slicing, we may only be interested in
                        # things that match.  We will allow users to
                        # (silently) ignore any attribute errors generated
                        # by concrete indices in the slice hierarchy...
                        if self._slice.attribute_errors_generate_exceptions:
                            raise
                        break
                else:
                    raise DeveloperError(
                        "Unexpected entry in IndexedComponent_slice "
                        "_call_stack: %s" % (_call[0],))
                idx += 1

            if idx == self._slice._len:
                # Check to make sure the custom iterator
                # (i.e._fill_in_known_wildcards) is complete
                self.advance_iter.check_complete()
                # We have a concrete object at the end of the chain. Return it
                return _comp

    def get_last_index(self):
        ans = sum(
            ( x.last_index for x in self._iter_stack if x is not None ),
            ()
        )
        if len(ans) == 1:
            return ans[0]
        else:
            return ans

    def get_last_index_wildcards(self):
        """Get a tuple of the values in the wildcard positions for the most
        recent indices corresponding to the last component returned by
        each _slice_generator in the iter stack.

        """
        # This method is how we iterate over keys.
        #
        # last_index is the index corresponding to the most recent
        # component data returned by the corresponding _slice_generator.
        # Extract the indices corresponding to the wildcard positions
        # for that slice.
        ans = sum(
            ( tuple( x.last_index[i]
                     for i in range(len(x.last_index))
                     if i not in x.fixed )
              for x in self._iter_stack if x is not None ),
            ()
        )
        if len(ans) == 1:
            return ans[0]
        else:
            return ans

