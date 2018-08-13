#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ['IndexedComponent', 'ActiveIndexedComponent']

import pyutilib.misc

from pyomo.core.expr.current import TemplateExpressionError
from pyomo.core.base.component import Component, ActiveComponent
from pyomo.core.base.config import PyomoOptions
from pyomo.common import DeveloperError

import collections
from six import PY3, itervalues, iteritems, advance_iterator
import sys

UnindexedComponent_set = set([None])

def normalize_index(index):
    """
    Flatten a component index.  If it has length 1, then
    return just the element.  If it has length > 1, then
    return a tuple.
    """
    idx = pyutilib.misc.flatten(index)
    if type(idx) is list:
        if len(idx) == 1:
            idx = idx[0]
        else:
            idx = tuple(idx)
    return idx
normalize_index.flatten = True

class _NotFound(object):
    pass

#
# Get the fully-qualified name for this index.  If there isn't anything
# in the _data dict (and there shouldn't be), then add something, get
# the name, and remove it.  This allows us to get the name of something
# that we haven't added yet without changing the state of the constraint
# object.
#
def _get_indexed_component_data_name(component, index):
    """Returns the fully-qualified component name for an unconstructed index.

    The ComponentData.name property assumes that the ComponentData has
    already been assigned to the owning Component.  This is a problem
    during the process of constructing a ComponentData instance, as we
    may need to throw an exception before the ComponentData is added to
    the owning component.  In those cases, we can use this function to
    generate the fully-qualified name without (permanently) adding the
    object to the Component.

    """
    if not component.is_indexed():
        return component.name
    elif index in component._data:
        ans = component._data[index].name
    else:
        for i in range(5):
            try:
                component._data[index] = component._ComponentDataClass(
                    *((None,)*i), component=component)
                i = None
                break
            except:
                pass
        if i is not None:
            # None of the generic positional arguments worked; raise an
            # exception
            component._data[index] = component._ComponentDataClass(
                component=component)
        try:
            ans = component._data[index].name
        except:
            ans = component.name + '[{unknown index}]'
        finally:
            del component._data[index]
    return ans


class _slice_generator(object):
    """Utility (iterator) for generating the elements of one slice

    Iterate through the component index and yield the component data
    values that match the slice template.
    """
    def __init__(self, component, fixed, sliced, ellipsis):
        self.component = component
        self.fixed = fixed
        self.sliced = sliced
        self.ellipsis = ellipsis

        self.explicit_index_count = len(fixed) + len(sliced)
        self.component_iter = component.__iter__()
        self.last_index = None

    def next(self):
        """__next__() iterator for Py2 compatibility"""
        return self.__next__()

    def __next__(self):
        while 1:
            # Note: running off the end of the underlying iterator will
            # generate a StopIteration exception that will propagate up
            # and end this iterator.
            index = advance_iterator(self.component_iter)

            # We want a tuple of indices, so convert scalars to tuples
            _idx = index if type(index) is tuple else (index,)

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
                if not val == _idx[key]:
                    valid = False
                    break
            if valid:
                # Remember the index tuple corresponding to the last
                # component data returned by this iterator
                self.last_index = _idx
                # Note: it is important to use __getitem__, as the
                # derived class may implement a non-standard storage
                # mechanism (e.g., Param)
                return self.component[index]

class _IndexedComponent_slice_iter(object):
    def __init__(self, component_slice, advance_iter=advance_iterator):
        # _iter_stack holds a list of elements X where X is either an
        # _slice_generator iterator (if this level in the hierarchy is a
        # slice) or None (if this level is either a SimpleComponent,
        # attribute, method, or is explicitly indexed).
        self._slice = component_slice
        self.advance_iter = advance_iter
        assert( self._slice._call_stack[0][0]
                == _IndexedComponent_slice.slice_info )
        self._iter_stack = [None]*len(self._slice._call_stack)
        self._iter_stack[0] = _slice_generator(*self._slice._call_stack[0][1])

    def __iter__(self):
        """This class implements the iterator API"""
        return self

    def next(self):
        """__next__() iterator for Py2 compatibility"""
        return self.__next__()

    def __next__(self):
        """Return the next element in the slice."""
        idx = len(self._iter_stack)-1
        while True:
            # Flush out any non-slice levels.  Since we initialize
            # _iter_stack with None, in the first call this will
            # immediately walk up to the beginning of the _iter_stack
            while self._iter_stack[idx] is None:
                idx -= 1
            # Get the next element in the deepest active slice
            try:
                _comp = self.advance_iter(self._iter_stack[idx])
                idx += 1
            except StopIteration:
                if not idx:
                    # Top-level iterator is done.  We are done.
                    # (This is how the infinite loop terminates!)
                    raise
                self._iter_stack[idx] = None
                idx -= 1
                continue
            # Walk down the hierarchy to get to the final object
            while idx < len(self._slice._call_stack):
                _call = self._slice._call_stack[idx]
                if _call[0] == _IndexedComponent_slice.get_attribute:
                    try:
                        _comp = getattr(_comp, _call[1])
                    except AttributeError:
                        # Since we are slicing, we may only be interested in
                        # things that match.  We will allow users to
                        # (silently) ignore any attribute errors generated
                        # by concrete indices in the slice hierarchy...
                        if self._slice.attribute_errors_generate_exceptions:
                            raise
                        break
                    self._iter_stack[idx] = None
                elif _call[0] == _IndexedComponent_slice.get_item:
                    try:
                        _comp = _comp.__getitem__( _call[1] )
                    except KeyError:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.key_errors_generate_exceptions:
                            raise
                        break
                    if _comp.__class__ is _IndexedComponent_slice:
                        # Extract the _slice_generator (for
                        # efficiency... these are always 1-level slices,
                        # so we don't need the overhead of the
                        # _IndexedComponent_slice object)
                        assert len(_comp._call_stack) == 1
                        self._iter_stack[idx] = _slice_generator(
                            *_comp._call_stack[0][1])
                        try:
                            _comp = self.advance_iter(self._iter_stack[idx])
                        except StopIteration:
                            # We got a slicer, but the slicer doesn't
                            # matching anything.  We should break here,
                            # which (due to 'while True' above) will
                            # walk back up to the next iterator and move
                            # on
                            self._iter_stack[idx] = None
                            break
                    else:
                        self._iter_stack[idx] = None
                elif _call[0] == _IndexedComponent_slice.call:
                    try:
                        _comp = _comp( *(_call[1]), **(_call[2]) )
                    except:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.call_errors_generate_exceptions:
                            raise
                        break
                    self._iter_stack[idx] = None
                elif _call[0] == _IndexedComponent_slice.set_attribute:
                    assert idx == len(self._slice._call_stack) - 1
                    try:
                        _comp = setattr(_comp, _call[1], _call[2])
                    except AttributeError:
                        # Since we are slicing, we may only be interested in
                        # things that match.  We will allow users to
                        # (silently) ignore any attribute errors generated
                        # by concrete indices in the slice hierarchy...
                        if self._slice.attribute_errors_generate_exceptions:
                            raise
                        break
                    self._iter_stack[idx] = None
                elif _call[0] == _IndexedComponent_slice.set_item:
                    assert idx == len(self._slice._call_stack) - 1
                    try:
                        _comp[_call[1]] = _call[2]
                        _comp = None
                    except KeyError:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.key_errors_generate_exceptions:
                            raise
                        break
                    self._iter_stack[idx] = None
                elif _call[0] == _IndexedComponent_slice.del_item:
                    assert idx == len(self._slice._call_stack) - 1
                    try:
                        del _comp[_call[1]]
                        _comp = None
                    except KeyError:
                        # Since we are slicing, we may only be
                        # interested in things that match.  We will
                        # allow users to (silently) ignore any key
                        # errors generated by concrete indices in the
                        # slice hierarchy...
                        if self._slice.key_errors_generate_exceptions:
                            raise
                        break
                    self._iter_stack[idx] = None
                else:
                    raise RuntimeError(
                        "Unexpected entry in _IndexedComponent_slice "
                        "_call_stack: %s" % (_call[0],))
                idx += 1

            if idx == len(self._slice._call_stack):
                # We have a concrete object at the end of the chain. Return it
                return _comp

    def get_last_index(self):
        return tuple(
            x.last_index
            for x in self._iter_stack if x is not None
        )

    def get_last_index_wildcards(self):
        return tuple(
            tuple( x.last_index[i]
                   for i in range(len(x.last_index))
                   if i not in x.fixed )
            for x in self._iter_stack if x is not None
        )


class _IndexedComponent_slice(object):
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
    slice_info = 0
    get_attribute = 1
    set_attribute = 4
    get_item = 2
    set_item = 5
    del_item = 6
    call = 3

    def __init__(self, component, fixed, sliced, ellipsis):
        # Note that because we use a custom __setattr__, we need to
        # define actual instance attributes using the base class
        # __setattr__.
        set_attr = super(_IndexedComponent_slice, self).__setattr__

        set_attr('_call_stack', [
            (_IndexedComponent_slice.slice_info,
             (component, fixed, sliced, ellipsis)) ])
        # Since this is an object, users may change these flags between
        # where they declare the slice and iterate over it.
        set_attr('call_errors_generate_exceptions', True)
        set_attr('key_errors_generate_exceptions', True)
        set_attr('attribute_errors_generate_exceptions', True)

    def __iter__(self):
        """Return an iterator over this slice"""
        return _IndexedComponent_slice_iter(self)

    def __getattr__(self, name):
        """Override the "." operator to defer resolution until iteration.

        Creating a slice of a component returns a
        _IndexedComponent_slice object.  Subsequent attempts to resolve
        attributes hit this method.
        """
        self._call_stack.append( (
            _IndexedComponent_slice.get_attribute, name ) )
        return self

    def __setattr__(self, name, value):
        """Override the "." operator implementing attribute assignment

        This supports notation similar to:

            del model.b[:].c.x = 5
        
        and immediately evaluates the slice.
        """
        # Don't overload any pre-existing attributes
        if name in self.__dict__:
            return super(_IndexedComponent_slice, self).__setattr__(name,value)

        self._call_stack.append( (
            _IndexedComponent_slice.set_attribute, name, value ) )
        # Immediately evaluate the slice and set the attributes
        for i in self: pass
        return None

    def __getitem__(self, idx):
        """Override the "[]" operator to defer resolution until iteration.

        Creating a slice of a component returns a
        _IndexedComponent_slice object.  Subsequent attempts to query
        items hit this method.
        """
        self._call_stack.append( (
            _IndexedComponent_slice.get_item, idx ) )
        return self

    def __setitem__(self, idx, val):
        """Override the "[]" operator for setting item values.

        This supports notation similar to:

            del model.b[:].c.x[1,:] = 5
        
        and immediately evaluates the slice.
        """
        self._call_stack.append( (
            _IndexedComponent_slice.set_item, idx, val ) )
        # Immediately evaluate the slice and set the attributes
        for i in self: pass
        return None

    def __delitem__(self, idx):
        """Override the "del []" operator for deleting item values.

        This supports notation similar to:

            del model.b[:].c.x[1,:]
        
        and immediately evaluates the slice.
        """
        self._call_stack.append( (
            _IndexedComponent_slice.del_item, idx ) )
        # Immediately evaluate the slice and set the attributes
        for i in self: pass
        return None

    def __call__(self, *idx, **kwds):
        """Special handling of the "()" operator for component slices.

        Creating a slice of a component returns a _IndexedComponent_slice
        object.  Subsequent attempts to call items hit this method.  We
        handle the __call__ method separately based on the item ( identifier
        immediately before the "()") being called:

        - if the item was 'component', then we defer resolution of this call
        until we are actually iterating over the slice.  This allows users
        to do operations like `m.b[:].component('foo').bar[:]`

        - if the item is anything else, then we will immediately iterate over
        the slice and call the item.  This allows "vector-like" operations
        like: `m.x[:,1].fix(0)`.
        """
        self._call_stack.append( (
            _IndexedComponent_slice.call, idx, kwds ) )
        if self._call_stack[-2][1] == 'component':
            return self
        else:
            # Note: simply calling "list(self)" results in infinite
            # recursion in python2.6
            return list( i for i in self )

    def duplicate(self):
        ans = _IndexedComponent_slice(None,None,None,None)
        ans.call_errors_generate_exceptions \
            = self.call_errors_generate_exceptions
        ans.key_errors_generate_exceptions \
            = self.key_errors_generate_exceptions
        ans.attribute_errors_generate_exceptions \
            = self.attribute_errors_generate_exceptions
        ans._call_stack = list(self._call_stack)
        return ans

    def wildcard_keys(self):
        _iter = self.__iter__()
        return (_iter.get_last_index_wildcards() for _ in _iter)

    def expanded_keys(self):
        _iter = self.__iter__()
        return (_iter.get_last_index() for _ in _iter)

    def expanded_items(self):
        _iter = self.__iter__()
        return ((_iter.get_last_index(), _) for _ in _iter)


if sys.version_info[0] >= 3:
    _IndexedComponent_slice.keys = _IndexedComponent_slice.iterkeys
    _IndexedComponent_slice.values = _IndexedComponent_slice.itervalues
    _IndexedComponent_slice.items = _IndexedComponent_slice.iteritems


class IndexedComponent(Component):
    """
    This is the base class for all indexed modeling components.
    This class stores a dictionary, self._data, that maps indices
    to component data objects.  The object self._index defines valid
    keys for this dictionary, and the dictionary keys may be a
    strict subset.

    The standard access and iteration methods iterate over the the
    keys of self._data.  This class supports a concept of a default
    component data value.  When enabled, the default does not
    change the access and iteration methods.

    IndexedComponent may be given a set over which indexing is restricted.
    Alternatively, IndexedComponent may be indexed over Any
    (pyomo.core.base.set_types.Any), in which case the IndexedComponent
    behaves like a dictionary - any hashable object can be used as a key
    and keys can be added on the fly.

    Constructor arguments:
        ctype       The class type for the derived subclass
        doc         A text string describing this component

    Private class attributes:
        _data               A dictionary from the index set to
                                component data objects
        _index              The set of valid indices
        _implicit_subsets   A temporary data element that stores
                                sets that are transfered to the model
    """

    #
    # If an index is supplied for which there is not a _data entry
    # (specifically, in a get call), then this flag determines whether
    # a check is performed to see if the input index is in the
    # index set _index. This is extremely expensive, and so this flag
    # is provided to disable that feature globally.
    #
    _DEFAULT_INDEX_CHECKING_ENABLED = True

    def __init__(self, *args, **kwds):
        from pyomo.core.base.sets import process_setarg
        #
        kwds.pop('noruleinit', None)
        Component.__init__(self, **kwds)
        #
        self._data = {}
        #
        if len(args) == 0 or (len(args) == 1 and
                              args[0] is UnindexedComponent_set):
            #
            # If no indexing sets are provided, generate a dummy index
            #
            self._implicit_subsets = None
            self._index = UnindexedComponent_set
        elif len(args) == 1:
            #
            # If a single indexing set is provided, just process it.
            #
            self._implicit_subsets = None
            self._index = process_setarg(args[0])
        else:
            #
            # If multiple indexing sets are provided, process them all,
            # and store the cross-product of these sets.  The individual
            # sets need to stored in the Pyomo model, so the
            # _implicit_subsets class data is used for this temporary
            # storage.
            #
            # Example:  Pyomo allows things like
            # "Param([1,2,3], range(100), initialize=0)".  This
            # needs to create *3* sets: two SetOf components and then
            # the SetProduct.  That means that the component needs to
            # hold on to the implicit SetOf objects until the component
            # is assigned to a model (where the implicit subsets can be
            # "transferred" to the model).
            #
            tmp = [process_setarg(x) for x in args]
            self._implicit_subsets = tmp
            self._index = tmp[0].cross(*tmp[1:])

    def __getstate__(self):
        # Special processing of getstate so that we never copy the
        # UnindexedComponent_set set
        state = super(IndexedComponent, self).__getstate__()
        if not self.is_indexed():
            state['_index'] = None
        return state

    def __setstate__(self, state):
        # Special processing of setstate so that we never copy the
        # UnindexedComponent_set set
        if state['_index'] is None:
            state['_index'] = UnindexedComponent_set
        super(IndexedComponent, self).__setstate__(state)

    def to_dense_data(self):
        """TODO"""
        for idx in self._index:
            if idx not in self._data:
                self._getitem_when_not_present(idx)

    def clear(self):
        """Clear the data in this component"""
        if self.is_indexed():
            self._data = {}
        else:
            raise DeveloperError(
                "Derived scalar component %s failed to define clear()."
                % (self.__class__.__name__,))

    def index_set(self):
        """Return the index set"""
        return self._index

    def is_indexed(self):
        """Return true if this component is indexed"""
        return self._index is not UnindexedComponent_set

    def dim(self):
        """Return the dimension of the index"""
        if not self.is_indexed():
            return 0
        return getattr(self._index, 'dimen', 0)

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.
        """
        return len(self._data)

    def __contains__(self, idx):
        """Return true if the index is in the dictionary"""
        return idx in self._data

    def __iter__(self):
        """Iterate over the keys in the dictionary"""

        if not getattr(self._index, 'concrete', True):
            #
            # If the index set is virtual (e.g., Any) then return the
            # data iterator.  Note that since we cannot check the length
            # of the underlying Set, there should be no warning if the
            # user iterates over the set when the _data dict is empty.
            #
            return self._data.__iter__()
        elif len(self._data) == len(self._index):
            #
            # If the data is dense then return the index iterator.
            #
            return self._index.__iter__()
        else:
            if not self._data and self._index and PyomoOptions.paranoia_level:
                logger.warning(
"""Iterating over a Component (%s)
defined by a non-empty concrete set before any data objects have
actually been added to the Component.  The iterator will be empty.
This is usually caused by Concrete models where you declare the
component (e.g., a Var) and apply component-level operations (e.g.,
x.fix(0)) before you use the component members (in something like a
constraint).

You can silence this warning by one of three ways:
    1) Declare the component to be dense with the 'dense=True' option.
       This will cause all data objects to be immediately created and
       added to the Component.
    2) Defer component-level iteration until after the component data
       members have been added (through explicit use).
    3) If you intend to iterate over a component that may be empty, test
       if the component is empty first and avoid iteration in the case
       where it is empty.
""" % (self.name,) )

            if not hasattr(self._index, 'ordered') or not self._index.ordered:
                #
                # If the index set is not ordered, then return the
                # data iterator.  This is in an arbitrary order, which is
                # fine because the data is unordered.
                #
                return self._data.__iter__()
            else:
                #
                # Test each element of a sparse data with an ordered
                # index set in order.  This is potentially *slow*: if
                # the component is in fact very sparse, we could be
                # iterating over a huge (dense) index in order to sort a
                # small number of indices.  However, this provides a
                # consistent ordering that the user expects.
                #
                def _sparse_iter_gen(self):
                    for idx in self._index.__iter__():
                        if idx in self._data:
                            yield idx
                return _sparse_iter_gen(self)

    def keys(self):
        """Return a list of keys in the dictionary"""
        return [ x for x in self ]

    def values(self):
        """Return a list of the component data objects in the dictionary"""
        return [ self[x] for x in self ]

    def items(self):
        """Return a list (index,data) tuples from the dictionary"""
        return [ (x, self[x]) for x in self ]

    def iterkeys(self):
        """Return an iterator of the keys in the dictionary"""
        return self.__iter__()

    def itervalues(self):
        """Return an iterator of the component data objects in the dictionary"""
        for key in self:
            yield self[key]

    def iteritems(self):
        """Return an iterator of (index,data) tuples from the dictionary"""
        for key in self:
            yield key, self[key]

    def __getitem__(self, index):
        """
        This method returns the data corresponding to the given index.
        """
        if self._constructed is False:
            self._not_constructed_error(index)

        try:
            obj = self._data.get(index, _NotFound)
        except TypeError:
            obj = _NotFound
            index = self._processUnhashableIndex(index)
            if index.__class__ is _IndexedComponent_slice:
                return index

        if obj is _NotFound:
            # Not good: we have to defer this import to now
            # due to circular imports (expr imports _VarData
            # imports indexed_component, but we need expr
            # here
            from pyomo.core.expr import current as EXPR
            if index.__class__ is EXPR.GetItemExpression:
                return index
            index = self._validate_index(index)
            # _processUnhashableIndex could have found a slice, or
            # _validate could have found an Ellipsis and returned a
            # slicer
            if index.__class__ is _IndexedComponent_slice:
                return index
            obj = self._data.get(index, _NotFound)
            #
            # Call the _getitem_when_not_present helper to retrieve/return
            # the default value
            #
            if obj is _NotFound:
                return self._getitem_when_not_present(index)

        return obj

    def __setitem__(self, index, val):
        #
        # Set the value: This relies on _setitem_when_not_present() to
        # insert the correct ComponentData into the _data dictionary
        # when it is not present and _setitem_impl to update an existing
        # entry.
        #
        # Note: it is important that we check _constructed is False and not
        # just evaluates to false: when constructing immutable Params,
        # _constructed will be None during the construction process when
        # setting the value is valid.
        #
        if self._constructed is False:
            self._not_constructed_error(index)

        try:
            obj = self._data.get(index, _NotFound)
        except TypeError:
            obj = _NotFound
            index = self._processUnhashableIndex(index)

        if obj is _NotFound:
            # If we didn't find the index in the data, then we need to
            # validate it against the underlying set (as long as
            # _processUnhashableIndex didn't return a slicer)
            if index.__class__ is not _IndexedComponent_slice:
                index = self._validate_index(index)
        else:
            return self._setitem_impl(index, obj, val)
        #
        # Call the _setitem_impl helper to populate the _data
        # dictionary and set the value
        #
        # Note that we need to RECHECK the class against
        # _IndexedComponent_slice, as _validate_index could have found
        # an Ellipsis (which is hashable) and returned a slicer
        #
        if index.__class__ is _IndexedComponent_slice:
            # support "m.x[:,1] = 5" through a simple recursive call.
            #
            # Assert that this slice ws just generated
            assert len(index._call_stack) == 1
            #
            # Note that the slicer will only slice over *existing*
            # entries, but they may not be in the data dictionary.  Make
            # a copy of the slicer items *before* we start iterating
            # over it in case the setter changes the _data dictionary.
            for idx, obj in list(index.expanded_items()):
                self._setitem_impl(idx, obj, val)
        else:
            obj = self._data.get(index, _NotFound)
            if obj is _NotFound:
                return self._setitem_when_not_present(index, val)
            else:
                return self._setitem_impl(index, obj, val)

    def __delitem__(self, index):
        if self._constructed is False:
            self._not_constructed_error(index)

        try:
            obj = self._data.get(index, _NotFound)
        except TypeError:
            obj = _NotFound
            index = self._processUnhashableIndex(index)

        if obj is _NotFound:
            if index.__class__ is not _IndexedComponent_slice:
                index = self._validate_index(index)

        # this supports "del m.x[:,1]" through a simple recursive call
        if index.__class__ is _IndexedComponent_slice:
            # Assert that this slice ws just generated
            assert len(index._call_stack) == 1
            # Make a copy of the slicer items *before* we start
            # iterating over it (since we will be removing items!).
            for idx in list(index.expanded_keys()):
                del self[idx]
        else:
            # Handle the normal deletion operation
            if self.is_indexed():
                # Remove reference to this object
                self._data[index]._component = None
            del self._data[index]

    def _not_constructed_error(self, idx):
        # Generate an error because the component is not constructed
        if not self.is_indexed():
            idx_str = ''
        elif idx.__class__ is tuple:
            idx_str = "[" + ",".join(str(i) for i in idx) + "]"
        else:
            idx_str = "[" + str(idx) + "]"
        raise ValueError(
            "Error retrieving component %s%s: The component has "
            "not been constructed." % (self.name, idx_str,))

    def _validate_index(self, idx):
        if not IndexedComponent._DEFAULT_INDEX_CHECKING_ENABLED:
            # Return whatever index was provided if the global flag dictates
            # that we should bypass all index checking and domain validation
            return idx

        # This is only called through __{get,set,del}item__, which has
        # already trapped unhashable objects.
        if idx in self._index:
            # If the index is in the underlying index set, then return it
            #  Note: This check is potentially expensive (e.g., when the
            # indexing set is a complex set operation)!
            return idx

        if idx.__class__ is _IndexedComponent_slice:
            return idx

        if normalize_index.flatten:
            # Now we normalize the index and check again.  Usually,
            # indices will be already be normalized, so we defer the
            # "automatic" call to normalize_index until now for the
            # sake of efficiency.
            idx = normalize_index(idx)
            if idx in self._data:
                return idx
            if idx in self._index:
                return idx
        # There is the chance that the index contains an Ellipsis,
        # so we should generate a slicer
        if idx is Ellipsis or idx.__class__ is tuple and Ellipsis in idx:
            return self._processUnhashableIndex(idx)
        #
        # Generate different errors, depending on the state of the index.
        #
        if not self.is_indexed():
            raise KeyError(
                "Cannot treat the scalar component '%s'"
                "as an indexed component" % ( self.name, ))
        #
        # Raise an exception
        #
        raise KeyError(
            "Index '%s' is not valid for indexed component '%s'"
            % ( idx, self.name, ))

    def _processUnhashableIndex(self, idx):
        """Process a call to __getitem__ with unhashable elements

        There are three basic ways to get here:
          1) the index contains one or more slices or ellipsis
          2) the index contains an unhashable type (e.g., a Pyomo
             (Simple)Component
          3) the index contains an IndexTemplate
        """
        from pyomo.core.expr import current as EXPR
        #
        # Iterate through the index and look for slices and constant
        # components
        #
        fixed = {}
        sliced = {}
        ellipsis = None
        _found_numeric = False
        #
        # Setup the slice template (in fixed)
        #
        if type(idx) is tuple:
            # We would normally do "flatten()" here, but the current
            # (10/2016) implementation of flatten() is too aggressive:
            # it will attempt to expand *any* iterable, including
            # SimpleParam.
            idx = pyutilib.misc.flatten_tuple(idx)
        elif type(idx) is list:
            idx = pyutilib.misc.flatten_tuple(tuple(idx))
        else:
            idx = (idx,)

        for i,val in enumerate(idx):
            if type(val) is slice:
                if val.start is not None or val.stop is not None:
                    raise IndexError(
                        "Indexed components can only be indexed with simple "
                        "slices: start and stop values are not allowed.")
                if val.step is not None:
                    logger.warning(
                        "DEPRECATION WARNING: The special wildcard slice "
                        "(::0) is deprecated.  Please use an ellipsis (...) "
                        "to indicate '0 or more' indices")
                    val = Ellipsis
                else:
                    if ellipsis is None:
                        sliced[i] = val
                    else:
                        sliced[i-len(idx)] = val
                    continue

            if val is Ellipsis:
                if ellipsis is not None:
                    raise IndexError(
                        "Indexed components can only be indexed with simple "
                        "slices: the Pyomo wildcard slice (Ellipsis; "
                        "e.g., '...') can only appear once")
                ellipsis = i
                continue

            if hasattr(val, 'is_expression_type'):
                _num_val = val
                # Attempt to retrieve the numeric value .. if this
                # is a template expression generation, then it
                # should raise a TemplateExpressionError
                try:
                    val = EXPR.evaluate_expression(val, constant=True)
                    _found_numeric = True

                except TemplateExpressionError:
                    #
                    # The index is a template expression, so return the 
                    # templatized expression.
                    #
                    from pyomo.core.expr import current as EXPR
                    return EXPR.GetItemExpression(tuple(idx), self)

                except EXPR.NonConstantExpressionError:
                    #
                    # The expression contains an unfixed variable
                    #
                    raise RuntimeError(
"""Error retrieving the value of an indexed item %s:
index %s is not a constant value.  This is likely not what you meant to
do, as if you later change the fixed value of the object this lookup
will not change.  If you understand the implications of using
non-constant values, you can get the current value of the object using
the value() function.""" % ( self.name, i ))

                except EXPR.FixedExpressionError:
                    #
                    # The expression contains a fixed variable
                    #
                    raise RuntimeError(
"""Error retrieving the value of an indexed item %s:
index %s is a fixed but not constant value.  This is likely not what you
meant to do, as if you later change the fixed value of the object this
lookup will not change.  If you understand the implications of using
fixed but not constant values, you can get the current value using the
value() function.""" % ( self.name, i ))
                #
                # There are other ways we could get an exception such as
                # evaluating a Param / Var that is not initialized.
                # These exceptions will continue up the call stack.
                #

            # verify that the value is hashable
            hash(val)
            if ellipsis is None:
                fixed[i] = val
            else:
                fixed[i - len(idx)] = val

        if sliced or ellipsis is not None:
            return _IndexedComponent_slice(self, fixed, sliced, ellipsis)
        elif _found_numeric:
            if len(idx) == 1:
                return fixed[0]
            else:
                return tuple( fixed[i] for i in range(len(idx)) )
        else:
            raise DeveloperError(
                "Unknown problem encountered when trying to retrieve "
                "index for component %s" % (self.name,) )

    def _getitem_when_not_present(self, index):
        """Returns/initializes a value when the index is not in the _data dict.

        Override this method if the component allows implicit member
        construction.  For classes that do not support a 'default' (at
        this point, everything except Param and Var), requesting
        _getitem_when_not_present will generate a KeyError (just like a
        normal dict).

        Implementations may assume that the index has already been validated
        and is a legitimate entry in the _data dict.

        """
        raise KeyError(index)

    def _setitem_impl(self, index, obj, value):
        """Perform the fundamental object value storage

        Components that want to implement a nonstandard storage mechanism
        should override this method.

        Implementations may assume that the index has already been
        validated and is a legitimate pre-existing entry in the _data
        dict.

        """
        obj.set_value(value)
        return obj

    def _setitem_when_not_present(self, index, value):
        """Perform the fundamental component item creation and storage.

        Components that want to implement a nonstandard storage mechanism
        should override this method.

        Implementations may assume that the index has already been
        validated and is a legitimate entry in the _data dict.
        """
        #
        # If we are a scalar, then idx will be None (_validate_index ensures
        # this)
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        try:
            obj.set_value(value)
            return obj
        except:
            del self._data[index]
            raise

    def set_value(self, value):
        """Set the value of a scalar component."""
        if self.is_indexed():
            raise ValueError(
                "Cannot set the value for the indexed component '%s' "
                "without specifying an index value.\n"
                "\tFor example, model.%s[i] = value"
                % (self.name, self.name))
        else:
            raise DeveloperError(
                "Derived component %s failed to define set_value() "
                "for scalar instances."
                % (self.__class__.__name__,))

    def id_index_map(self):
        """
        Return an dictionary id->index for
        all ComponentData instances.
        """
        result = {}
        for index, component_data in iteritems(self):
            result[id(component_data)] = index
        return result


# In Python3, the items(), etc methods of dict-like things return
# generator-like objects.
if PY3:
    IndexedComponent.keys   = IndexedComponent.iterkeys
    IndexedComponent.values = IndexedComponent.itervalues
    IndexedComponent.items  = IndexedComponent.iteritems

class ActiveIndexedComponent(IndexedComponent, ActiveComponent):
    """
    This is the base class for all indexed modeling components
    whose data members are subclasses of ActiveComponentData, e.g.,
    can be activated or deactivated.

    The activate and deactivate methods activate both the
    component as well as all component data values.
    """

    def __init__(self, *args, **kwds):
        IndexedComponent.__init__(self, *args, **kwds)
        # Replicate the ActiveComponent.__init__() here.  We don't want
        # to use super, because that will run afoul of certain
        # assumptions for derived SimpleComponents' __init__()
        #
        # FIXME: eliminate multiple inheritance of SimpleComponents
        self._active = True

    def activate(self):
        """Set the active attribute to True"""
        super(ActiveIndexedComponent, self).activate()
        if self.is_indexed():
            for component_data in itervalues(self):
                component_data.activate()

    def deactivate(self):
        """Set the active attribute to False"""
        super(ActiveIndexedComponent, self).deactivate()
        if self.is_indexed():
            for component_data in itervalues(self):
                component_data.deactivate()


class _fill_in_known_wildcards(object):
    def __init__(self, wildcard_values):
        self.key = wildcard_values
        self.last_slice = None

    def __call__(self, _slice):
        if self.last_slice is _slice:
            raise StopIteration()
        self.last_slice = _slice

        if _slice.ellipsis is not None:
            raise RuntimeError(
                "Cannot lookup elements in a _ReferenceDict when the "
                "underlying slice object contains ellipsis")
        idx = tuple(
            _slice.fixed[i] if i in _slice.fixed else self.key.pop(0)
            for i in range(_slice.explicit_index_count))
        if idx in _slice.component:
            return _slice.component[idx]
        else:
            raise KeyError("KeyError: %s" % (idx,))


class _ReferenceDict(collections.MutableMapping):
    def __init__(self, component_slice):
        self._slice = component_slice

    def __getitem__(self, key):
        return self._find_element(self._slice, key)

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
        return self._find_element(tmp, key)

    def __delitem__(self, key):
        tmp = self._slice.duplicate()
        op = tmp._call_stack[-1][0]
        if op == _IndexedComponent_slice.get_item:
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.del_item,
                tmp._call_stack[-1][1] )
        elif op == _IndexedComponent_slice.slice_info:
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.del_item,
                tmp )
        elif op == _IndexedComponent_slice.get_attribute:
            tmp._call_stack[-1] = (
                _IndexedComponent_slice.del_attribute,
                tmp._call_stack[-1][1] )
        else:
            raise DeveloperError(
                "Unexpected slice _call_stack operation: %s" % op)
        return self._find_element(tmp, key)

    def __iter__(self):
        return self._slice.wildcard_keys()

    def __len__(self):
        return sum(1 for i in self._slice)

    def _find_element(self, _slice, key):
        key = list(pyutilib.misc.flatten_tuple(key))
        _iter = _IndexedComponent_slice_iter(
            _slice, _fill_in_known_wildcards(key))
        try:
            return advance_iterator(_iter)
        except StopIteration:
            raise KeyError("KeyError: %s" % (key,))
