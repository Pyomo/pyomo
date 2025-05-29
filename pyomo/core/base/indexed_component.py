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

import inspect
import logging
import sys
import textwrap

import pyomo.core.expr as EXPR
import pyomo.core.base as BASE
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.component import Component, ActiveComponent, ComponentData
from pyomo.core.base.config import PyomoOptions
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.core.expr.numeric_expr import _ndarray
from pyomo.core.pyomoobject import PyomoObject
from pyomo.common import DeveloperError
from pyomo.common.autoslots import fast_deepcopy
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types
from pyomo.common.sorting import sorted_robust

from collections.abc import Sequence

logger = logging.getLogger('pyomo.core')

sequence_types = {tuple, list}
slicer_types = {slice, Ellipsis.__class__, IndexedComponent_slice}


def normalize_index(x):
    """Normalize a component index.

    This flattens nested sequences into a single tuple.  There is a
    "global" flag (normalize_index.flatten) that will turn off index
    flattening across Pyomo.

    Scalar values will be returned unchanged.  Tuples with a single
    value will be unpacked and returned as a single value.

    Returns
    -------
    scalar or tuple

    """
    if x.__class__ in native_types:
        return x
    elif x.__class__ in sequence_types:
        # Note that casting a tuple to a tuple is cheap (no copy, no
        # new object)
        x = tuple(x)
    else:
        # Note: new Sequence types will be caught below and added to the
        # sequence_types set
        x = (x,)

    x_len = len(x)
    i = 0
    while i < x_len:
        _xi_class = x[i].__class__
        if _xi_class in native_types:
            i += 1
        elif _xi_class in sequence_types:
            x_len += len(x[i]) - 1
            # Note that casting a tuple to a tuple is cheap (no copy, no
            # new object)
            x = x[:i] + tuple(x[i]) + x[i + 1 :]
        elif issubclass(_xi_class, Sequence):
            if issubclass(_xi_class, str):
                # This is very difficult to get to: it would require a
                # user creating a custom derived string type
                native_types.add(_xi_class)
                i += 1
            else:
                sequence_types.add(_xi_class)
                x_len += len(x[i]) - 1
                x = x[:i] + tuple(x[i]) + x[i + 1 :]
        else:
            i += 1

    if x_len == 1:
        return x[0]
    return x


# Pyomo will normalize indices by default
normalize_index.flatten = True


class _NotFound(object):
    pass


class _NotSpecified(object):
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
                    *((None,) * i), component=component
                )
                i = None
                break
            except:
                pass
        if i is not None:
            # None of the generic positional arguments worked; raise an
            # exception
            component._data[index] = component._ComponentDataClass(component=component)
        try:
            ans = component._data[index].name
        except:
            ans = component.name + '[{unknown index}]'
        finally:
            del component._data[index]
    return ans


_rule_returned_none_error = """%s '%s': rule returned None.

%s rules must return either a valid expression, numeric value, or
%s.Skip.  The most common cause of this error is forgetting to
include the "return" statement at the end of your rule.
"""


def rule_result_substituter(result_map, map_types):
    _map = result_map
    if map_types is None:
        _map_types = set(type(key) for key in result_map)
    else:
        _map_types = map_types

    def rule_result_substituter_impl(rule, *args, **kwargs):
        if rule.__class__ in _map_types:
            #
            # The argument is a trivial type and will be mapped
            #
            value = rule
        elif isinstance(rule, PyomoObject):
            #
            # The argument is a Pyomo component.  This can happen when
            # the rule isn't a rule at all, but instead the decorator
            # was used as a function to wrap an inline definition (not
            # something I think we should support, but exists in some
            # [old] examples).
            #
            return rule
        else:
            #
            # Otherwise, the argument is a functor, so call it to
            # generate the rule result.
            #
            value = rule(*args, **kwargs)
        #
        # Map the returned value:
        #
        if value.__class__ in _map_types and value in _map:
            return _map[value]
        return value

    return rule_result_substituter_impl


_map_rule_funcdef = """def wrapper_function%s:
    args, varargs, kwds, local_env = inspect.getargvalues(
        inspect.currentframe())
    args = tuple(local_env[_] for _ in args) + (varargs or ())
    return wrapping_fcn(rule, *args, **(kwds or {}))
"""


def rule_wrapper(rule, wrapping_fcn, positional_arg_map=None, map_types=None):
    """Wrap a rule with another function

    This utility method provides a way to wrap a function (rule) with
    another function while preserving the original function signature.
    This is important for rules, as the :py:func:`Initializer`
    argument processor relies on knowing the number of positional
    arguments.

    Parameters
    ----------
    rule: function
        The original rule being wrapped
    wrapping_fcn: function or Dict
        The wrapping function.  The `wrapping_fcn` will be called with
        ``(rule, *args, **kwargs)``.  For convenience, if a `dict` is
        passed as the `wrapping_fcn`, then the result of
        :py:func:`rule_result_substituter(wrapping_fcn)` is used as the
        wrapping function.
    positional_arg_map: iterable[int]
        An iterable of indices of rule positional arguments to expose in
        the wrapped function signature.  For example,
        `positional_arg_map=(2, 0)` and `rule=fcn(a, b, c)` would produce a
        wrapped function with a signature `wrapper_function(c, a)`

    """
    if isinstance(wrapping_fcn, dict):
        wrapping_fcn = rule_result_substituter(wrapping_fcn, map_types)
        if not inspect.isfunction(rule):
            return wrapping_fcn(rule)
    # Because some of our processing of initializer functions relies on
    # knowing the number of positional arguments, we will go to extra
    # effort here to preserve the original function signature.
    rule_sig = inspect.signature(rule)
    if positional_arg_map is not None:
        param = list(rule_sig.parameters.values())
        rule_sig = rule_sig.replace(parameters=(param[i] for i in positional_arg_map))
    _funcdef = _map_rule_funcdef % (str(rule_sig),)
    # Create the wrapper in a temporary environment that mimics this
    # function's environment.
    _env = dict(globals())
    _env.update(locals())
    exec(_funcdef, _env)
    return _env['wrapper_function']


class IndexedComponent(Component):
    """This is the base class for all indexed modeling components.
    This class stores a dictionary, self._data, that maps indices
    to component data objects.  The object self._index_set defines valid
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

        _data:  A dictionary from the index set to component data objects

        _index_set:  The set of valid indices

        _anonymous_sets: A ComponentSet of "anonymous" sets used by this
            component.  Anonymous sets are Set / SetOperator / RangeSet
            that compose attributes like _index_set, but are not
            themselves explicitly assigned (and named) on any Block

    """

    class Skip(object):
        pass

    #
    # If an index is supplied for which there is not a _data entry
    # (specifically, in a get call), then this flag determines whether
    # a check is performed to see if the input index is in the
    # index set _index_set. This is extremely expensive, and so this flag
    # is provided to disable that feature globally.
    #
    _DEFAULT_INDEX_CHECKING_ENABLED = True

    def __init__(self, *args, **kwds):
        #
        kwds.pop('noruleinit', None)
        Component.__init__(self, **kwds)
        #
        self._data = {}
        #
        if len(args) == 0 or (args[0] is UnindexedComponent_set and len(args) == 1):
            #
            # If no indexing sets are provided, generate a dummy index
            #
            self._index_set = UnindexedComponent_set
            self._anonymous_sets = None
        elif len(args) == 1:
            #
            # If a single indexing set is provided, just process it.
            #
            self._index_set, self._anonymous_sets = BASE.set.process_setarg(args[0])
        else:
            #
            # If multiple indexing sets are provided, process them all,
            # and store the cross-product of these sets.
            #
            # Example: Pyomo allows things like "Param([1,2,3],
            # range(100), initialize=0)".  This needs to create *3*
            # sets: two SetOf components and then the SetProduct.  As
            # the user declined to name any of these sets, we will not
            # make up names and instead store them on the model as
            # "anonymous components"
            #
            self._index_set = BASE.set.SetProduct(*args)
            self._anonymous_sets = ComponentSet((self._index_set,))
            if self._index_set._anonymous_sets is not None:
                self._anonymous_sets.update(self._index_set._anonymous_sets)

    def _create_objects_for_deepcopy(self, memo, component_list):
        _new = self.__class__.__new__(self.__class__)
        _ans = memo.setdefault(id(self), _new)
        if _ans is _new:
            component_list.append((self, _new))
            # For indexed components, we will pre-emptively clone all
            # component data objects as well (as those are the objects
            # that will be referenced by things like expressions).  It
            # is important to only clone "normal" ComponentData objects:
            # so we will want to skip this for all scalar components
            # (where the _data points back to self) and references
            # (where the data may be stored outside this block tree and
            # therefore may not be cloned)
            if self.is_indexed() and not self.is_reference():
                # Because we are already checking / updating the memo
                # for the _data dict, we can effectively "deepcopy" it
                # right now (almost for free!)
                _src = self._data
                memo[id(_src)] = _new._data = _src.__class__()
                _setter = _new._data.__setitem__
                for idx, obj in _src.items():
                    _setter(
                        fast_deepcopy(idx, memo),
                        obj._create_objects_for_deepcopy(memo, component_list),
                    )

        return _ans

    def to_dense_data(self):
        """TODO"""
        for idx in self._index_set:
            if idx in self._data:
                continue
            try:
                self._getitem_when_not_present(idx)
            except KeyError:
                # Rule could have returned Skip, which we will silently ignore
                pass

    def clear(self):
        """Clear the data in this component"""
        if self.is_indexed():
            self._data = {}
        else:
            raise DeveloperError(
                "Derived scalar component %s failed to define clear()."
                % (self.__class__.__name__,)
            )

    def index_set(self):
        """Return the index set"""
        return self._index_set

    def is_indexed(self):
        """Return true if this component is indexed"""
        return self._index_set is not UnindexedComponent_set

    def is_reference(self):
        """Return True if this component is a reference, where
        "reference" is interpreted as any component that does not
        own its own data.
        """
        return self._data is not None and type(self._data) is not dict

    def dim(self):
        """Return the dimension of the index"""
        if not self.is_indexed():
            return 0
        return self._index_set.dimen

    def __len__(self):
        """
        Return the number of component data objects stored by this
        component.
        """
        return len(self._data)

    def __contains__(self, idx):
        """Return true if the index is in the dictionary"""
        return idx in self._data

    # The default implementation is for keys() and __iter__ to be
    # synonyms.  The logic is implemented in keys() so that
    # keys/values/items continue to work for components that implement
    # other definitions for __iter__ (e.g., Set)
    def __iter__(self):
        """Return an iterator of the component data keys"""
        return self.keys()

    def keys(self, sort=SortComponents.UNSORTED, ordered=NOTSET):
        """Return an iterator over the component data keys

        This method sets the ordering of component data objects within
        this IndexedComponent container.  For consistency,
        :py:meth:`__init__()`, :py:meth:`values`, and :py:meth:`items`
        all leverage this method to ensure consistent ordering.

        Parameters
        ----------
        sort: bool or SortComponents
            Iterate over the declared component keys in a specified
            sorted order.  See :py:class:`SortComponents` for valid
            options and descriptions.

        ordered: bool
            DEPRECATED: Please use `sort=SortComponents.ORDERED_INDICES`.
            If True, then the keys are returned in a deterministic order
            (using the underlying set's `ordered_iter()`).

        """
        sort = SortComponents(sort)
        if ordered is not NOTSET:
            deprecation_warning(
                f"keys(ordered={ordered}) is deprecated.  "
                "Please use `sort=SortComponents.ORDERED_INDICES`",
                version='6.6.0',
            )
            if ordered:
                sort = sort | SortComponents.ORDERED_INDICES
        if not self._index_set.isfinite():
            #
            # If the index set is virtual (e.g., Any) then return the
            # data iterator.  Note that since we cannot check the length
            # of the underlying Set, there should be no warning if the
            # user iterates over the set when the _data dict is empty.
            #
            if (
                SortComponents.SORTED_INDICES in sort
                or SortComponents.ORDERED_INDICES in sort
            ):
                return iter(sorted_robust(self._data))
            else:
                return self._data.__iter__()

        if SortComponents.SORTED_INDICES in sort:
            ans = self._index_set.sorted_iter()
        elif SortComponents.ORDERED_INDICES in sort:
            ans = self._index_set.ordered_iter()
        else:
            ans = iter(self._index_set)

        if self._data.__class__ is not dict:
            # We currently only need to worry about sparse data
            # structures when the underlying _data is a dict.  Avoiding
            # the len() and filter() below is especially important for
            # References (where both can be expensive linear-time
            # operations)
            pass
        elif len(self) == len(self._index_set):
            #
            # If the data is dense then return the index iterator.
            #
            pass
        elif not self._data and self._index_set and PyomoOptions.paranoia_level:
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
"""
                % (self.name,)
            )
        else:
            #
            # Test each element of a sparse data with an ordered
            # index set in order.  This is potentially *slow*: if
            # the component is in fact very sparse, we could be
            # iterating over a huge (dense) index in order to sort a
            # small number of indices.  However, this provides a
            # consistent ordering that the user expects.
            #
            ans = filter(self._data.__contains__, ans)
        return ans

    def values(self, sort=SortComponents.UNSORTED, ordered=NOTSET):
        """Return an iterator of the component data objects

        Parameters
        ----------
        sort: bool or SortComponents
            Iterate over the declared component values in a specified
            sorted order.  See :py:class:`SortComponents` for valid
            options and descriptions.

        ordered: bool
            DEPRECATED: Please use `sort=SortComponents.ORDERED_INDICES`.
            If True, then the values are returned in a deterministic order
            (using the underlying set's `ordered_iter()`.
        """
        if ordered is not NOTSET:
            deprecation_warning(
                f"values(ordered={ordered}) is deprecated.  "
                "Please use `sort=SortComponents.ORDERED_INDICES`",
                version='6.6.0',
            )
            if ordered:
                sort = SortComponents(sort) | SortComponents.ORDERED_INDICES
        # Note that looking up the values in a reference may be an
        # expensive operation (linear time).  To avoid making this a
        # quadratic time operation, we will leverage _ReferenceDict's
        # values().  This may fail for references created from mappings
        # or sequences, raising the TypeError
        if self.is_reference():
            try:
                return self._data.values(sort)
            except TypeError:
                pass
        return map(self.__getitem__, self.keys(sort))

    def items(self, sort=SortComponents.UNSORTED, ordered=NOTSET):
        """Return an iterator of (index,data) component data tuples

        Parameters
        ----------
        sort: bool or SortComponents
            Iterate over the declared component items in a specified
            sorted order.  See :py:class:`SortComponents` for valid
            options and descriptions.

        ordered: bool
            DEPRECATED: Please use `sort=SortComponents.ORDERED_INDICES`.
            If True, then the items are returned in a deterministic order
            (using the underlying set's `ordered_iter()`.
        """
        if ordered is not NOTSET:
            deprecation_warning(
                f"items(ordered={ordered}) is deprecated.  "
                "Please use `sort=SortComponents.ORDERED_INDICES`",
                version='6.6.0',
            )
            if ordered:
                sort = SortComponents(sort) | SortComponents.ORDERED_INDICES
        # Note that looking up the values in a reference may be an
        # expensive operation (linear time).  To avoid making this a
        # quadratic time operation, we will try and use _ReferenceDict's
        # items().  This may fail for references created from mappings
        # or sequences, raising the TypeError
        if self.is_reference():
            try:
                return self._data.items(sort)
            except TypeError:
                pass
        return ((s, self[s]) for s in self.keys(sort))

    @deprecated('The iterkeys method is deprecated. Use dict.keys().', version='6.0')
    def iterkeys(self):
        """Return a list of keys in the dictionary"""
        return self.keys()

    @deprecated(
        'The itervalues method is deprecated. Use dict.values().', version='6.0'
    )
    def itervalues(self):
        """Return a list of the component data objects in the dictionary"""
        return self.values()

    @deprecated('The iteritems method is deprecated. Use dict.items().', version='6.0')
    def iteritems(self):
        """Return a list (index,data) tuples from the dictionary"""
        return self.items()

    def __getitem__(self, index) -> ComponentData:
        """
        This method returns the data corresponding to the given index.
        """
        if self._constructed is False:
            self._not_constructed_error(index)

        try:
            return self._data[index]
        except KeyError:
            obj = _NotFound
        except TypeError:
            try:
                index = self._processUnhashableIndex(index)
            except TypeError:
                # This index is really unhashable.  Set a flag so that
                # we can re-raise the original exception (not this one)
                index = TypeError
            if index is TypeError:
                raise
            if index.__class__ is IndexedComponent_slice:
                return index
            # The index could have contained constant but nonhashable
            # objects (e.g., scalar immutable Params).
            # _processUnhashableIndex will evaluate those constants, so
            # if it made any changes to the index, we need to re-check
            # the _data dict for membership.
            try:
                obj = self._data.get(index, _NotFound)
            except TypeError:
                obj = _NotFound

        if obj is _NotFound:
            if isinstance(index, EXPR.GetItemExpression):
                return index
            validated_index = self._validate_index(index)
            if validated_index is not index:
                index = validated_index
                # _processUnhashableIndex could have found a slice, or
                # _validate could have found an Ellipsis and returned a
                # slicer
                if index.__class__ is IndexedComponent_slice:
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
            if index.__class__ is not IndexedComponent_slice:
                index = self._validate_index(index)
        else:
            return self._setitem_impl(index, obj, val)
        #
        # Call the _setitem_impl helper to populate the _data
        # dictionary and set the value
        #
        # Note that we need to RECHECK the class against
        # IndexedComponent_slice, as _validate_index could have found
        # an Ellipsis (which is hashable) and returned a slicer
        #
        if index.__class__ is IndexedComponent_slice:
            # support "m.x[:,1] = 5" through a simple recursive call.
            #
            # Assert that this slice was just generated
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
            if index.__class__ is not IndexedComponent_slice:
                index = self._validate_index(index)

        # this supports "del m.x[:,1]" through a simple recursive call
        if index.__class__ is IndexedComponent_slice:
            # Assert that this slice was just generated
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

    def _construct_from_rule_using_setitem(self):
        if self._rule is None:
            return
        index = None  # set so it is defined for scalars for `except:` below
        rule = self._rule
        block = self.parent_block()
        try:
            if rule.constant() and self.is_indexed():
                # A constant rule could return a dict-like thing or
                # matrix that we would then want to process with
                # Initializer().  If the rule actually returned a
                # constant, then this is just a little overhead.
                self._rule = rule = Initializer(
                    rule(block, None),
                    treat_sequences_as_mappings=False,
                    arg_not_specified=NOTSET,
                )

            if rule.contains_indices():
                # The index is coming in externally; we need to validate it
                for index in rule.indices():
                    self[index] = rule(block, index)
            elif not self.index_set().isfinite():
                # If the index is not finite, then we cannot iterate
                # over it.  Since the rule doesn't provide explicit
                # indices, then there is nothing we can do (the
                # assumption is that the user will trigger specific
                # indices to be created at a later time).
                pass
            elif rule.constant():
                # Slight optimization: if the initializer is known to be
                # constant, then only call the rule once.
                val = rule(block, None)
                for index in self.index_set():
                    self._setitem_when_not_present(index, val)
            else:
                for index in self.index_set():
                    self._setitem_when_not_present(index, rule(block, index))
        except:
            err = sys.exc_info()[1]
            logger.error(
                "Rule failed for %s '%s' with index %s:\n%s: %s"
                % (self.ctype.__name__, self.name, str(index), type(err).__name__, err)
            )
            raise

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
            "not been constructed." % (self.name, idx_str)
        )

    def _validate_index(self, idx):
        if not IndexedComponent._DEFAULT_INDEX_CHECKING_ENABLED:
            # Return whatever index was provided if the global flag dictates
            # that we should bypass all index checking and domain validation
            return idx

        # This is only called through __{get,set,del}item__, which has
        # already trapped unhashable objects.  Unfortunately, Python
        # 3.12 made slices hashable.  This means that slices will get
        # here and potentially be looked up in the index_set.  This will
        # cause problems with Any, where Any will happily return the
        # index as a valid set.  We will only validate the index for
        # non-Any sets.  Any will pass through so that normalize_index
        # can be called (which can generate the TypeError for slices)
        _any = isinstance(self._index_set, BASE.set._AnySet)
        if _any:
            validated_idx = _NotFound
        else:
            validated_idx = self._index_set.get(idx, _NotFound)
            if validated_idx is not _NotFound:
                # If the index is in the underlying index set, then return it
                #  Note: This check is potentially expensive (e.g., when the
                # indexing set is a complex set operation)!
                return validated_idx

        if normalize_index.flatten:
            # Now we normalize the index and check again.  Usually,
            # indices will be already be normalized, so we defer the
            # "automatic" call to normalize_index until now for the
            # sake of efficiency.
            normalized_idx = normalize_index(idx)
            if normalized_idx is not idx and not _any:
                if normalized_idx in self._data:
                    return normalized_idx
                if normalized_idx in self._index_set:
                    return normalized_idx
        else:
            normalized_idx = idx

        # There is the chance that the index contains an Ellipsis,
        # so we should generate a slicer
        if (
            normalized_idx.__class__ in slicer_types
            or normalized_idx.__class__ is tuple
            and any(_.__class__ in slicer_types for _ in normalized_idx)
        ):
            return self._processUnhashableIndex(normalized_idx)
        if _any:
            return idx
        #
        # Generate different errors, depending on the state of the index.
        #
        if not self.is_indexed():
            raise KeyError(
                "Cannot treat the scalar component '%s' "
                "as an indexed component" % (self.name,)
            )
        #
        # Raise an exception
        #
        raise KeyError(
            "Index '%s' is not valid for indexed component '%s'"
            % (normalized_idx, self.name)
        )

    def _processUnhashableIndex(self, idx):
        """Process a call to __getitem__ with unhashable elements

        There are three basic ways to get here:
          1) the index contains one or more slices or ellipsis
          2) the index contains an unhashable type (e.g., a Pyomo
             (Scalar)Component)
          3) the index contains an IndexTemplate
        """
        #
        # Iterate through the index and look for slices and constant
        # components
        #
        orig_idx = idx
        fixed = {}
        sliced = {}
        ellipsis = None
        #
        # Setup the slice template (in fixed)
        #
        if normalize_index.flatten:
            idx = normalize_index(idx)
        if idx.__class__ is not tuple:
            idx = (idx,)

        for i, val in enumerate(idx):
            if type(val) is slice:
                if (
                    val.start is not None
                    or val.stop is not None
                    or val.step is not None
                ):
                    raise IndexError(
                        "Indexed components can only be indexed with simple "
                        "slices: start and stop values are not allowed."
                    )
                else:
                    if ellipsis is None:
                        sliced[i] = val
                    else:
                        sliced[i - len(idx)] = val
                    continue

            if val is Ellipsis:
                if ellipsis is not None:
                    raise IndexError(
                        "Indexed components can only be indexed with simple "
                        "slices: the Pyomo wildcard slice (Ellipsis; "
                        "e.g., '...') can only appear once"
                    )
                ellipsis = i
                continue

            if hasattr(val, 'is_expression_type'):
                _num_val = val
                # Attempt to retrieve the numeric value .. if this
                # is a template expression generation, then it
                # should raise a TemplateExpressionError
                try:
                    val = EXPR.evaluate_expression(val, constant=True)
                except TemplateExpressionError:
                    #
                    # The index is a template expression, so return the
                    # templatized expression.
                    #
                    return EXPR.GetItemExpression((self,) + tuple(idx))

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
the value() function."""
                        % (self.name, i)
                    )

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
value() function."""
                        % (self.name, i)
                    )
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
            slice_dim = len(idx)
            if ellipsis is not None:
                slice_dim -= 1
            if normalize_index.flatten:
                set_dim = self.dim()
            elif not self.is_indexed():
                # Scalar component.
                set_dim = 0
            else:
                set_dim = self.index_set().dimen
                if set_dim is None:
                    set_dim = 1

            structurally_valid = False
            if slice_dim == set_dim or set_dim is None:
                structurally_valid = True
            elif type(set_dim) is type:
                pass  # UnknownSetDimen
            elif ellipsis is not None and slice_dim < set_dim:
                structurally_valid = True
            elif set_dim == 0 and idx == (slice(None),):
                # If dim == 0 and idx is slice(None), the component was
                # a scalar passed a single slice. Since scalar components
                # can be accessed with a "1-dimensional" index of None,
                # this behavior is allowed.
                #
                # Note that x[...] is caught above, as slice_dim will be
                # 0 in that case
                structurally_valid = True

            if not structurally_valid:
                msg = (
                    "Index %s contains an invalid number of entries for "
                    "component '%s'. Expected %s, got %s."
                )
                if type(set_dim) is type:
                    set_dim = set_dim.__name__
                    msg += '\n    ' + '\n    '.join(
                        textwrap.wrap(
                            textwrap.dedent(
                                """
                                Slicing components relies on knowing the
                                underlying set dimensionality (even if the
                                dimensionality is None).  The underlying
                                component set ('%s') dimensionality has not been
                                determined (likely because it is an empty Set).
                                You can avoid this error by specifying the Set
                                dimensionality (with the 'dimen=' keyword)."""
                                % (self.index_set(),)
                            ).strip()
                        )
                    )
                raise IndexError(
                    msg
                    % (
                        IndexedComponent_slice._getitem_args_to_str(list(idx)),
                        self.name,
                        set_dim,
                        slice_dim,
                    )
                )
            return IndexedComponent_slice(self, fixed, sliced, ellipsis)
        elif len(idx) == len(fixed):
            if len(idx) == 1:
                return fixed[0]
            else:
                return tuple(fixed[i] for i in range(len(idx)))
        else:
            raise DeveloperError(
                "Unknown problem encountered when trying to retrieve "
                f"index '{orig_idx}' for component '{self.name}'"
            )

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
        if value is IndexedComponent.Skip:
            del self[index]
            return None
        else:
            obj.set_value(value)
        return obj

    def _setitem_when_not_present(self, index, value=_NotSpecified):
        """Perform the fundamental component item creation and storage.

        Components that want to implement a nonstandard storage mechanism
        should override this method.

        Implementations may assume that the index has already been
        validated and is a legitimate entry to add to the _data dict.
        """
        # If the value is "Skip" do not add anything
        if value is IndexedComponent.Skip:
            return None
        #
        # If we are a scalar, then idx will be None (_validate_index ensures
        # this)
        if index is None and not self.is_indexed():
            obj = self._data[index] = self
        else:
            obj = self._data[index] = self._ComponentDataClass(component=self)
        obj._index = index
        try:
            if value is not _NotSpecified:
                obj.set_value(value)
        except:
            self._data.pop(index, None)
            raise
        return obj

    def set_value(self, value):
        """Set the value of a scalar component."""
        if self.is_indexed():
            raise ValueError(
                "Cannot set the value for the indexed component '%s' "
                "without specifying an index value.\n"
                "\tFor example, model.%s[i] = value" % (self.name, self.name)
            )
        else:
            raise DeveloperError(
                "Derived component %s failed to define set_value() "
                "for scalar instances." % (self.__class__.__name__,)
            )

    def _pprint(self):
        """Print component information."""
        return (
            [
                ("Size", len(self)),
                ("Index", self._index_set if self.is_indexed() else None),
            ],
            self._data.items(),
            ("Object",),
            lambda k, v: [type(v)],
        )

    def id_index_map(self):
        """
        Return an dictionary id->index for
        all ComponentData instances.
        """
        result = {}
        for index, component_data in self.items():
            result[id(component_data)] = index
        return result


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
            for component_data in self.values():
                component_data.activate()

    def deactivate(self):
        """Set the active attribute to False"""
        super(ActiveIndexedComponent, self).deactivate()
        if self.is_indexed():
            for component_data in self.values():
                component_data.deactivate()


# Ideally, this would inherit from np.lib.mixins.NDArrayOperatorsMixin,
# but doing so overrides things like __contains__ in addition to the
# operators that we are interested in.
class IndexedComponent_NDArrayMixin(object):
    """Support using IndexedComponent with numpy.ndarray

    This IndexedComponent mixin class adds support for implicitly using
    the IndexedComponent as a term in an expression with numpy ndarray
    objects.

    """

    def __array__(self, dtype=None, copy=None):
        if dtype not in (None, object):
            raise ValueError(
                "Pyomo IndexedComponents can only be converted to NumPy "
                f"arrays with dtype=object (received {dtype=})"
            )
        if copy is not None and not copy:
            raise ValueError(
                "Pyomo IndexedComponents do not support conversion to NumPy "
                "arrays without generating a new array"
            )
        if not self.is_indexed():
            ans = _ndarray.NumericNDArray(shape=(1,), dtype=object)
            ans[0] = self
            return ans.reshape(())

        _dim = self.dim()
        if _dim is None:
            raise TypeError(
                "Cannot convert a non-dimensioned Pyomo IndexedComponent "
                "(%s) into a numpy array" % (self,)
            )
        bounds = self.index_set().bounds()
        if not isinstance(bounds[0], Sequence):
            bounds = ((bounds[0],), (bounds[1],))
        if any(b != 0 for b in bounds[0]):
            raise TypeError(
                "Cannot convert a Pyomo IndexedComponent "
                "(%s) with bounds [%s, %s] into a numpy array"
                % (self, bounds[0], bounds[1])
            )
        shape = tuple(b + 1 for b in bounds[1])
        ans = _ndarray.NumericNDArray(shape=shape, dtype=object)
        for k, v in self.items():
            ans[k] = v
        return ans

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _ndarray.NumericNDArray.__array_ufunc__(
            None, ufunc, method, *inputs, **kwargs
        )
