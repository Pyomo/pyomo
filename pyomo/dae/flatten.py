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
"""A module for "flattening" the components in a block-hierarchical model
with respect to common indexing sets

"""

from pyomo.core.base import Block, Reference
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.set import SetProduct
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from collections import OrderedDict


def get_slice_for_set(s):
    """
    Get the slice of the proper dimension for a set.
    """
    if s.dimen != 0:
        if not normalize_index.flatten:
            return slice(None)
        else:
            if s.dimen is not None:
                # We will arrive here and fail for sets of dimension
                # UnknownSetDimen.
                return (slice(None),) * s.dimen
            else:
                return (Ellipsis,)
    else:
        # Case for e.g. UnindexedComponent_set
        # Should this be None or tuple()? RBP 202110
        return None


class _NotAnIndex(object):
    """
    `None` is a valid index, so we use a dummy class to
    denote a slot that needs to get filled with indices
    from our product.
    """

    pass


def _fill_indices(filled_index, index):
    """
    `filled_index` is a list with some entries `_NotAnIndex`.
    We fill those entries with values from `index`, a tuple.
    """
    j = 0
    for i, val in enumerate(filled_index):
        if val is _NotAnIndex:
            filled_index[i] = index[j]
            # `index` is always a tuple, so this is valid
            j += 1
    # Make sure `filled_index` had the same number of vacancies
    # as the original SetProduct has factors. Not _strictly_ necessary.
    assert j == len(index)
    filled_index = tuple(filled_index)

    if len(filled_index) == 1:
        return filled_index[0]
    else:
        return filled_index


def _fill_indices_from_product(partial_index_list, product):
    """
    `partial_index_list` is a list of indices, each corresponding to a
    set. If an entry in `partial_index_list` is `_NotAnIndex`, that
    slot will get filled in by an entry from `product`.

    `product` is a `SetProduct` with as many "factors" as there are
    missing indices in `partial_index_list`.
    """
    # We will manipulate `normalize_index.flatten`.
    # Store its original value so we can reset it when we're done.
    _normalize_index_flatten = normalize_index.flatten
    try:
        normalize_index.flatten = False
        for index in product:
            # Since `normalize_index.flatten` is False, `index` is a
            # scalar or (tuple of (scalars or tuples)). Conveniently,
            # each entry in the tuple belongs to a single factor set.
            # I do not have to worry about having a product-of-products
            # here because I created the product from "unfactorable sets"

            # We need to generate a new index for every entry of `product`,
            # and want to reuse `partial_index_list` as a starting point,
            # so we copy it here.
            filled_index = list(partial_index_list)

            normalize_index.flatten = _normalize_index_flatten

            # filled_index can now be used in the user's intended way
            # This determines how we will access the component's data
            # with our new index, which is currently _completely_ unflattened
            # (i.e. a tuple-of-tuples, no further nesting).
            #
            # This will not work well if the user's component is indexed by
            # a nested product of sets AND normalize_index.flatten is False.
            # In this case we may try to access a component's data with
            # >>> comp[(1,'a',1)]
            # (each coordinate belongs to its own set) when it expects
            # >>> comp[((1,'a'),1)]
            # because `comp` was created with two set arguments, the first
            # of which was already a product.

            yield (index, _fill_indices(filled_index, index))

            normalize_index.flatten = False
            # Want to get the unflattened factors when we advance the
            # iterator of `product`
    finally:
        # Reset `normalize_index.flatten`
        normalize_index.flatten = _normalize_index_flatten


def slice_component_along_sets(component, sets, context_slice=None, normalize=None):
    """This function generates all possible slices of the provided component
    along the provided sets. That is, it will iterate over the component's
    other indexing sets and, for each index, yield a slice along the
    sets specified in the call signature.

    Arguments
    ---------
    component: Component
        The component whose slices will be yielded
    sets: ComponentSet
        ComponentSet of Pyomo sets that will be sliced along
    context_slice: IndexedComponent_slice
        If provided, instead of creating a new slice, we will extend this
        one with appropriate getattr and getitem calls.
    normalize: Bool
        If False, the returned index (from the product of "other sets")
        is not normalized, regardless of the value of normalize_index.flatten.
        This is necessary to use this index with _fill_indices.

    Yields
    ------
    tuple
        The first entry is the index in the product of "other sets"
        corresponding to the slice, and the second entry is the slice
        at that index.

    """
    set_set = ComponentSet(sets)
    subsets = list(component.index_set().subsets())
    temp_idx = [get_slice_for_set(s) if s in set_set else _NotAnIndex for s in subsets]
    other_sets = [s for s in subsets if s not in set_set]

    if context_slice is None:
        base_component = component
    else:
        base_component = getattr(context_slice, component.local_name)

    if component.is_indexed():
        # We need to iterate over sets that aren't sliced
        # `c.is_indexed()` covers the case when UnindexedComponent_set
        # is in `other_sets`.
        if other_sets:
            cross_prod = other_sets[0].cross(*other_sets[1:])
        else:
            # If we are only indexed by sets we need to slice, we
            # should just use tuple(temp_idx) as our index. We spoof
            # a cross_prod here so we don't have to repeat the try/except
            # logic below in a separate branch. An empty tuple is the right
            # singleton to work in the embedded call to _fill_indices.
            cross_prod = [tuple()]

        for prod_index, new_index in _fill_indices_from_product(temp_idx, cross_prod):
            try:
                if normalize_index.flatten:
                    # This index is always normalized if normalize_index.flatten
                    # is True. I have not encountered a situation where
                    # "denormalization" makes sense here.
                    # As normalization is also done in the IndexedComponent,
                    # normalizing here primarily just affects what the resulting
                    # slice "looks like." E.g. slice(None) vs (slice(None),).
                    # This has implications for generating CUIDs from these
                    # slices, where we would like consistency in the string
                    # representation.
                    # TODO: Should CUID normalize (slice(None),)?
                    new_index = normalize_index(new_index)
                c_slice = base_component[new_index]
                if type(c_slice) is IndexedComponent_slice:
                    # This is just to make sure we do not have an
                    # empty slice.
                    #
                    # Note that c_slice is not necessarily a slice.
                    # We enter this loop even if no sets need slicing.
                    try:
                        next(iter(c_slice.duplicate()))
                    except IndexError:
                        if normalize_index.flatten:
                            raise
                        # There is an edge case where when we are not
                        # flattening indices the dimensionality of an
                        # index can change between a SetProduct and the
                        # member Sets: the member set can have dimen>1
                        # (or even None!), but the dimen of that portion
                        # of the SetProduct is always 1.  Since we are
                        # just checking that the c_slice isn't
                        # completely empty, we will allow matching with
                        # an Ellipsis
                        _empty = True
                        try:
                            next(iter(base_component[...]))
                            _empty = False
                        except:
                            pass
                        if _empty:
                            raise
                if (normalize is None and normalize_index.flatten) or normalize:
                    # Most users probably want this index to be normalized,
                    # so they can more conveniently use it as a key in a
                    # mapping. (E.g. they will get "a" as opposed to ("a",).)
                    # However, to use it in the calling routine
                    # generate_sliced_components, we need this index to not
                    # have been normalized, so that indices are tuples,
                    # partitioned according to their "factor sets."
                    # This is why we allow the argument normalize=False to
                    # override normalize_index.flatten.
                    prod_index = normalize_index(prod_index)
                yield prod_index, c_slice
            except StopIteration:
                # We have an empty slice for some reason, e.g.
                # a coordinate of `new_index` from the cross
                # product was skipped in the original component.
                pass
            except KeyError:
                # We are creating scalar components from a product of
                # sets. Components may be undefined for certain indices.
                # We want to simply skip that index and move on.
                pass
    else:
        # Component is a data object
        c_slice = base_component
        yield (), c_slice


def generate_sliced_components(
    b, index_stack, slice_, sets, ctype, index_map, active=None
):
    """Recursively generate slices of the specified ctype along the
    specified sets

    Parameters
    ----------

    b: BlockData
        Block whose components will be sliced

    index_stack: list
        Sets above ``b`` in the block hierarchy, including on its parent
        component, that have been sliced. This is necessary to return the
        sets that have been sliced.

    slice_: IndexedComponent_slice or BlockData
        Slice generated so far.  This function will yield extensions to
        this slice at the current level of the block hierarchy.

    sets: ComponentSet of Pyomo sets
        Sets that will be sliced

    ctype: Subclass of Component
        Type of components to generate

    index_map: ComponentMap
        Map from (some of) the specified sets to a "representative index"
        to use when descending into subblocks. While this map does not need
        to contain every set in the sliced sets, it must not contain any
        sets that will not be sliced.

    active: Bool or None
        If not None, this is a boolean flag used to filter component objects
        by their active status.

    Yields
    ------

    Tuple of Sets and an IndexedComponent_slice or ComponentData
        The sets indexing the returned component or slice. If the component
        is indexed, an IndexedComponent_slice is returned. Otherwise, a
        ComponentData is returned.

    """
    if type(slice_) is IndexedComponent_slice:
        context_slice = slice_.duplicate()
    else:
        context_slice = None

    # If active argument is specified and does not match the block's
    # active flag, we return immediately. This matches the behavior of
    # component_objects. We only need this check as we may modify the
    # active argument sent to component_objects if ctype is not an
    # ActiveComponent type.
    if active is not None and active != b.active:
        return

    # Define this class so we don't have to call issubclass again later.
    check_active = issubclass(ctype, ActiveComponent) and (active != None)

    # If active=False and ctype is not an ActiveComponent (e.g. it is Var)
    # we will not generate any components. To prevent this, only pass the
    # active argument if we are looking for active components.
    c_active = active if check_active else None

    # Looks for components indexed by specified sets immediately in our block.
    for c in b.component_objects(ctype, descend_into=False, active=c_active):
        subsets = list(c.index_set().subsets())
        new_sets = [s for s in subsets if s in sets]
        # Extend our "index stack"
        sliced_sets = index_stack + new_sets

        # Extend our slice with this component
        for idx, new_slice in slice_component_along_sets(
            c, sets, context_slice=context_slice, normalize=False
        ):
            # If we have to check activity, check data objects defined by
            # slice. If any match, we yield the slice. This is done for
            # compatibility with the behavior when slicing blocks, where
            # we can only descend into a block that matches our active flag.
            #
            # Note that new_slice can be a data object. This happens if the
            # component doesn't contain any sets we are slicing, i.e. new_sets
            # is empty.
            if (
                # Yield if (a) we're not checking activity
                not check_active
                # or (b) we have not sliced and data object activity matches
                or (not sliced_sets and new_slice.active == c_active)
                # or (c) we did slice and *any* data object activity matches
                or (
                    sliced_sets
                    and any(data.active == c_active for data in new_slice.duplicate())
                )
            ):
                yield sliced_sets, new_slice

    # We now descend into subblocks
    for sub in b.component_objects(Block, descend_into=False, active=active):
        subsets = list(sub.index_set().subsets())
        new_sets = [s for s in subsets if s in sets]

        # Extend stack with new matched indices.
        index_stack.extend(new_sets)

        # Need to construct an index to descend into for each slice-of-block
        # we are about generate.
        # Note that any remaining _NotAnIndex placeholders after this loop
        # will be replaced with the corresponding indices of the non-sliced
        # sets.
        given_descend_idx = [_NotAnIndex for _ in subsets]
        for i, s in enumerate(subsets):
            # NOTE: index_map better only contain sets that we are slicing.
            if s in index_map:
                # Use a user-given index if available.
                given_descend_idx[i] = index_map[s]
                if s not in sets:
                    raise RuntimeError(
                        "Encountered a specified index for a set %s that we"
                        " are not slicing. This is not supported" % s
                    )
            elif s in sets:
                # Otherwise use a slice. We will advanced the slice iter
                # to try to get a concrete component from this slice.
                given_descend_idx[i] = get_slice_for_set(s)

        # Generate slices from this sub-block
        for idx, new_slice in slice_component_along_sets(
            sub, sets, context_slice=context_slice, normalize=False
        ):
            # TODO: Can this branch happen outside of the loop?
            # If it's not indexed, we don't need to slice...
            if sub.is_indexed():
                # fill any remaining placeholders with the "index" of our slice
                descend_idx = _fill_indices(list(given_descend_idx), idx)
                # create a slice-or-data object
                descend_data = sub[descend_idx]
                if type(descend_data) is IndexedComponent_slice:
                    try:
                        slice_iter = iter(descend_data)
                        # Try to find a data object defined by the slice
                        # that matches the active argument. In doing so,
                        # we treat a slice as inactive if all of its data
                        # objects are inactive. We need to find a data obj
                        # with the correct active flag, otherwise we run into
                        # problems when we descend (component_objects will
                        # not yield anything).
                        _data = next(slice_iter)
                        while active is not None and _data.active != active:
                            _data = next(slice_iter)
                        descend_data = _data
                    except StopIteration:
                        # For this particular idx, we have no BlockData
                        # to descend into.
                        continue
                elif active is not None and descend_data.active != active:
                    # descend_data is a BlockData object. This particular
                    # BlockData was specified by the index map. In this case,
                    # we want to respect "activity".
                    continue
            else:
                # Have encountered a ScalarBlock. Do not need to check the
                # active flag as this came straight from component_objects.
                descend_data = sub

            # Recursively generate sliced components from this data object
            for st, v in generate_sliced_components(
                descend_data,
                index_stack,
                new_slice,
                sets,
                ctype,
                index_map,
                active=active,
            ):
                yield tuple(st), v

        # pop the index sets of the block whose sub-components
        # we just finished iterating over.
        for _ in new_sets:
            index_stack.pop()


def flatten_components_along_sets(m, sets, ctype, indices=None, active=None):
    """This function iterates over components (recursively) contained
    in a block and partitions their data objects into components
    indexed only by the specified sets.

    Parameters
    ----------

    m: BlockData
        Block whose components (and their sub-components) will be
        partitioned

    sets: Tuple of Pyomo Sets
        Sets to be sliced. Returned components will be indexed by
        some combination of these sets, if at all.

    ctype: Subclass of Component
        Type of component to identify and partition

    indices: Iterable or ComponentMap
        Indices of sets to use when descending into subblocks. If an
        iterable is provided, the order corresponds to the order in
        ``sets``. If a ``ComponentMap`` is provided, the keys must be
        in ``sets``.

    active: Bool or None
        If not None, this is a boolean flag used to filter component objects
        by their active status. A reference-to-slice is returned if any data
        object defined by the slice matches this flag.

    Returns
    -------

    List of tuples of Sets, list of lists of Components
        The first entry is a list of tuples of Pyomo Sets. The second is a
        list of lists of Components, indexed by the corresponding sets in
        the first list. If the components are unindexed, ComponentData are
        returned and the tuple of sets contains only UnindexedComponent_set.
        If the components are indexed, they are references-to-slices.

    """
    set_of_sets = ComponentSet(sets)
    if indices is None:
        index_map = ComponentMap()
    elif type(indices) is ComponentMap:
        index_map = indices
    else:
        index_map = ComponentMap(zip(sets, indices))
    for s, idx in index_map.items():
        if idx not in s:
            raise ValueError(
                "%s is a bad index for set %s. \nPlease provide an index "
                "that is in the set." % (idx, s.name)
            )
        if s not in set_of_sets:
            raise RuntimeError(
                "Index specified for set %s that is not one of the sets"
                " that will be sliced. Indices should only be provided"
                " for sets that will be sliced." % s.name
            )
    index_stack = []

    # Using these two `OrderedDict`s is a workaround because I can't
    # reliably use tuples of components as keys in a `ComponentMap`.
    sets_dict = OrderedDict()
    comps_dict = OrderedDict()
    for index_sets, slice_ in generate_sliced_components(
        m, index_stack, m, set_of_sets, ctype, index_map, active=active
    ):
        # Note that index_sets should always be a tuple, never a scalar.

        # TODO: Potentially re-order sets at this point.
        # In this way (time, space) would have the same key as (space, time).
        # They we'd have to somehow "swap indexing sets" when we create
        # the reference below.
        key = tuple(id(c) for c in index_sets)
        if key not in sets_dict:
            if len(key) == 0:
                sets_dict[key] = (UnindexedComponent_set,)
            else:
                sets_dict[key] = index_sets
        if key not in comps_dict:
            comps_dict[key] = []
        if len(key) == 0:
            comps_dict[key].append(slice_)
        else:
            # If the user wants to change these flags, they can access the
            # slice via the `referent` attribute of each reference component.
            slice_.attribute_errors_generate_exceptions = False
            slice_.key_errors_generate_exceptions = False
            comps_dict[key].append(Reference(slice_))

    # list-of-tuples of Sets:
    sets_list = list(sets for sets in sets_dict.values())
    # list-of-lists of components:
    comps_list = list(comps for comps in comps_dict.values())
    # E.g. we return: (
    #          [(time, space), (time,)],
    #          [[some_component, ...], [other, ...]],
    #      )                            ^ These components are indexed by time
    #            ^ These components are indexed by time and space
    return sets_list, comps_list


def flatten_dae_components(model, time, ctype, indices=None, active=None):
    """Partitions components into ComponentData and Components indexed only
    by the provided set.

    Parameters
    ----------

    model: BlockData
        Block whose components are partitioned

    time: Set
        Indexing by this set (and only this set) will be preserved in the
        returned components.

    ctype: Subclass of Component
        Type of component to identify, partition, and return

    indices: Tuple or ComponentMap
        Contains the index of the specified set to be used when descending
        into blocks

    active: Bool or None
        If provided, used as a filter to only return components with the
        specified active flag. A reference-to-slice is returned if any
        data object defined by the slice matches this flag.

    Returns
    -------
    List of ComponentData, list of Component
        The first list contains ComponentData for all components not
        indexed by the provided set. The second contains references-to
        -slices for all components indexed by the provided set.

    """
    target = ComponentSet((time,))
    sets_list, comps_list = flatten_components_along_sets(
        model, target, ctype, indices=indices, active=active
    )
    # Initialize these variables as, if no components of either category are
    # found, we expect to get an empty list.
    scalar_comps = []
    dae_comps = []
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is time:
            dae_comps = comps
        elif len(sets) == 0 or (len(sets) == 1 and sets[0] is UnindexedComponent_set):
            scalar_comps = comps
        else:
            raise RuntimeError(
                "Invalid model for `flatten_dae_components`.\n"
                "This can happen if your model has components that are\n"
                "indexed by time (explicitly or implicitly) multiple times."
            )
    return scalar_comps, dae_comps
