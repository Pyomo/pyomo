#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.core.base import Block, Reference
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.block import SubclassOf
from pyomo.core.base.set import SetProduct
from pyomo.core.base.indexed_component import (
        UnindexedComponent_set,
        normalize_index,
        )
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
                return (slice(None),)*s.dimen
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


def slice_component_along_sets(
        component, sets, context_slice=None, normalize=None,
        ):
    """
    This function generates all possible slices of the provided component
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
    temp_idx = [get_slice_for_set(s) if s in set_set else _NotAnIndex
            for s in subsets]
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

        for prod_index, new_index in _fill_indices_from_product(
                temp_idx,
                cross_prod,
                ):
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
                    temp_slice = c_slice.duplicate()
                    next(iter(temp_slice))
                if ((normalize is None and normalize_index.flatten)
                        or normalize):
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


def generate_sliced_components(b, index_stack, slice_, sets, ctype, index_map):
    """
    Recursively generate sliced components of a block and its subblocks, along
    with the sets that were sliced for each component.

    `b` is a _BlockData object.

    `index_stack` is a list of indices "above" `b` in the
    hierarchy. Note that `b` is a data object, so any index
    of its parent component should be included in the stack.

    `slice_` is the slice generated so far. Our goal here is to
    yield extensions to `slice_` at this level of the hierarchy.

    `sets` is a ComponentSet of Pyomo sets that should be sliced.

    `ctype` is the type we are looking for.

    `index_map` is potentially a map from each set in `sets` to a 
    "representative index" to use when descending into subblocks.
    """
    if type(slice_) is IndexedComponent_slice:
        context_slice = slice_.duplicate()
    else:
        context_slice = None

    # Looks for components indexed by these sets immediately in our block
    for c in b.component_objects(ctype, descend_into=False):
        subsets = list(c.index_set().subsets())
        new_sets = [s for s in subsets if s in sets]
        # Extend our "index stack"
        sliced_sets = index_stack + new_sets

        # Extend our slice with this component
        for idx, new_slice in slice_component_along_sets(
                c, sets, context_slice=context_slice, normalize=False
                ):
            yield sliced_sets, new_slice

    # We now descend into subblocks
    for sub in b.component_objects(Block, descend_into=False):
        subsets = list(sub.index_set().subsets())
        new_sets = [s for s in subsets if s in sets]

        # Extend stack with new matched indices.
        index_stack.extend(new_sets)

        # Need to construct an index to descend into for each slice-of-block
        # we are about generate.
        given_descend_idx = [_NotAnIndex for _ in subsets]
        for i, s in enumerate(subsets):
            if s in index_map:
                # Use a user-given index if available
                given_descend_idx[i] = index_map[s]
            elif s in sets:
                # Otherwise use a slice. We will advanced the slice iter
                # to try to get a concrete component from this slice.
                given_descend_idx[i] = get_slice_for_set(s)

        # Generate slices from this sub-block
        for idx, new_slice in slice_component_along_sets(
                sub, sets, context_slice=context_slice, normalize=False
                ):
            if sub.is_indexed():
                # fill any remaining placeholders with the "index" of our slice
                descend_idx = _fill_indices(list(given_descend_idx), idx)
                # create a slice-or-data object
                descend_data = sub[descend_idx]
                if type(descend_data) is IndexedComponent_slice:
                    try:
                        # Attempt to find a data object matching this slice
                        descend_data = next(iter(descend_data))
                    except StopIteration:
                        # For this particular idx (and given indices), no
                        # block data object exists to descend into.
                        # Not sure if we should raise an error here... -RBP
                        continue
            else:
                descend_data = sub
            
            # Recursively generate sliced components from this data object
            for st, v in generate_sliced_components(
                    descend_data, index_stack, new_slice, sets, ctype, index_map
                    ):
                yield tuple(st), v

        # pop the index sets of the block whose sub-components
        # we just finished iterating over.
        for _ in new_sets:
            index_stack.pop()


def flatten_components_along_sets(m, sets, ctype, indices=None):
    """
    This function iterates over components (recursively) contained
    in a block and partitions their data objects into components
    indexed only by the specified sets.

    Args:
        m : Block whose components (and their sub-components) will be
            partitioned
        sets : Possible indexing sets for the returned components
        ctype : Type of component to identify and partition
        indices : indices of sets to use when descending into subblocks

    Returns:
        tuple: The first entry is a list of tuples of Pyomo Sets. The
               second is a list of lists of components, each indexed by
               the corresponding sets in the first entry.
        
    """
    if indices is None:
        index_map = ComponentMap()
    elif type(indices) is ComponentMap:
        index_map = indices
    else:
        index_map = ComponentMap(zip(sets, indices))
    for s, idx in index_map.items():
        if not idx in s:
            raise ValueError(
                "%s is a bad index for set %s. \nPlease provide an index "
                "that is in the set." % (idx, s.name)
            )
    index_stack = []

    set_of_sets = ComponentSet(sets)
    # Using these two `OrderedDict`s is a workaround because I can't
    # reliably use tuples of components as keys in a `ComponentMap`.
    sets_dict = OrderedDict()
    comps_dict = OrderedDict()
    for index_sets, slice_ in generate_sliced_components(m, index_stack,
            m, set_of_sets, ctype, index_map):
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


def flatten_dae_components(model, time, ctype, indices=None):
    target = ComponentSet((time,))
    sets_list, comps_list = flatten_components_along_sets(model, target, ctype,
            indices=indices)
    # Initialize these variables as, if no components of either category are
    # found, we expect to get an empty list.
    scalar_comps = []
    dae_comps = []
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is time:
            dae_comps = comps
        elif len(sets) == 0 or (len(sets) == 1 and 
                sets[0] is UnindexedComponent_set):
            scalar_comps = comps
        else:
            raise RuntimeError(
                "Invalid model for `flatten_dae_components`.\n"
                "This can happen if your model has components that are\n"
                "indexed by time (explicitly or implicitly) multiple times."
                )
    return scalar_comps, dae_comps
