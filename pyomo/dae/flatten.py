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


def generate_time_only_slices(obj, time):
    o_sets = obj.index_set().subsets()
    # Given a potentially complex set, determine the index of the TIME
    # set, as well as all other "fixed" indices.  We will even support a
    # single Set with dimen==None (using ellipsis in the slice).
    ellipsis_idx = None
    time_idx = None
    regular_idx = []
    idx = 0
    for s in o_sets:
        if s is time:
            time_idx = idx
            idx += 1
        elif s.dimen is not None:
            for sub_idx in range(s.dimen):
                regular_idx.append(idx+sub_idx)
            idx += s.dimen
        elif ellipsis_idx is None:
            ellipsis_idx = idx
            idx += 1
        else:
            raise RuntimeError(
                "We can only handle a single Set with dimen=None")
    # To support Sets with dimen==None (using ellipsis), we need to have
    # all fixed/time indices be positive if they appear before the
    # ellipsis and negative (counting from the end of the list) if they
    # are after the ellipsis.
    if ellipsis_idx:
        if time_idx > ellipsis_idx:
            time_idx = time_idx - idx
        regular_idx = [ i - idx if i > ellipsis_idx else i
                      for i in fixed_idx ]
    # We now form a temporary slice that slices over all the regular
    # indices for a fixed value of the time index.
    tmp_sliced = {i: slice(None) for i in regular_idx}

    # Need to choose some arbitrary time index to fix, but indices
    # may have been skipped, so choosing e.g. time.first() is not
    # reliable. Approach here is to take the first index we encounter
    # when iterating over obj.
    for idx in obj:
        if idx.__class__ is not tuple:
            idx = (idx,)
        t = idx[time_idx]
        break
    tmp_fixed = {time_idx: t}

    tmp_ellipsis = ellipsis_idx
    _slice = IndexedComponent_slice(
        obj, tmp_fixed, tmp_sliced, tmp_ellipsis
    )
    # For each combination of regular indices, we can generate a single
    # slice over the time index
    time_sliced = {time_idx: slice(None)}
    for key in _slice.wildcard_keys():
        if type(key) is not tuple:
            key = (key,)
        time_fixed = dict(
            (i, val) if i<time_idx else (i+1, val)
            for i,val in enumerate(key)
        )
        yield IndexedComponent_slice(obj, time_fixed, time_sliced, None)


def generate_time_indexed_block_slices(block, time, ctype):
    # TODO: We should probably do a sanity check that time does not
    # appear in any sub-block / var indices.
    queue = list( generate_time_only_slices(block, time) )
    while queue:
        _slice = queue.pop(0)
        # Pick a random block from this slice (i.e. TIME == TIME.first())
        #
        # TODO: we should probably check that the OTHER blocks
        # in the time set have the same variables.
        b = next(iter(_slice))
        # Any sub-blocks must be put on the queue to descend into and
        # process
        for sub_b in b.component_objects(Block, descend_into=False):
            _name = sub_b.local_name
            for idx in sub_b:
                queue.append(_slice.component(_name)[idx])
        # Any Vars must be mapped to slices and returned
        for v in b.component_objects(ctype, descend_into=False):
            _name = v.local_name
            for idx in v:
                yield _slice.component(_name)[idx]
        

#def flatten_dae_components(model, time, ctype):
#    """
#    This function takes in a (hierarchical, block-structured) Pyomo
#    model and a `ContinuousSet` and returns two lists of "flattened"
#    components. The first is a list of all `_ComponentData` that are not
#    indexed by the `ContinuousSet` and the second is a list of
#    `Reference` components such that each reference is indexed only by
#    the specified `ContinuousSet`. This function is convenient for
#    identifying components that are implicitly indexed by the
#    `ContinuousSet`, for example, a singleton `Component` living on a 
#    `Block` that is indexed by the `ContinuousSet`.
#
#    Parameters
#    ----------
#    model : Concrete Pyomo model
#
#    time : ``pyomo.dae.ContinuousSet``
#
#    ctype : Pyomo Component type
#
#    Returns
#    -------
#    Two lists
#    """
#    assert time.model() is model.model()
#
#    block_queue = [model]
#    regular_comps = []
#    time_indexed_comps = []
#    while block_queue:
#        b = block_queue.pop(0)
#        b_sets = b.index_set().subsets()
#        if time in b_sets:
#            for _slice in generate_time_indexed_block_slices(b, time, ctype):
#                time_indexed_comps.append(Reference(_slice))
#            continue
#        for blkdata in b.values():
#            block_queue.extend(
#                blkdata.component_objects(Block, descend_into=False)
#            )
#        for blkdata in b.values():
#            for v in blkdata.component_objects(SubclassOf(ctype), 
#                    descend_into=False):
#                v_sets = v.index_set().subsets()
#                if time in v_sets:
#                    for _slice in generate_time_only_slices(v, time):
#                        time_indexed_comps.append(Reference(_slice))
#                else:
#                    regular_comps.extend(v.values())
#
#    return regular_comps, time_indexed_comps

def get_slice_for_set(s):
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
        return None

_UNINDEXED_INDEX = next(iter(UnindexedComponent_set))

class _NotAnIndex(object):
    """ 
    `None` is a valid index, so we use a dummy class to 
    denote a slot that needs to get filled with indices
    from our product.
    """
    pass

def _fill_indices(filled_index, index):
    # We need to generate a new index for every entry of `product`,
    # and want to reuse `partial_index_list` as a starting point,
    # so we copy it here.
    j = 0
    for i, val in enumerate(filled_index):
        if val is _NotAnIndex:
            filled_index[i] = index[j]
            # `index` is always a tuple, so this is valid
            j += 1
    # Make sure `partial_index_list` has the same number of vacancies
    # as `product` has factors. Not _strictly_ necessary.
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
    slot will get filled in by an entry from `setprod`.

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
            #
# TODO: Remove this code
#            # To simplify some later code we convert scalar to tuple.
#            if type(index) is not tuple:
#                # NOTE: I don't think I ever get here.
#                # TODO: test this.
#                # This would not be reliable as it could not distinguish
#                # between ('a', 1) and (('a', 1),)
#                # NOTE: My tests pass without this check and conversion
#                index = (index,)
#            # We need to generate a new index for every entry of `product`,
#            # and want to reuse `partial_index_list` as a starting point,
#            # so we copy it here.
#            filled_index = partial_index_list.copy()
#            j = 0
#            for i, val in enumerate(filled_index):
#                if val is _NotAnIndex:
#                    filled_index[i] = index[j]
#                    # `index` is always a tuple, so this is valid
#                    j += 1
#            # Make sure `partial_index_list` has the same number of vacancies
#            # as `product` has factors. Not _strictly_ necessary.
#            assert j == len(index)
#            filled_index = tuple(filled_index)
#

            filled_index = partial_index_list.copy()

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

#            if len(filled_index) == 1:
#                yield filled_index[0]
#            else:
#                yield filled_index

            yield _fill_indices(filled_index, index)

            normalize_index.flatten = False
            # Want to get the unflattened factors when we advance the
            # `product` iterator.
    finally:
        # Reset `normalize_index.flatten`
        normalize_index.flatten = _normalize_index_flatten

def generate_sliced_components(b, index_stack, _slice, sets, ctype, index_map):
    """
    `b` is a _BlockData object.

    `index_stack` is a list of indices "above" `b` in the
    hierarchy. Note that `b` is a data object, so any index
    of its parent component should be included in the stack.

    `_slice` is the slice generated so far. Our goal here is to
    yield extensions to `_slice` at this level of the hierarchy.

    `ctype` is the type we are looking for.

    `index_map` is potentially a map from each set in `sets` to a "representative
              index" to use if we ever have
    """
    for c in b.component_objects(ctype, descend_into=False):
        subsets = list(c.index_set().subsets())
        temp_idx = [get_slice_for_set(s) if s in sets else _NotAnIndex
                for s in subsets]
        new_sets = [s for s in subsets if s in sets]
        other_sets = [s for s in subsets if s not in sets]
        sliced_sets = index_stack + new_sets

        c_is_indexed = not (len(other_sets) == 1 and
                other_sets[0] is UnindexedComponent_set)
        # Why not just c.is_indexed()

        # We have extended our "index stack;" now we must extend
        # our slice.

        if other_sets and c_is_indexed:
            cross_prod = other_sets[0].cross(*other_sets[1:])
            # The original implementation was to pick an arbitrary index
            # from the "flattened sets" and slice all the other indices.

            for new_index in _fill_indices_from_product(temp_idx, cross_prod):
                try:
                    c_slice = getattr(_slice, c.local_name)[new_index]
                    if type(c_slice) is IndexedComponent_slice:
                        # This is just to make sure we do not have an
                        # empty slice.
                        next(iter(c_slice.duplicate()))
                    yield sliced_sets, c_slice
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
            # `c` is indexed only by sets we would like to slice.
            # At this point we could just yield sliced_sets, new_slice
            idx = Ellipsis if c_is_indexed else None
            # TODO: next(UnindexedComponent_set) instead of None
            c_slice = getattr(_slice, c.local_name)[idx]
            # ^ This could fail if we try to apply the same code to
            # data objects rather than just simple components.
            yield sliced_sets, c_slice

    # We now descend into subblocks
    for sub in b.component_objects(Block, descend_into=False):
        subsets = list(sub.index_set().subsets())
        # TODO: What if one of the user's `sets` is a SetProduct?
        # Should probably expand it before entering this generator.
        temp_idx = [get_slice_for_set(s) if s in sets else _NotAnIndex
                for s in subsets]
        new_sets = [s for s in subsets if s in sets]
        other_sets = [s for s in subsets if s not in sets]

        # For each set we are slicing, if the user specified an index, put it
        # here. Otherwise, slice the set and we will call next(iter(_slice))
        # once the full index is constructed.
        descend_index_sliced_sets = tuple(index_map[s] if s in index_map else 
                get_slice_for_set(s) for s in new_sets)

        # Extend stack with new matched indices.
        index_stack.extend(new_sets)

        is_indexed = sub.is_indexed()

        if other_sets and is_indexed:
            cross_prod = other_sets[0].cross(*other_sets[1:])

            for new_index in _fill_indices_from_product(temp_idx, cross_prod):
                try:
                    sub_slice = getattr(_slice, sub.local_name)[new_index]
                    if type(sub_slice) is IndexedComponent_slice:
                        # Get the first index of the sliced subblock to descend
                        # into. TODO: Should probably allow some "representative
                        # index" to be specified for each set so we don't miss
                        # components that are skipped at endpoints.
                        #
                        # Need a function to replace slices in new_index with
                        # indices from temp_idx...
#                        ref = Reference(sub_slice.duplicate(), ctype=ctype)
                        # Need to get the index I want for ref.
                        # Should use the indices from `cross_prod`, plus the
                        # indices from my reference map. The former are contained
                        # in `new_index`. The latter need to be obtained for s
                        # in sets, then put in the right place in new_index.
                        tupl_new_index = (new_index,) if type(new_index) \
                                is not tuple else new_index
                        incomplete_descend_index = list(
                                idx if subset not in sets else _NotAnIndex
                                for idx, subset in zip(tupl_new_index, subsets)
                                )
                        descend_index = _fill_indices(incomplete_descend_index,
                                descend_index_sliced_sets)

                        descend_slice = sub[descend_index]
                        data = descend_slice if type(descend_slice) is not \
                            IndexedComponent_slice else next(iter(descend_slice))
                        # If the user has supplied enough indices that we can
                        # descend into a concrete component, we do so. Otherwise
                        # we use the user's indices, slice the rest, and advance
                        # the iterator.

                        # TODO: try/except
                        # What is the risk here? sub_slice is "empty."
                        # This is definitely possible if the block is (somehow)
                        # skipped at some index of `cross_prod`. This should
                        # be rare though.
                        #
                        # And if new_index is concrete, we will just get keyerrors
                        # for the indices we want to skip
                    else:
                        # sub_slice is a block data object
                        data = sub_slice
                    for st, v in generate_sliced_components(data, index_stack,
                            sub_slice, sets, ctype, index_map):
                        yield tuple(st), v
                # TODO: Make sure I can actually encounter these exceptions.
                # Should I have an option to raise these errors?
                except StopIteration:
                    # Empty slice due to "skipped" index in subblock.
                    pass
                except KeyError:
                    # Trying to access a concrete data object for a "skipped" index.
                    pass
        else:
            # Either `sub` is a simple component, or we are slicing
            # all of its sets.
            #
            # In this case it may be unnecessary to slice at all...
            #
            sub_slice = getattr(_slice, sub.local_name)[...] if new_sets else \
                    getattr(_slice, sub.local_name)
                    # Don't use __getitem__(None) to avoid unnecessary additions
                    # to the call stack.
            #try:
            if type(sub_slice) is IndexedComponent_slice:
                # Get an arbitrary BlockData to descend into.
                # Should be able to use a supplied index here...
                # TODO: access this component at the specified index.
                data = next(iter(sub_slice.duplicate()))
            else:
                # `sub_slice` is a simple component
                data = sub_slice
            for st, v in generate_sliced_components(data, index_stack,
                    sub_slice, sets, ctype, index_map):
                yield tuple(st), v
            # TODO: Make sure we can actually encounter this exception.
            #except StopIteration:
            #    # We encountered an empty slice.
            #    pass

        # pop the index sets of the block whose sub-components
        # we just finished iterating over.
        for _ in new_sets:
            index_stack.pop()

# NOTE: There are several cases of "skipped component" that we must be aware of:
#     - Explicitly skipped _ConstraintData
#       If the skipped indices are "within slices" (e.g. time.first()),
#       we simply must be careful to not access these references at the
#       skipped indices, probably with try/except.
#       If the skipped indices are "outside slices," we will simply not
#       generate slices for these indices. (This raises StopIteration above.)
#     - Skipped _BlockData ? Should be more rare, as blocks are dense by default (?)
#     - Implicitly skipped components on certain indices of subblocks.
#       These appear to be relatively common in practice.
#       I think the only remedy here is to control what index of sliced blocks
#       we actually descend into...
#       If we had a "templatized representation" of the block, then we may not
#       need to choose...

def flatten_components_along_sets(m, sets, ctype, indices=None, index_stack=None):
    """
    """
    if index_stack is None:
        index_stack = []

    if indices is None:
        index_map = ComponentMap()
    else:
        index_map = ComponentMap(zip(sets, indices))

    set_of_sets = ComponentSet(sets)
    # Using these two `OrderedDict`s is a workaround because I can't
    # reliably use tuples of components as keys in a `ComponentMap`.
    sets_dict = OrderedDict()
    comps_dict = OrderedDict()
    for index_sets, _slice in generate_sliced_components(m, index_stack,
            m, set_of_sets, ctype, index_map):
        # Note that index_sets should always be a tuple, never a scalar

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
            comps_dict[key].append(_slice)
        else:
            _slice.attribute_errors_generate_exceptions = False
            _slice.key_errors_generate_exceptions = False
            comps_dict[key].append(Reference(_slice))
    # list-of-tuples of Sets:
    sets_list = list(sets for sets in sets_dict.values())
    # list-of-lists of components:
    comps_list = list(comps for comps in comps_dict.values())
    return sets_list, comps_list

def flatten_dae_components(model, time, ctype):
    target = ComponentSet((time,))
    sets_list, comps_list = flatten_components_along_sets(model, target, ctype)
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
