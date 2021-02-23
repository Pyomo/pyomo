#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import six

from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
from pyomo.common.collections import ComponentSet, ComponentMap

def _to_iterable(source):
    iterable_scalars = six.string_types + (six.binary_type, six.text_type)
    if hasattr(source, '__iter__'):
        if isinstance(source, iterable_scalars):
            yield source
        else:
            for obj in source:
                yield obj
    else:
        yield source

def get_component_call_stack(comp, context=None):
    """Get the call stack necessary to locate a `Component`

    The call stack is a `list` of `tuple`s where the first entry is a 
    code for `__getattr__` or  `__getitem__`, using the same convention
    as `IndexedComponent_slice`. The second entry is the argument of
    the corresponding function. Following this sequence of calls from
    `context` (or the top-level model if `context is None`) will
    produce comp.

    Parameters:
    -----------
    comp : `pyomo.core.base.component.Component`
        The component to locate
    context : `pyomo.core.base.block.Block`
        The block within which to locate the component. If `None`, the
        top-level model will be used.

    Returns:
    --------
    `list` : Contains the necessary method calls and their arguments.
             Note that the calls should be applied in reverse order.
             This is the opposite direction as in IndexedComponent_slice.

    """
    # If comp is context, an empty call stack is returned.
    # A component is not said to exist in "the context of" itself.
    call_stack = []
    while comp.parent_block() is not None:
        # If parent_block is None, comp is the root 
        # of the model, so we don't add anything else
        # to the call stack.
        if comp is context:
            # We are done
            break

        parent_component = comp.parent_component()
        # Add (get_item, index) to the call stack
        if parent_component.is_indexed() and parent_component is not comp:
            # I.e. `comp` is a data object
            call_stack.append((IndexedComponent_slice.get_item, comp.index()))

        if parent_component is context:
            # We are done
            break

        # Add (get_attribute, name) to the call stack
        call_stack.append((
            IndexedComponent_slice.get_attribute,
            parent_component.local_name
            ))
        comp = comp.parent_block()
    return call_stack

def slice_component_along_sets(comp, sets, context=None):
    """Slice a component along the indices corresponding to some sets,
    wherever they appear in the component's block hierarchy.

    Given a component or component data object, for all parent components
    and parent blocks between the object and the `context` block, replace
    any index corresponding to a set in `sets` with slices or an 
    ellipsis.

    Parameters:
    -----------
    comp: `pyomo.core.base.component.Component` or 
    `pyomo.core.base.component.ComponentData`
        Component whose parent structure to search and replace
    sets: `pyomo.common.collections.ComponentSet`
        Contains the sets to replace with slices
    context: `pyomo.core.base.block.Block` or
    `pyomo.core.base.block._BlockData`
        Block below which to search for sets

    Returns:
    --------
    `pyomo.core.base.indexed_component_slice.IndexedComponent_slice`:
        Slice of `comp` with wildcards replacing the indices of `sets`    

    """
    if context is None:
        context = comp.model()
    call_stack = get_component_call_stack(comp, context)

    # Maintain a pointer to the component so we can get
    # the index set and know which locations to slice.
    comp = context
    sliced_comp = context
    while call_stack:
        call, arg = call_stack.pop()
        if call is IndexedComponent_slice.get_attribute:
            comp = getattr(comp, arg)
            sliced_comp = getattr(sliced_comp, arg)
        elif call is IndexedComponent_slice.get_item:
            index_set = comp.index_set()
            comp = comp[arg]
            # Process arg to replace desired indices with slices.
            location_set_map = get_location_set_map(arg, index_set)
            arg = replace_indices(arg, location_set_map, sets)
            sliced_comp = sliced_comp[arg]

    return sliced_comp

def replace_indices(index, location_set_map, sets):
    """Use `location_set_map` to replace values in `index` with slices
    or an Ellipsis.

    Parameters:
    -----------
    index: `tuple` or scalar
        Index whose values to replace
    location_set_map: `dict`
        Maps locations ("indices") within the index to their
        corresponding set
    sets: `pyomo.common.collections.ComponentSet`
        Contains the sets to replace with slices

    Returns:
    --------
    `tuple`: Index with values replaced by slices

    """
    index = tuple(_to_iterable(index))
    new_index = []
    loc = 0
    len_index = len(index)
    while loc < len_index:
        val = index[loc]
        _set = location_set_map[loc]
        dimen = _set.dimen
        if _set not in sets:
            new_index.append(val)
        elif dimen is not None:
            new_index.append(slice(None, None, None))
        else:
            dimen_none_set = _set
            new_index.append(Ellipsis)
            loc += 1
            while loc < len_index:
                # Skip all adjacent locations belonging to the same
                # set. These are covered by the Ellipsis.
                _set = location_set_map[loc]
                if _set is not dimen_none_set:
                    break
                loc += 1
            continue
        loc += 1
    return tuple(new_index)

def get_location_set_map(index, index_set):
    """Map each value in an index to the set from which it originates

    This function iterates over the "subsets" of `index_set` in the
    forward direction, assigning sets to each value in `index`, until
    it finds a set of dimension `None`. Then it iterates over the
    reversed list of subsets, assigning sets, until it encounters
    the same set of dimension `None`. All remaining values are assigned
    to the set of dimension `None`. If a second such set is found,
    an error is raised.

    Parameters:
    -----------
    index: `tuple` or hashable scalar
        The index whose values will be assigned to sets
    index_set: `pyomo.core.base.set.SetProduct` or `pyomo.core.base.set.Set`
        The index set from which `index` originates

    Returns:
    --------
    `dict`: Maps the "locations" (indices) within the index to the set
        from which it originates.

    """
    index = tuple(_to_iterable(index))
    len_index = len(index)
    locations_left = set(range(len_index))
    location_set_map = {}

    # Determine if we should exit early
    if index_set is UnindexedComponent_set:
        # index == (None,) and dimen is None in this case,
        # so without this catch, the function would return
        # {0: None}, which is not what we want for an
        # unindexed component.
        return {0: UnindexedComponent_set}
    elif not normalize_index.flatten:
        raise ValueError(
            'get_location_set_map does not support the case where '
            'normalize_index.flatten is False.'
            # Although in this case, the location of an index should
            # just be its position in the subsets list, so maybe
            # the info we need is actually more simple to obtain.
            )

    subsets = list(index_set.subsets())

    n_subsets = len(subsets)

    location = 0
    dimen_none_set = None
    dimen_none_set_coord = None
    # Step through subsets in forward order to assign as many
    # locations as possible.
    for sub_coord, sub in enumerate(subsets):
        dimen = sub.dimen
        if dimen is None:
            dimen_none_set = sub
            dimen_none_set_coord = sub_coord
            break
        for i in range(dimen):
            location_set_map[location+i] = sub
            locations_left.remove(location+i)
        location += dimen

    # We are either done or have encountered a set of dimen None
    if not locations_left:
        return location_set_map

    location = len_index - 1
    # Step through subsets in reverse order, assigning locations,
    # until a set of dimen==None is encountered.
    for sub_coord, sub in enumerate(reversed(subsets)):
        sub_coord = n_subsets - sub_coord - 1
        dimen = sub.dimen
        if dimen is None:
            if dimen_none_set_coord != sub_coord:
                # Make sure this set is the same one we encountered
                # earlier. It is sufficient to check the coordinate.
                raise RuntimeError(
                'Cannot get locations when multiple sets of dimen==None '
                'are present.'
                '\nFound %s at position %s and %s at position %s.'
                '\nLocation is ambiguous in this case.'
                % (dimen_none_set, dimen_none_set_coord, sub, sub_coord)
                )
            break
        for i in range(dimen):
            location_set_map[location-i] = sub
            locations_left.remove(location-i)
        location -= sub.dimen

    for loc in locations_left:
        # All remaining locations, that cannot be accessed from some 
        # constant offset from the beginning or end of the tuple,
        # must belong to the dimen-None set.
        location_set_map[loc] = dimen_none_set

    return location_set_map
